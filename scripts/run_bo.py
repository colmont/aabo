import sys
sys.path.append("../")
from svgp.model import GPModel
from utils.model_loader import get_inducing_points
import gpytorch
import wandb 
import torch
import hydra
from omegaconf import DictConfig, OmegaConf
import warnings
warnings.filterwarnings('ignore')
import os
os.environ["WANDB_SILENT"] = "True"
import signal 
from svgp.generate_candidates import generate_batch
from svgp.train_model import (
    update_model_elbo, 
    update_model_and_generate_candidates_eulbo,
)
from utils.setup import handle_interrupt, set_device, set_dtype, set_seed, set_wandb_tracker, validate_config 
from utils.data_loader import get_objective, get_random_init_data
from utils.turbo import TurboState, update_state
# for exact gp baseline: 
from botorch.models import SingleTaskGP
from botorch.fit import fit_gpytorch_mll
from gpytorch.mlls import ExactMarginalLogLikelihood

@hydra.main(config_path='../configs', config_name='conf')
def main(cfg: DictConfig):

    # Set-up 
    validate_config(cfg)
    set_seed(cfg.seed)
    set_dtype(cfg.float_dtype_as_int)
    set_device()
    tracker = set_wandb_tracker(cfg)
    handle_interrupt(tracker)
    INIT_TRAINING_COMPLETE = False

    # Obtain random initial training data
    objective = get_objective(cfg.task_id)
    train_x, train_y = get_random_init_data(
        task_id=cfg.task_id,
        objective=objective,
        num_initialization_points=cfg.num_initialization_points,
        init_mol_tasks_w_guacamol_data=cfg.init_mol_tasks_w_guacamol_data,
        update_on_n_pts=cfg.update_on_n_pts,
    )

    # Optional logging for training data shapes
    if cfg.verbose:
        print(f"train x shape: {train_x.shape}")
        print(f"train y shape: {train_y.shape}")

    # Get normalized train y
    init_train_y_mean, init_train_y_std = train_y.mean(), train_y.std()
    init_train_y_std = init_train_y_std or 1
    train_y_origscale = train_y
    if cfg.normalize_ys:
        train_y = (train_y - init_train_y_mean) / init_train_y_std

    # Initialize turbo state 
    tr_state = TurboState(
        dim=train_x.shape[-1],
        batch_size=cfg.bsz, 
        best_value=train_y_origscale.max().item(),
    )

    # Initialize exact GP model 
    if cfg.exact_gp_baseline:
        model = SingleTaskGP(
            train_x, 
            train_y, 
            covar_module=gpytorch.kernels.ScaleKernel(gpytorch.kernels.RBFKernel()),
            likelihood=gpytorch.likelihoods.GaussianLikelihood(),
        )
    # Initialize approximate GP model 
    else:
        inducing_points = get_inducing_points(
            train_x=train_x,
            objective=objective,
            inducing_pt_init_w_moss23=cfg.inducing_pt_init_w_moss23,
            n_inducing_pts=cfg.n_inducing_pts,
        )
        learn_inducing_locations = not cfg.moss23_baseline
        model = GPModel(
            inducing_points=inducing_points, 
            likelihood=gpytorch.likelihoods.GaussianLikelihood(),
            learn_inducing_locations=learn_inducing_locations,
        )

    # Main loop
    while objective.num_calls < cfg.max_n_oracle_calls:

        # Select all datapoints for first fit of the model
        if not INIT_TRAINING_COMPLETE:
            train_x_lastn = train_x
            train_y_lastn = train_y
            INIT_TRAINING_COMPLETE = True
        # Select last n datapoints for subsequent fits
        else:
            train_x_lastn = train_x[-cfg.update_on_n_pts:]
            train_y_lastn = train_y[-cfg.update_on_n_pts:]
        
        # Normalize train ys
        if cfg.normalize_ys:
            train_y = (train_y_origscale - init_train_y_mean) / init_train_y_std
            train_y_lastn = train_y[-cfg.update_on_n_pts:]

        # Update wandb with optimization progress
        best_score_found = train_y_origscale.max().item()
        n_calls_ = objective.num_calls
        if cfg.verbose:
            print(f"After {n_calls_} oracle calls, Best reward = {best_score_found}")
        # log data to wandb on each loop
        dict_log = {
            "best_found":best_score_found,
            "n_oracle_calls":n_calls_,
        }
        tracker.log(dict_log) 

        # Train exact GP model (until convergence)
        if cfg.exact_gp_baseline:
            model.set_train_data(train_x, train_y, strict=False)
            exact_gp_mll = ExactMarginalLogLikelihood(model.likelihood, model)
            fit_gpytorch_mll(exact_gp_mll)

        else:
            # EULBO: train for n epochs as warm-start
            if cfg.eulbo:
                n_epochs_elbo = cfg.n_warm_start_epochs
                train_to_convergence_elbo = False 
            # ELBO: train until convergence
            else:
                n_epochs_elbo = cfg.n_update_epochs
                train_to_convergence_elbo = True 

            # Train approximate GP model 
            update_model_dict = update_model_elbo(
                    model=model,
                    train_x=train_x_lastn,
                    train_y=train_y_lastn.squeeze(),
                    lr=cfg.lr,
                    n_epochs=n_epochs_elbo,
                    train_bsz=cfg.train_bsz,
                    grad_clip=cfg.grad_clip,
                    train_to_convergence=train_to_convergence_elbo,
                    max_allowed_n_failures_improve_loss=cfg.max_allowed_n_failures_improve_loss,
                    max_allowed_n_epochs=cfg.max_allowed_n_epochs,
                    moss23_baseline=cfg.moss23_baseline,
                    ppgpr=cfg.ppgpr,
                )
            model = update_model_dict["model"]

        # Generate a batch of candidates 
        x_next = generate_batch( 
            model=model,
            X=train_x,  
            Y=train_y,
            batch_size=cfg.bsz,
            acqf=cfg.acq_fun,
            absolute_bounds=(objective.lb, objective.ub),
            use_turbo=cfg.use_turbo,
            tr_length=tr_state.length,
        )

        # If using eulbo, use above model update and candidate generaiton as warm start
        if cfg.eulbo:
            lb = objective.lb
            ub = objective.ub
            update_model_dict = update_model_and_generate_candidates_eulbo(
                model=model,
                train_x=train_x_lastn,
                train_y=train_y_lastn.squeeze(),
                lb=lb,
                ub=ub,
                lr=cfg.lr,
                n_epochs=cfg.n_update_epochs,
                train_bsz=cfg.train_bsz,
                grad_clip=cfg.grad_clip,
                normed_best_f=train_y.max(),
                acquisition_bsz=cfg.bsz,
                max_allowed_n_failures_improve_loss=cfg.max_allowed_n_failures_improve_loss,
                max_allowed_n_epochs=cfg.max_allowed_n_epochs,
                init_x_next=x_next, 
                x_next_lr=cfg.x_next_lr,
                alternate_updates=cfg.alternate_eulbo_updates,
                num_kg_samples=cfg.num_kg_samples, 
                acq_fun=cfg.acq_fun,
                num_mc_samples_qei=cfg.num_mc_samples_qei,
                ablation1_fix_indpts_and_hypers=cfg.ablation1_fix_indpts_and_hypers,
                ablation2_fix_hypers=cfg.ablation2_fix_hypers,
                use_turbo=cfg.use_turbo,
                tr_length=tr_state.length,
                use_botorch_stable_log_softplus=cfg.use_botorch_stable_log_softplus,
                ppgpr=cfg.ppgpr,
            )
            model = update_model_dict["model"]
            x_next = update_model_dict["x_next"]

        # Evaluate candidates 
        y_next = objective(x_next)
        
        # Update data 
        train_x = torch.cat((train_x, x_next), dim=-2)
        train_y_origscale = torch.cat((train_y_origscale, y_next), dim=-2)

        # If running TuRBO, update trust region state 
        if cfg.use_turbo:
            tr_state = update_state(
                state=tr_state, 
                Y_next=y_next,
            )
            if tr_state.restart_triggered:
                tr_state = TurboState( 
                    dim=train_x.shape[-1],
                    batch_size=cfg.bsz, 
                    best_value=train_y_origscale.max().item(),
                )
        
    tracker.finish()

if __name__ == "__main__":
    main()