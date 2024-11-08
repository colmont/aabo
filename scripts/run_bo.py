import sys

import wandb 
sys.path.append("../")
import torch
import hydra
from omegaconf import DictConfig, OmegaConf
import pandas as pd
import warnings
warnings.filterwarnings('ignore')
import os
os.environ["WANDB_SILENT"] = "True"
import signal 
import gpytorch
from svgp.model import GPModel
from svgp.generate_candidates import generate_batch
from svgp.train_model import (
    update_model_elbo, 
    update_model_and_generate_candidates_eulbo,
)
from utils.create_wandb_tracker import create_wandb_tracker
from utils.set_seed import set_seed 
from utils.get_random_init_data import get_random_init_data
from utils.turbo import TurboState, update_state
from utils.set_inducing_points_with_moss23 import get_optimal_inducing_points
# for exact gp baseline: 
from botorch.models import SingleTaskGP
from botorch.fit import fit_gpytorch_mll
from gpytorch.mlls import ExactMarginalLogLikelihood
# for specific tasks
from tasks.hartmannn import Hartmann6D
from tasks.rover import RoverObjective
try:
    from tasks.lunar import LunarLanderObjective
    successful_lunar_import = True
except:
    print("Warning: failed to import LunarLanderObjective, current environment does not support needed imports for lunar lander task")
    successful_lunar_import = False 
try:
    from tasks.lasso_dna import LassoDNA
    successful_lasso_dna_import = True 
except:
    print("Warning: failed to import LassoDNA Objective, current environment does not support needed imports for Lasso DNA task")
    successful_lasso_dna_import = False 
try:
    from tasks.guacamol_objective import GuacamolObjective
except:
    print("Warning: failed to import GuacamolObjective, current environment does not support needed imports for guacamol tasks")

task_id_to_objective = {}
task_id_to_objective['hartmann6'] = Hartmann6D
if successful_lunar_import:
    task_id_to_objective['lunar'] = LunarLanderObjective 
task_id_to_objective['rover'] = RoverObjective
if successful_lasso_dna_import: 
    task_id_to_objective['dna'] = LassoDNA

@hydra.main(config_path='../configs', config_name='conf')
def main(cfg: DictConfig):

    # assertions on config
    if cfg.ablation1_fix_indpts_and_hypers:
        assert cfg.eulbo
        assert not cfg.ablation2_fix_hypers
    if cfg.ablation2_fix_hypers:
        assert cfg.eulbo
        assert not cfg.ablation1_fix_indpts_and_hypers
    if cfg.moss23_baseline:
        assert not cfg.eulbo
        assert not cfg.exact_gp_baseline
        assert cfg.inducing_pt_init_w_moss23
    if cfg.eulbo:
        assert not cfg.exact_gp_baseline
    if cfg.exact_gp_baseline: 
        assert not cfg.eulbo

    # set-up 
    set_seed(cfg.seed)
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu") 
    dtype_map = {
        32: torch.float32,
        64: torch.float64
    }
    DTYPE = dtype_map.get(cfg.float_dtype_as_int)
    if DTYPE is not None:
        torch.set_default_dtype(DTYPE)
    else:
        raise ValueError(f"float_dtype_as_int must be one of: {list(dtype_map.keys())}, instead got {cfg.float_dtype_as_int}")
    INIT_TRAINING_COMPLETE = False
    
        
    # log all args to wandb
    tracker = wandb.init(
        project=cfg.wandb_project_name if cfg.wandb_project_name else f"run-aabo-{cfg.task_id}",
        entity=cfg.wandb_entity,
        config=OmegaConf.to_container(cfg, resolve=True)
    )

    def handler(signum, frame):
        # if we Ctrl-c, make sure we terminate wandb tracker
        print("Ctrl-c hass been pressed, wait while we terminate wandb tracker...")
        tracker.finish() 
        msg = "tracker terminated, now exiting..."
        print(msg, end="", flush=True)
        exit(1)

    signal.signal(signal.SIGINT, handler)

    # Define objective and initialize training data
    if cfg.task_id in task_id_to_objective:
        objective = task_id_to_objective[cfg.task_id](dtype=DTYPE)
        # Obtain random initial training data
        train_x, train_y = get_random_init_data(
            num_initialization_points=cfg.num_initialization_points,
            objective=objective,
        )
    else:
        # Default to Guacamol objective if task_id not found in task_id_to_objective
        objective = GuacamolObjective(guacamol_task_id=cfg.task_id, dtype=DTYPE)
        
        if cfg.init_mol_tasks_w_guacamol_data:
            # Load Guacamol initial data
            df = pd.read_csv("../tasks/utils/selfies_vae/train_ys_v2.csv")
            train_y = torch.from_numpy(df[cfg.task_id].values).float()[:cfg.num_initialization_points]
            train_x = torch.load("../tasks/utils/selfies_vae/train_zs.pt")[:cfg.num_initialization_points]
            
            # Select top-k data points if specified
            train_y, top_k_idxs = torch.topk(train_y, min(cfg.update_on_n_pts, len(train_y)))
            train_x = train_x[top_k_idxs]
            
            # Convert data to specified dtype and adjust dimensions
            train_x, train_y = train_x.to(dtype=DTYPE), train_y.unsqueeze(-1).to(dtype=DTYPE)
        else:
            # Fall back to random initial training data if Guacamol data not specified
            train_x, train_y = get_random_init_data(
                num_initialization_points=cfg.num_initialization_points,
                objective=objective,
            )

    # Optional logging for training data shapes
    if cfg.verbose:
        print(f"train x shape: {train_x.shape}")
        print(f"train y shape: {train_y.shape}")

    # get normalized train y
    train_y_mean = train_y.mean()
    train_y_std = train_y.std()
    if train_y_std == 0:
        train_y_std = 1

    if cfg.normalize_ys:
        normed_train_y = (train_y - train_y_mean) / train_y_std
    else:
        normed_train_y = train_y 

    # initialize turbo state 
    tr_state = TurboState(
        dim=train_x.shape[-1],
        batch_size=cfg.bsz, 
        best_value=train_y.max().item(),
    )

    # Initialize GP model 
    if cfg.exact_gp_baseline:
        model = SingleTaskGP(
            train_x, 
            normed_train_y, 
            covar_module=gpytorch.kernels.ScaleKernel(gpytorch.kernels.RBFKernel()),
            likelihood=gpytorch.likelihoods.GaussianLikelihood().to(DEVICE),
        )
    else:
        # get inducing points 
        if len(train_x) >= cfg.n_inducing_pts:
            inducing_points = train_x[0:cfg.n_inducing_pts,:]
        else:
            n_extra_ind_pts = cfg.n_inducing_pts - len(train_x)
            extra_ind_pts = torch.rand(n_extra_ind_pts, objective.dim)*(objective.ub - objective.lb) + objective.lb
            inducing_points = torch.cat((train_x, extra_ind_pts), -2)
        
        # define approximate GP model 
        learn_inducing_locations = True 
        if cfg.moss23_baseline:
            learn_inducing_locations = False 
        model = GPModel(
            inducing_points=inducing_points.to(DEVICE), 
            likelihood=gpytorch.likelihoods.GaussianLikelihood().to(DEVICE),
            learn_inducing_locations=learn_inducing_locations,
        ).to(DEVICE)
        if cfg.inducing_pt_init_w_moss23: 
            optimal_inducing_points = get_optimal_inducing_points(
                model=model,
                prev_inducing_points=inducing_points, 
            )
            model = GPModel(
                inducing_points=optimal_inducing_points, 
                likelihood=gpytorch.likelihoods.GaussianLikelihood().to(DEVICE),
                learn_inducing_locations=learn_inducing_locations,
            ).to(DEVICE)

    def grab_data_for_update(init_training_complete):
        if not init_training_complete:
            x_update_on = train_x 
            normed_y_update_on = normed_train_y.squeeze() 
            init_training_complete = True 
        else:
            x_update_on = train_x[-cfg.update_on_n_pts:]
            normed_y_update_on = normed_train_y.squeeze()[-cfg.update_on_n_pts:]

        return x_update_on, normed_y_update_on, init_training_complete

    # Main optimization loop
    while objective.num_calls < cfg.max_n_oracle_calls:
        # Update wandb with optimization progress
        best_score_found = train_y.max().item()
        n_calls_ = objective.num_calls
        if cfg.verbose:
            print(f"After {n_calls_} oracle calls, Best reward = {best_score_found}")
        # log data to wandb on each loop
        dict_log = {
            "best_found":best_score_found,
            "n_oracle_calls":n_calls_,
        }
        tracker.log(dict_log) 

        # Normalize train ys
        if cfg.normalize_ys:
            normed_train_y = (train_y - train_y_mean) / train_y_std
        else:
            normed_train_y = train_y

        # Update model on data collected 
        if cfg.exact_gp_baseline:
            model.set_train_data(train_x, normed_train_y, strict=False)
            exact_gp_mll = ExactMarginalLogLikelihood(model.likelihood, model)
            fit_gpytorch_mll(exact_gp_mll)
        else:
            if cfg.eulbo:
                n_epochs_elbo = cfg.n_warm_start_epochs
                train_to_convergence_elbo = False 
            else:
                n_epochs_elbo = cfg.n_update_epochs
                train_to_convergence_elbo = True 
            x_update_on, normed_y_update_on, INIT_TRAINING_COMPLETE = grab_data_for_update(INIT_TRAINING_COMPLETE)
            update_model_dict = update_model_elbo(
                    model=model,
                    train_x=x_update_on,
                    train_y=normed_y_update_on,
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
            Y=normed_train_y,
            batch_size=cfg.bsz,
            acqf=cfg.acq_func,
            device=DEVICE,
            absolute_bounds=(objective.lb, objective.ub),
            use_turbo=cfg.use_turbo,
            tr_length=tr_state.length,
            dtype=DTYPE,
        )

        # If using eulbo, use above model update and candidate generaiton as warm start
        if cfg.eulbo:
            lb = objective.lb
            ub = objective.ub
            update_model_dict = update_model_and_generate_candidates_eulbo(
                model=model,
                train_x=x_update_on,
                train_y=normed_y_update_on,
                lb=lb,
                ub=ub,
                lr=cfg.lr,
                n_epochs=cfg.n_update_epochs,
                train_bsz=cfg.train_bsz,
                grad_clip=cfg.grad_clip,
                normed_best_f=normed_train_y.max(),
                acquisition_bsz=cfg.bsz,
                max_allowed_n_failures_improve_loss=cfg.max_allowed_n_failures_improve_loss,
                max_allowed_n_epochs=cfg.max_allowed_n_epochs,
                init_x_next=x_next, 
                x_next_lr=cfg.x_next_lr,
                alternate_updates=cfg.alternate_eulbo_updates,
                num_kg_samples=cfg.num_kg_samples, 
                use_kg=cfg.use_kg,
                dtype=DTYPE,
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
        train_y = torch.cat((train_y, y_next), dim=-2)

        # if running TuRBO, update trust region state 
        if cfg.use_turbo:
            tr_state = update_state(
                state=tr_state, 
                Y_next=y_next,
            )
            if tr_state.restart_triggered:
                tr_state = TurboState( 
                    dim=train_x.shape[-1],
                    batch_size=cfg.bsz, 
                    best_value=train_y.max().item(),
                )
        
    tracker.finish()

if __name__ == "__main__":
    main()