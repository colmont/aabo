import os
import warnings
import hydra
from omegaconf import DictConfig
import torch
from aabo.utils.data_loader import get_objective, get_random_init_data
from aabo.utils.get_turbo_lb_ub import get_turbo_lb_ub
from aabo.utils.turbo import TurboState, update_state
from aabo.utils.setup import (
    validate_config,
    set_seed,
    set_dtype,
    set_device,
    set_wandb_tracker,
    handle_interrupt,
)

warnings.filterwarnings('ignore')
os.environ["WANDB_SILENT"] = "True"


@hydra.main(config_path='../configs', config_name='conf')
def main(cfg: DictConfig):

    # Set-up 
    validate_config(cfg)
    set_seed(cfg.seed)
    set_dtype(cfg.float_dtype_as_int)
    set_device()
    tracker = set_wandb_tracker(cfg)
    handle_interrupt(tracker)

    # Obtain random initial training data
    objective = get_objective(cfg.benchmark.name)
    train_x, train_y = get_random_init_data(
        task_id=cfg.benchmark.name,
        objective=objective,
        num_initialization_points=cfg.benchmark.num_initialization_points,
        init_mol_tasks_w_guacamol_data=cfg.init_mol_tasks_w_guacamol_data,
        update_on_n_pts=cfg.update_on_n_pts,
    )

    # Optional logging for training data shapes
    if cfg.verbose:
        print(f"train x shape: {train_x.shape}")
        print(f"train y shape: {train_y.shape}")

    # No need to normalize for random search
    train_y_origscale = train_y

    # Initialize turbo state 
    if cfg.benchmark.use_turbo:
        tr_state = TurboState(
            dim=train_x.shape[-1],
            batch_size=cfg.benchmark.bsz, 
            best_value=train_y_origscale.max().item(),
        )

    # Main loop
    while objective.num_calls < cfg.benchmark.max_n_oracle_calls:
        
        # Update wandb with optimization progress
        best_score_found = train_y_origscale.max().item()
        n_calls_ = objective.num_calls
        dict_log = {
            "best_found":best_score_found,
            "n_oracle_calls":n_calls_,
        }
        tracker.log(dict_log) 
        if cfg.verbose:
            print(f"After {n_calls_} oracle calls, Best reward = {best_score_found}")

        # Randomly pick a point to evaluate (in the trust region)
        lb = objective.lb
        ub = objective.ub
        if cfg.benchmark.use_turbo: 
            assert tr_state.length is not None 
            lb, ub = get_turbo_lb_ub(
                ub=ub,
                lb=lb,
                X=train_x, 
                Y=train_y,
                tr_length=tr_state.length,
            )
        x_next = torch.rand(cfg.benchmark.bsz, train_x.shape[-1], requires_grad=False) * (ub - lb) + lb

        # Evaluate candidates 
        y_next = objective(x_next)
        
        # Update data 
        train_x = torch.cat((train_x, x_next), dim=-2)
        train_y_origscale = torch.cat((train_y_origscale, y_next), dim=-2)

        # If running TuRBO, update trust region state 
        if cfg.benchmark.use_turbo:
            tr_state = update_state(
                state=tr_state, 
                Y_next=y_next,
            )
            if tr_state.restart_triggered:
                tr_state = TurboState( 
                    dim=train_x.shape[-1],
                    batch_size=cfg.benchmark.bsz, 
                    best_value=train_y_origscale.max().item(),
                )
        
    tracker.finish()

if __name__ == "__main__":
    main()