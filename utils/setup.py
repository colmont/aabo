import os
import random
import signal
import sys
import numpy as np
import torch
import wandb
from omegaconf import DictConfig, OmegaConf

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def set_seed(seed: int):
    """Set seed for reproducibility."""
    if not isinstance(seed, int):
        raise ValueError("Seed must be an integer")
    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    if device == "cuda":
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True

def validate_config(cfg: DictConfig):
    """Validate configuration settings."""
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

def set_dtype(float_dtype_as_int: int):
    """Set default dtype for torch tensors."""
    if float_dtype_as_int not in [32, 64]:
        raise ValueError(f"Invalid float_dtype_as_int: {float_dtype_as_int}. Must be one of: [32, 64]")
    dtype_map = {32: torch.float32, 64: torch.float64}
    dtype = dtype_map.get(float_dtype_as_int)
    torch.set_default_dtype(dtype)

def set_device():
    """Set default device for torch tensors."""
    torch.set_default_device(device)

def set_wandb_tracker(cfg: DictConfig):
    """Set-up Weights & Biases tracker."""
    project_name = cfg.wandb_project_name or f"run-aabo-{cfg.task_id}"
    return wandb.init(
        project=project_name,
        entity=cfg.wandb_entity,
        config=OmegaConf.to_container(cfg, resolve=True)
    )

def handle_interrupt(tracker):
    """Handles keyboard interrupt to ensure graceful tracker termination."""
    def signal_handler(signum, frame):
        print("Ctrl-C pressed, terminating tracker...")
        tracker.finish()
        print("Tracker terminated. Exiting.")
        sys.exit(1)

    signal.signal(signal.SIGINT, signal_handler)