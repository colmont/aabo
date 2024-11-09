import pandas as pd
import torch 
from tasks.hartmannn import Hartmann6D
from tasks.rover import RoverObjective
from importlib import import_module

def get_objective(task_id, dtype):
    non_tricky_tasks = {
        'hartmann6': Hartmann6D,
        'rover': RoverObjective,
    }
    guacamol_tasks = ['osmb', 'fexo', 'med1', 'med2']
    tricky_tasks = {
        'lunar': ('tasks.lunar', 'LunarLanderObjective'),
        'dna': ('tasks.lasso_dna', 'LassoDNA'),
    }
    
    # Hartmann6D and Rover should not be an issue
    if task_id in non_tricky_tasks:
        return non_tricky_tasks[task_id](dtype=dtype)

    # Guacamol tasks require more care and import could fail
    elif task_id in guacamol_tasks:
        try:
            from tasks.guacamol_objective import GuacamolObjective
        except ImportError:
            print("Warning: failed to import GuacamolObjective, current environment does not support needed imports for guacamol tasks")
        return GuacamolObjective(task_id)

    # Lunar and LassoDNA tasks require more care and import could fail
    elif task_id in tricky_tasks:
        try:
            module_name, class_name = tricky_tasks[task_id]
            module = import_module(module_name)
        except ImportError:
            print(f"Warning: failed to import {class_name} from {module_name}, current environment does not support needed imports for {task_id} task")
        return getattr(module, class_name)(dtype=dtype)
    
def get_random_init_data(task_id, objective, num_initialization_points, init_mol_tasks_w_guacamol_data, update_on_n_pts, dtype):
    if init_mol_tasks_w_guacamol_data:
        assert task_id in ['osmb', 'fexo', 'med1', 'med2'], "Only guacamol tasks are supported for this initialization method"

        # Load Guacamol initial data
        df = pd.read_csv("../tasks/utils/selfies_vae/train_ys_v2.csv")
        train_y = torch.from_numpy(df[task_id].values).float()[:num_initialization_points]
        train_x = torch.load("../tasks/utils/selfies_vae/train_zs.pt")[:num_initialization_points]
        
        # Select top-k data points if specified
        train_y, top_k_idxs = torch.topk(train_y, min(update_on_n_pts, len(train_y)))
        train_x = train_x[top_k_idxs]
        
        # Convert data to specified dtype and adjust dimensions
        train_x, train_y = train_x.to(dtype=dtype), train_y.unsqueeze(-1).to(dtype=dtype)
    else:
        lb, ub = objective.lb, objective.ub 
        train_x = torch.rand(num_initialization_points, objective.dim)*(ub - lb) + lb
        train_y = objective(train_x)

    return train_x, train_y