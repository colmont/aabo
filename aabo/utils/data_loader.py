import os
import pandas as pd
import torch 
from importlib import import_module

from aabo.tasks.hartmannn import Hartmann6D
from aabo.tasks.rover import RoverObjective

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def get_objective(task_id):
    non_tricky_tasks = {
        'hartmann6': Hartmann6D,
        'rover': RoverObjective,
    }
    guacamol_tasks = ['osmb', 'fexo', 'med1', 'med2']
    tricky_tasks = {
        'lunar': ('aabo.tasks.lunar', 'LunarLanderObjective'),
        'dna': ('aabo.tasks.lasso_dna', 'LassoDNA'),
    }
    
    # Hartmann6D and Rover should not be an issue
    if task_id in non_tricky_tasks:
        return non_tricky_tasks[task_id]()

    # Guacamol tasks require more care and import could fail
    elif task_id in guacamol_tasks:
        from aabo.tasks.guacamol_objective import GuacamolObjective
        return GuacamolObjective(task_id)

    # Lunar and LassoDNA tasks require more care and import could fail
    elif task_id in tricky_tasks:
        module_name, class_name = tricky_tasks[task_id]
        module = import_module(module_name)
        return getattr(module, class_name)()
    
def get_random_init_data(task_id, objective, num_initialization_points, init_mol_tasks_w_guacamol_data, update_on_n_pts):
    if init_mol_tasks_w_guacamol_data:
        assert task_id in ['osmb', 'fexo', 'med1', 'med2'], "Only guacamol tasks are supported for this initialization method"

        # Load Guacamol data 
        base_dir = os.path.dirname(os.path.abspath(__file__))
        selfies_vae_dir = os.path.join(os.path.dirname(base_dir), 'tasks', 'utils', 'selfies_vae')
        df = pd.read_csv(os.path.join(selfies_vae_dir, 'train_ys_v2.csv'))
        train_y = torch.from_numpy(df[task_id].values).float()[:num_initialization_points]
        train_x = torch.load(os.path.join(selfies_vae_dir, 'train_zs.pt'))[:num_initialization_points]
        
        # Select top-k data points if specified
        train_y, top_k_idxs = torch.topk(train_y, min(update_on_n_pts, len(train_y)))
        train_x = train_x[top_k_idxs]
        
        # Adjust dimensions, dtype and device
        train_x, train_y = train_x, train_y.unsqueeze(-1)
        train_x, train_y = train_x.to(device), train_y.to(device)
        train_x, train_y = train_x.to(torch.get_default_dtype()), train_y.to(torch.get_default_dtype())

    else:
        lb, ub = objective.lb, objective.ub 
        train_x = torch.rand(num_initialization_points, objective.dim)*(ub - lb) + lb
        train_y = objective(train_x)

    return train_x, train_y