import torch
import gpytorch
from svgp.model import GPModel
from utils.set_inducing_points_with_moss23 import get_optimal_inducing_points

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def get_inducing_points(
    init_train_x,
    objective,
    inducing_pt_init_w_moss23,
    n_inducing_pts,
):
    # Set inducing points as first n training points if enough data, else add random points
    if len(init_train_x) >= n_inducing_pts:
        inducing_points = init_train_x[0:n_inducing_pts,:]
    else:
        n_extra_ind_pts = n_inducing_pts - len(init_train_x)
        extra_ind_pts = torch.rand(n_extra_ind_pts, objective.dim)*(objective.ub - objective.lb) + objective.lb
        inducing_points = torch.cat((init_train_x, extra_ind_pts), -2)

    # Get optimal inducing points with Moss et al. (2023), starting from above inducing points
    if inducing_pt_init_w_moss23:
        model = GPModel(
            inducing_points=inducing_points, 
            likelihood=gpytorch.likelihoods.GaussianLikelihood().to(DEVICE),
            learn_inducing_locations=False,
        ).to(DEVICE)
        inducing_points = get_optimal_inducing_points(
            model=model,
            prev_inducing_points=inducing_points, 
        )

    return inducing_points