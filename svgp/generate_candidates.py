import math
import torch
from torch.quasirandom import SobolEngine
from botorch.acquisition import qExpectedImprovement
from botorch.acquisition.logei import qLogExpectedImprovement
from botorch.optim import optimize_acqf
from botorch.generation.sampling import MaxPosteriorSampling
import copy 

def generate_batch(
    model,
    X,
    Y,
    batch_size,
    n_candidates=None,
    num_restarts=10,
    raw_samples=256,
    acqf="ts",
    dtype=torch.float32,
    device=torch.device('cuda'),
    absolute_bounds=None, 
    use_turbo=False, 
    tr_length=None,
):
    assert acqf in ("ts", "ei", "logei", "kg")
    assert torch.all(torch.isfinite(Y))
    if n_candidates is None: n_candidates = min(5000, max(2000, 200 * X.shape[-1]))

    x_center = copy.deepcopy(X[Y.argmax(), :]) 
    weights = torch.ones_like(x_center)
    
    if absolute_bounds is None:
        lb = X.min().item() 
        ub = X.max().item()
    else:
        lb, ub = absolute_bounds
    
    if use_turbo:
        assert tr_length is not None 
        weights = weights * (ub - lb)
        tr_lb = torch.clamp(x_center - weights * tr_length / 2.0, lb, ub) 
        tr_ub = torch.clamp(x_center + weights * tr_length / 2.0, lb, ub) 
        lb = tr_lb 
        ub = tr_ub 
    else:
        lb = lb*weights 
        ub = ub*weights 
    if acqf == "logei":
        qLogEI = qLogExpectedImprovement(model=model.cuda(), best_f=Y.max().cuda())
        X_next, _ = optimize_acqf(qLogEI,bounds=torch.stack([lb, ub]).cuda(),q=batch_size, num_restarts=num_restarts,raw_samples=raw_samples,)
    elif acqf == "ei":
        ei = qExpectedImprovement(model.cuda(), Y.max().cuda() ) 
        X_next, _ = optimize_acqf(ei,bounds=torch.stack([lb, ub]).cuda(),q=batch_size, num_restarts=num_restarts,raw_samples=raw_samples,)
    elif acqf == "ts":
        dim = X.shape[-1]
        lb = lb.cuda()
        ub = ub.cuda() 
        sobol = SobolEngine(dim, scramble=True) 
        pert = sobol.draw(n_candidates).to(dtype=dtype).cuda()
        pert = lb + (ub - lb) * pert
        lb = lb.cuda()
        ub = ub.cuda() 
        # Create a perturbation mask 
        prob_perturb = min(20.0 / dim, 1.0)
        mask = (torch.rand(n_candidates, dim, dtype=dtype, device=device)<= prob_perturb)
        ind = torch.where(mask.sum(dim=1) == 0)[0]
        mask[ind, torch.randint(0, dim - 1, size=(len(ind),), device=device)] = 1
        mask = mask.cuda()
        # Create candidate points from the perturbations and the mask
        X_cand = x_center.expand(n_candidates, dim).clone()
        X_cand = X_cand.cuda()
        X_cand[mask] = pert[mask]
        thompson_sampling = MaxPosteriorSampling(
            model=model,
            replacement=False,
        ) 
        with torch.no_grad():
            X_next = thompson_sampling(X_cand.cuda(), num_samples=batch_size )
    else:
        assert 0, f"unsupported acqf: {acqf}"


    return X_next.detach().cpu()
