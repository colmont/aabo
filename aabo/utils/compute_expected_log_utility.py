import torch 
from torch.quasirandom import SobolEngine
from linear_operator.operators import TriangularLinearOperator
from gpytorch.utils.quadrature import GaussHermiteQuadrature1D
from botorch.generation.sampling import MaxPosteriorSampling
from botorch.utils.safe_math import log_softplus 

class ExpectedLogUtility:

    def __init__(
        self, 
        acq_fun, 
        acquisition_bsz, 
        normed_best_f, 
        num_mc_samples, 
        dim,
        lb,
        ub,
        model,
        ei_samples=None, 
        kg_samples=None, 
        kg_zs=None, 
        use_botorch_stable_log_softplus=False
    ):
        self.acq_fun = acq_fun
        self.acquisition_bsz = acquisition_bsz
        self.normed_best_f = normed_best_f
        self.num_mc_samples = num_mc_samples
        self.use_botorch_stable_log_softplus = use_botorch_stable_log_softplus

        #FIXME: technically, we should only compute ei_samples if acq_fun==ei
        # -> did not do it here to preseve seed reproducibility with orig aabo code
        self.ei_samples = self._get_ei_samples() if ei_samples is None else ei_samples
        if kg_samples is None and acq_fun == "kg":
            self.kg_samples, self.kg_zs = self._get_kg_samples_and_zs(dim, lb, ub, model)
        else:
            self.kg_samples = kg_samples
            self.kg_zs = kg_zs

        acq_fun_map = {
            ("kg", 1): self._get_expected_log_utility_knowledge_gradient,
            ("kg", "q"): self._get_q_expected_log_utility_knowledge_gradient,
            ("ei", 1): self._get_expected_log_utility_ei,
            ("ei", "q"): self._get_q_expected_log_utility_ei,
        }

        key = (self.acq_fun, self.acquisition_bsz if self.acquisition_bsz == 1 else "q")
        if key not in acq_fun_map:
            raise ValueError(f"Invalid acquisition function: {self.acq_fun}")

        self.expected_log_utility_fn = acq_fun_map[key]

    def _log_softplus(self, x):
        if self.use_botorch_stable_log_softplus:
            return log_softplus(x)
        else:
            softplus = torch.nn.Softplus()
            return torch.log(softplus(x))

    def _get_ei_samples(self):
        return torch.randn(self.num_mc_samples, self.acquisition_bsz)

    def _get_kg_samples_and_zs(self, dim, lb, ub, model):
        # Use ts to initialize kg_samples 
        thompson_sampling = MaxPosteriorSampling(model=model, replacement=False) 
        n_ts_candidates = min(5000, max(2000, 200 * dim)) 
        sobol = SobolEngine(dim, scramble=True) 
        ts_x_cands = sobol.draw(n_ts_candidates)
        ts_x_cands = ts_x_cands*(ub - lb) + lb 
        with torch.no_grad():
            kg_samples = thompson_sampling(ts_x_cands, num_samples=self.num_mc_samples) 
        kg_samples = torch.clone(kg_samples.detach()).requires_grad_(True)

        if self.acquisition_bsz == 1:
            zs = torch.randn(self.num_mc_samples, requires_grad=True) 
        else:
            zs = torch.randn(self.num_mc_samples, self.acquisition_bsz, requires_grad=True)

        return kg_samples, zs 

    def _get_q_expected_log_utility_ei(self, model, x_next):
        # x_next.shape (q, d)
        output = model(x_next) # q-dim multivariate normal 
        # use MC sampling 
        samples = output.rsample(torch.Size([self.num_mc_samples]), base_samples=self.ei_samples) 
        # compute log utility of each sample 
        log_utilities = self._log_softplus(samples - self.normed_best_f) # (S, q) of utilities for each sample 
        # max over q dimension, mean over s dimension to get final expected_log_utility
        expected_log_utility = log_utilities.amax(-1) # (S,) 
        return expected_log_utility 

    def _get_expected_log_utility_ei(self, model, x_next):
        output = model(x_next)  # x_next.shape = (q,d)
        def log_utility(y):
            return self._log_softplus(y - self.normed_best_f)
        ghq = GaussHermiteQuadrature1D().to(output.mean.device)
        expected_log_utility = ghq(log_utility, output)
        return expected_log_utility

    def _get_q_expected_log_utility_knowledge_gradient(self, model, x_next):
        x_next_pred = model(x_next)
        y_samples = x_next_pred.rsample(
            torch.Size([self.kg_samples.shape[0]]), 
            base_samples=self.kg_zs
        ) + x_next_pred.stddev * self.kg_zs  # (num_kg_samples, q) (S, q)
        chol_factor = model.variational_strategy._cholesky_factor(None) # (M,M)  
        U = model.covar_module(model.variational_strategy.inducing_points, x_next) # (M,q) 
        S = model.covar_module(x_next, x_next, diag=True) # K(x_next, x_next), torch.Size(q), 
        chol_factor_tensor = chol_factor._tensor.tensor # (M,M) 
        chol_factor_tensor_repeated = chol_factor_tensor.repeat(x_next.shape[0], 1, 1,) # (q, M, M)
        L = torch.cat((chol_factor_tensor_repeated, torch.zeros(x_next.shape[0], chol_factor_tensor.shape[-1], 1)), -1) # (q, M, M+1)
        var_mean = chol_factor @ model.variational_strategy.variational_distribution.mean
        var_mean = var_mean.repeat(x_next.shape[0], 1).unsqueeze(-1) # (q,M,1)
        var_mean_repeated = var_mean.repeat(1, 1, y_samples.shape[-2]) # (q,M,num_kg_samples)
        y_samples_reshaped = y_samples.reshape(y_samples.shape[-1], y_samples.shape[-2]) # (q,S)
        y_samples_reshaped = y_samples_reshaped.unsqueeze(-2) # (q,1,S)
        var_mean_repeated = torch.cat((var_mean_repeated, y_samples_reshaped), -2) # (q,M+1,num_kg_samples)
        L_12 = chol_factor.solve(U.evaluate_kernel().tensor) # (M,q)
        L_12_mt = L_12.mT.unsqueeze(-1) # (q,M,1)
        schur_complement = S - (L_12_mt * L_12_mt).squeeze(-1).sum(-1) # (q,)
        schur_complement = schur_complement.unsqueeze(-1).unsqueeze(-1) # (q,1,1)
        L_22 = schur_complement.to_dense()**0.5  # (q,1,1)
        L_temp = torch.cat((L_12_mt, L_22), -2) # (q, M+1, 1)
        L_temp_reshaped = L_temp.squeeze().unsqueeze(-2) 
        L = torch.cat((L, L_temp_reshaped), -2) # (q, M+1, M+1)
        L = TriangularLinearOperator(L) 
        alphas = L._transpose_nonbatch().solve(L.solve(var_mean_repeated)) # (q, M+1, S)
        x_next_temp = x_next.unsqueeze(-2) # (q,1,d)
        q_Zs = model.variational_strategy.inducing_points.repeat(x_next.shape[0], 1, 1) # (q,M,d)
        inducing_points_and_x_next = torch.cat((q_Zs, x_next_temp), -2) # (q, M+1, D)
        constant_mean = model.mean_module.constant
        pred_mean_each_x_sample = model.covar_module(self.kg_samples, inducing_points_and_x_next) # (q, S, M+1)
        pred_mean_each_x_sample = pred_mean_each_x_sample * alphas.mT 
        pred_mean_each_x_sample = pred_mean_each_x_sample.sum(-1) + constant_mean # (q,S)

        expected_log_utility_kg = self._log_softplus(pred_mean_each_x_sample - self.normed_best_f) # (q, S,)
        expected_log_utility_kg = expected_log_utility_kg.amax(-2) # (S,)
        
        return expected_log_utility_kg 

    def _get_expected_log_utility_knowledge_gradient(self, model, x_next):
        x_next_pred = model(x_next)
        y_samples = x_next_pred.mean + x_next_pred.stddev * self.kg_zs # (num_kg_samples,)
        y_samples = y_samples.unsqueeze(-2) # (1, num_kg_samples) = (1,S)
        chol_factor = model.variational_strategy._cholesky_factor(None) # (M,M)
        U = model.covar_module(model.variational_strategy.inducing_points, x_next) # (M,1)
        S = model.covar_module(x_next, x_next) # K(x_next, x_next)
        chol_factor_tensor = chol_factor._tensor.tensor # (M,M)
        L = torch.cat((chol_factor_tensor, torch.zeros(chol_factor_tensor.shape[-1], 1)), -1) # (M, M+1)
        var_mean = chol_factor @ model.variational_strategy.variational_distribution.mean
        var_mean = var_mean.unsqueeze(-1) # (M,1)
        var_mean_repeated = var_mean.repeat(1, y_samples.shape[-1]) # (M,num_kg_samples) = (M,S) 
        var_mean_repeated = torch.cat((var_mean_repeated, y_samples)) # (M+1,num_kg_samples) = (M+1, S)
        L_12 = chol_factor.solve(U.evaluate_kernel().tensor) # (M,1)
        schur_complement = S - L_12.mT @ L_12 
        L_22 = schur_complement.to_dense()**0.5 
        L_temp = torch.cat((L_12, L_22), -2)
        L_temp = L_temp.squeeze().unsqueeze(-2) 
        L = torch.cat((L, L_temp), -2) # (M+1, M+1) 
        L = TriangularLinearOperator(L) 
        alphas = L._transpose_nonbatch().solve(L.solve(var_mean_repeated)) # (M+1, S) 
        inducing_points_and_x_next = torch.cat((model.variational_strategy.inducing_points, x_next), -2) # (M+1, D)
        constant_mean = model.mean_module.constant
        pred_mean_each_x_sample = model.covar_module(self.kg_samples, inducing_points_and_x_next) # (S, M+1) 
        pred_mean_each_x_sample = pred_mean_each_x_sample * alphas.mT 
        pred_mean_each_x_sample = pred_mean_each_x_sample.sum(-1) + constant_mean # (S,) 

        expected_log_utility_kg = self._log_softplus(pred_mean_each_x_sample - self.normed_best_f) # (S,)

        return expected_log_utility_kg

    def __call__(self, model, x_next):
        expected_log_utility_x_next = self.expected_log_utility_fn(model=model, x_next=x_next)
        return expected_log_utility_x_next.mean()