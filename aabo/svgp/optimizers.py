from gpytorch.optim import NGD
from torch.optim import Adam

#TODO: the natural gradient parts in both optimizers have not been tested properly, they need more attention

#TODO: add grad clip as part of this setup?
class OptimizerELBO:
    def __init__(self, model, num_data, lr=0.01, natural_gradient=False, alternating_updates=False):
        #TODO: assertion error in case NaturalVariationalDistribution is not used in model (natural_gradient=True)
        self.model = model
        self.num_data = num_data
        self.lr = lr
        self.natural_gradient = natural_gradient
        self.alternating_updates = alternating_updates
        self.optimizer_list = self._get_optimizer_list()
        self.optimizer_idx = 0
    
    def _get_optimizer_list(self):
        if self.natural_gradient:
            #TODO: custom natural gradient lr in conf, instead of default 0.1 used here
            variational_ngd_optimizer = NGD(self.model.variational_parameters(), num_data=self.num_data, lr=0.1)
            hyperparameter_optimizer = Adam([{'params': self.model.hyperparameters()}], lr=self.lr)
            return [variational_ngd_optimizer, hyperparameter_optimizer]
        else:
            return [Adam([{'params': self.model.parameters()}], lr=self.lr)]

    def zero_grad(self):
        if self.alternating_updates:
            self.optimizer_list[self.optimizer_idx].zero_grad()
        else:
            for optimizer in self.optimizer_list:
                optimizer.zero_grad()

    def step(self):
        if self.alternating_updates:
            self.optimizer_list[self.optimizer_idx].step()
            self.optimizer_idx = (self.optimizer_idx + 1) % len(self.optimizer_list)
        else:
            for optimizer in self.optimizer_list:
                optimizer.step()


#TODO: add grad clip as part of this setup?
class OptimizerEULBO:
    def __init__(
        self, 
        model, 
        x_next, 
        num_data, 
        x_next_lr=0.001, 
        lr=0.01, 
        natural_gradient=False, 
        alternating_updates=False,
        ablation1_fix_indpts_and_hypers=False,
        ablation2_fix_hypers=False,
    ):
        #TODO: assertion error in case NaturalVariationalDistribution is not used in model (natural_gradient=True)
        if (ablation1_fix_indpts_and_hypers and natural_gradient) or (ablation2_fix_hypers and natural_gradient):
            raise ValueError("Natural gradient not supported for ablation1_fix_indpts_and_hypers and ablation2_fix_hypers")
        self.model = model
        self.x_next = x_next
        self.num_data = num_data
        self.x_next_lr = x_next_lr
        self.lr = lr
        self.natural_gradient = natural_gradient
        self.alternating_updates = alternating_updates
        self.ablation1_fix_indpts_and_hypers = ablation1_fix_indpts_and_hypers
        self.ablation2_fix_hypers = ablation2_fix_hypers
        self.params_to_update = self._get_params_to_update()
        if self.alternating_updates:
            self.model_optimizers, self.x_next_optimizers = self._get_optimizers()
        else:
            self.joint_optimizers = self._get_optimizers()
        self.optimizer_idx = 0
    
    def _get_params_to_update(self):
        if self.ablation1_fix_indpts_and_hypers: 
            return list(self.model.variational_parameters())
        elif self.ablation2_fix_hypers: 
            return list(self.model.variational_parameters()) + [self.model.variational_strategy.inducing_points]
        else:
            return list(self.model.parameters())

    def _get_optimizers(self):
        if self.natural_gradient:
            #TODO: custom natural gradient lr in conf, instead of default 0.1 used here
            variational_ngd_optimizer = NGD(self.model.variational_parameters(), num_data=self.num_data, lr=0.1)
            if self.alternating_updates:
                x_next_optimizer = Adam([{'params': self.x_next}], lr=self.x_next_lr)
                hyperparameter_optimizer = Adam([{'params': self.model.hyperparameters()}], lr=self.lr)
                return [variational_ngd_optimizer, hyperparameter_optimizer], [x_next_optimizer]
            else:
                hyperparams_x_next_optimizer = Adam([{'params': self.model.hyperparameters()}, {'params': self.x_next}], lr=self.lr)
                return [variational_ngd_optimizer, hyperparams_x_next_optimizer]
        else:
            if self.alternating_updates:
                x_next_optimizer = Adam([{'params': self.x_next}], lr=self.x_next_lr)
                model_optimizer = Adam([{'params': self.params_to_update} ], lr=self.lr)
                return [model_optimizer], [x_next_optimizer]
            else:
                return [Adam([{'params': self.params_to_update}, {'params': self.x_next}], lr=self.lr)]

    def zero_grad(self, model_training=True):
        if self.alternating_updates:
            if model_training:
                for optimizer in self.model_optimizers:
                    optimizer.zero_grad()
            else:
                for optimizer in self.x_next_optimizers:
                    optimizer.zero_grad()
        else:
            for optimizer in self.joint_optimizers:
                optimizer.zero_grad()

    def step(self, model_training=True):
        if self.alternating_updates:
            if model_training:
                for optimizer in self.model_optimizers:
                    optimizer.step()
            else:
                for optimizer in self.x_next_optimizers:
                    optimizer.step()
        else:
            for optimizer in self.joint_optimizers:
                optimizer.step()