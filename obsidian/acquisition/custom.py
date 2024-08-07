"""Custom implementations of acquisition functions using BoTorch API"""

from botorch.acquisition.monte_carlo import MCAcquisitionFunction
from botorch.sampling.normal import SobolQMCNormalSampler
from botorch.utils import t_batch_mode_transform
from botorch.acquisition.multi_objective.objective import IdentityMCMultiOutputObjective
from botorch.models.model import Model
from botorch.sampling.base import MCSampler
from botorch.acquisition.objective import MCAcquisitionObjective, PosteriorTransform

import torch
from torch import Tensor


class qMean(MCAcquisitionFunction):
    """
    Acquisition function which optimizes for the
    maximum value of the posterior mean
    """
    def __init__(self,
                 model: Model,
                 sampler: MCSampler | None = None,
                 objective: MCAcquisitionObjective | None = None,
                 posterior_transform: PosteriorTransform | None = None,
                 X_pending: Tensor | None = None):
        
        if sampler is None:
            sampler = SobolQMCNormalSampler(sample_shape=torch.Size([512]))
        
        # An objective is required if this aq is used on a multi-output model
        if objective is None:
            if model.num_outputs > 1:
                objective = IdentityMCMultiOutputObjective()
            else:
                objective = None  # This will be set to identity in super call

        super().__init__(model=model, sampler=sampler, objective=objective,
                         posterior_transform=posterior_transform, X_pending=X_pending)

    @t_batch_mode_transform()
    def forward(self,
                x: Tensor) -> Tensor:
        # x dimensions: b * q * d
        posterior = self.model.posterior(x)  # dimensions: b * q
        samples = self.get_posterior_samples(posterior)  # dimensions: n * b * q * o

        o = self.objective(samples, x)
        if self.objective._is_mo:
            o = o.sum(dim=-1)  # Sum over o if objective is MOO
        else:
            o = samples.sum(dim=-1)  # Sum over o if f no objectives

        # Mean over posterior, sum over q in any case
        return o.mean(dim=0).sum(dim=-1)


class qSpaceFill(MCAcquisitionFunction):
    """
    Acquisition function which optimizes for the
    maximum value of minimum distance between a point and the training data
    """
    def __init__(self,
                 model: Model,
                 sampler: MCSampler | None = None,
                 objective: MCAcquisitionObjective | None = None,
                 posterior_transform: PosteriorTransform | None = None,
                 X_pending: Tensor | None = None):
        
        if sampler is None:
            sampler = SobolQMCNormalSampler(sample_shape=torch.Size([512]))
        
        # An objective is required if this aq is used on a multi-output model
        if model.num_outputs > 1:
            objective = IdentityMCMultiOutputObjective()
        else:
            objective = None

        super().__init__(model=model, sampler=sampler, objective=objective,
                         posterior_transform=posterior_transform, X_pending=X_pending)
        
    @t_batch_mode_transform()
    def forward(self,
                x: Tensor) -> Tensor:
        # x dimensions: b * q * d
        x_train = self.model.train_inputs[0][0]  # train_inputs is a list of tuples
        
        # For sequential mode, add pending data points to "train"
        if self.X_pending is not None:
            x_train = torch.concat((x_train, self.X_pending))
               
        x = x.unsqueeze(-2)  # Insert expmt dimension before xdim
        x_train = x_train.unsqueeze(0).unsqueeze(0)  # Add batch and q dimensions
        
        # Norm distance over xdim
        dist = torch.norm(x-x_train, p=2, dim=-1)
        
        # Min over expmt and q (for joint evaluation)
        closest_dist = dist.min(dim=-1)[0].min(dim=-1)[0]

        return closest_dist
