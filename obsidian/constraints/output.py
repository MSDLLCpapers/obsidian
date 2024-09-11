"""Constraints on the output responses of a model"""

from .base import Constraint

from obsidian.parameters import Target
from obsidian.utils import unscale_samples
from obsidian.config import TORCH_DTYPE

import torch
from torch import Tensor
from typing import Callable


class Output_Constraint(Constraint):
    """
    Output constraint for a given set of targets.

    Must return a callable function that computes feasibility, where
        negative values imply feasible space.

    Note: Saving and loading input constraints is managed by Campaign
    """
    def __init__(self,
                 target: Target | list[Target]):
        super().__init__()
        self.target = self._validate_target(target)

    def _validate_target(self, target: Target | list[Target]):

        if not isinstance(target, (Target, list)):
            raise TypeError('Target must be a Target object or a list of Target objects')
        if isinstance(target, list):
            for t in target:
                if not isinstance(t, Target):
                    raise TypeError('Target must be a Target object or a list of Target objects')
        if isinstance(target, Target):
            target = [target]
        
        return target

class Blank_Constraint(Output_Constraint):
    """
    Dummy constraint function that proposes all samples as feasible.
    """
    def __init__(self,
                 target: Target | list[Target]):
        super().__init__(target)

    def forward(self,
                scale: bool = True) -> Callable:
        def constraint(samples: Tensor) -> Tensor:
            if scale:
                samples = unscale_samples(samples, self.target)
            feasibility = -1*torch.ones(size=samples.shape).max(dim=-1).values
            return feasibility
        return constraint
    
class L1_Constraint(Output_Constraint):
    """
    Calculates the L1 (absolute-value penalized) constraint
    """
    def __init__(self,
                 target: Target | list[Target],
                 offset: int | float = 1):
        super().__init__(target)
        self.register_buffer('offset', torch.tensor(offset, dtype=TORCH_DTYPE))

    def forward(self,
                scale: bool = True) -> Callable:
        def constraint(samples: Tensor) -> Tensor:
            if scale:
                samples = unscale_samples(samples, self.target)
            feasibility = (samples.sum(dim=-1) - self.offset)
            return feasibility
        return constraint