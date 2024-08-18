"""Constraints on the output responses of a model"""

from obsidian.parameters import Target
from obsidian.utils import unscale_samples

import torch
from torch import Tensor
from typing import Callable

# Negative values imply feasibility!
# Note that these are OUTPUT constraints


def OutConstraint_Blank(target: Target | list[Target]) -> Callable:
    """
    Dummy constraint function that proposes all samples as feasible.

    Args:
        target (Target or list[Target]): The target or list of targets.

    Returns:
        callable: callable constraint function

    """
    def constraint(samples: Tensor) -> Tensor:
        samples = unscale_samples(samples, target)
        feasibility = -1*torch.ones(size=samples.shape).max(dim=-1).values
        return feasibility
    return constraint


def OutConstraint_L1(target: Target | list[Target],
                     offset: int | float = 1) -> Callable:
    """
    Calculates the L1 (absolute-value penalized) constraint

    Args:
        target (Target | list[Target]): The target value or a list of target values.
        offset (int | float, optional): The offset value for the constraint. Defaults to 1.

    Returns:
        callable: callable constraint function

    """
    def constraint(samples: Tensor) -> Tensor:
        samples = unscale_samples(samples, target)
        feasibility = (samples.sum(dim=-1) - offset)
        return feasibility
    return constraint
