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
    
    def __repr__(self):
        """String representation of object"""
        return f'{self.__class__.__name__}'
    
    
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
    
    def __repr__(self):
        """String representation of object"""
        return f'{self.__class__.__name__}(offset={self.offset})'


class ThresholdConstraint(Output_Constraint):
    """
    Base class for constraining a specific output to be within bounds.

    Constrains output such that: lower <= output <= upper (where bounds are optional).
    Negative constraint values indicate feasible space.

    Args:
        target: Target(s) for the campaign
        target_name: Name of the target to constrain
        lower: Lower bound (inclusive, optional). None means no lower bound.
        upper: Upper bound (inclusive, optional). None means no upper bound.

    Note:
        At least one of lower or upper must be specified.

    Example:
        # Upper bound only: Yield <= 95
        constraint = ThresholdConstraint(target=targets, target_name="Yield", upper=95.0)

        # Lower bound only: Yield >= 70
        constraint = ThresholdConstraint(target=targets, target_name="Yield", lower=70.0)

        # Both bounds: 70 <= Yield <= 95
        constraint = ThresholdConstraint(target=targets, target_name="Yield", lower=70.0, upper=95.0)
    """

    def __init__(
        self,
        target: Target | list[Target],
        target_name: str,
        lower: float | None = None,
        upper: float | None = None
    ):
        super().__init__(target)

        if lower is None and upper is None:
            raise ValueError("At least one of 'lower' or 'upper' must be specified")

        # Find target index by name
        target_names = [t.name for t in self.target]
        if target_name not in target_names:
            raise ValueError(f"Target '{target_name}' not found. Available: {target_names}")

        self.target_name = target_name
        self.output_index = target_names.index(target_name)

        # Register bounds as buffers (None handled separately)
        self.has_lower = lower is not None
        self.has_upper = upper is not None

        if self.has_lower:
            self.register_buffer('lower', torch.tensor(lower, dtype=TORCH_DTYPE))
        if self.has_upper:
            self.register_buffer('upper', torch.tensor(upper, dtype=TORCH_DTYPE))

    def forward(self, scale: bool = True) -> Callable:
        def constraint(samples: Tensor) -> Tensor:
            if scale:
                samples = unscale_samples(samples, self.target)

            val = samples[..., self.output_index]

            constraints = []

            # Lower bound: lower <= val (feasible when val >= lower, so lower - val <= 0)
            if self.has_lower:
                lower_satisfied = self.lower - val  # Negative when val >= lower
                constraints.append(lower_satisfied)

            # Upper bound: val <= upper (feasible when val <= upper, so val - upper <= 0)
            if self.has_upper:
                upper_satisfied = val - self.upper  # Negative when val <= upper
                constraints.append(upper_satisfied)

            # Return max (worst constraint violation) - positive = infeasible, negative = feasible
            if len(constraints) == 1:
                return constraints[0]
            else:
                return torch.stack(constraints, dim=-1).max(dim=-1).values

        return constraint

    def __repr__(self):
        bounds = []
        if self.has_lower:
            bounds.append(f"lower={self.lower.item()}")
        if self.has_upper:
            bounds.append(f"upper={self.upper.item()}")
        return f"{self.__class__.__name__}(target={self.target_name}, {', '.join(bounds)})"


class UpperBoundConstraint(ThresholdConstraint):
    """
    Convenience class for upper bound constraint: output <= upper.

    Args:
        target: Target(s) for the campaign
        target_name: Name of the target to constrain
        upper: Upper bound (inclusive)

    Example:
        # Constrain "Temperature" to be at most 100
        constraint = UpperBoundConstraint(
            target=targets,
            target_name="Temperature",
            upper=100.0
        )
    """

    def __init__(self, target: Target | list[Target], target_name: str, upper: float):
        super().__init__(target=target, target_name=target_name, lower=None, upper=upper)


class LowerBoundConstraint(ThresholdConstraint):
    """
    Convenience class for lower bound constraint: output >= lower.

    Args:
        target: Target(s) for the campaign
        target_name: Name of the target to constrain
        lower: Lower bound (inclusive)

    Example:
        # Constrain "Yield" to be at least 70
        constraint = LowerBoundConstraint(
            target=targets,
            target_name="Yield",
            lower=70.0
        )
    """

    def __init__(self, target: Target | list[Target], target_name: str, lower: float):
        super().__init__(target=target, target_name=target_name, lower=lower, upper=None)


class InRangeConstraint(ThresholdConstraint):
    """
    Convenience class for range constraint: lower <= output <= upper.

    Args:
        target: Target(s) for the campaign
        target_name: Name of the target to constrain
        lower: Lower bound (inclusive)
        upper: Upper bound (inclusive)

    Example:
        # Constrain "Purity" to be between 85 and 95
        constraint = InRangeConstraint(
            target=targets,
            target_name="Purity",
            lower=85.0,
            upper=95.0
        )
    """

    def __init__(self, target: Target | list[Target], target_name: str, lower: float, upper: float):
        if lower >= upper:
            raise ValueError(f"Lower bound ({lower}) must be less than upper bound ({upper})")
        super().__init__(target=target, target_name=target_name, lower=lower, upper=upper)
