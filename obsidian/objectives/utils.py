"""Utility objectives and utility functions for objectives handling"""


import torch
from torch import Tensor
from obsidian.parameters import Target
from .base import Objective
from obsidian.surrogates.utils import tensordict_to_dict
from obsidian.utils import TORCH_DTYPE


class Utopian_Distance(Objective):
    """
    A custom objective for calculating the distance between an output response
    and a most-desirable, ideal/utopian point.

    Args:
        utopian (list[float]): A list of values representing the utopian point.
        targets (Target | list[Target]): A single Target object or a list of Target objects.

    Attributes:
        u_t (Tensor): The utopian coordinates in transformed space
    
    Raises:
        TypeError: If the input is not a list of Target objects.
        ValueError: If the length of the utopian point and targets are not the same.

    """

    def __init__(self,
                 utopian: list[float],
                 targets: Target | list[Target]) -> None:

        if not isinstance(targets, list):
            targets = [targets]
        if not all(isinstance(t, Target) for t in targets):
            raise TypeError('All elements in the list must be Target objects')

        if len(utopian) != len(targets):
            raise ValueError("Length of utopian and targets must be the same")

        super().__init__(mo=len(targets) > 1)
        
        self.register_buffer('utopian', torch.tensor(utopian, dtype=TORCH_DTYPE))
        self.targets = targets
        
        # Must transform the utopian coordinates into the transformed space where
        # the optimizer operates on target samples
        u_t = [t.transform_f(u) for u, t in zip(utopian, targets)]
        self.u_t = torch.tensor(u_t, dtype=TORCH_DTYPE).flatten()

    def forward(self,
                samples: Tensor,
                X: Tensor | None = None) -> Tensor:
        
        distance = (-1)*(self.u_t-samples).abs()
        
        return self.output_shape(distance)
    
    def __repr__(self):
        return f'{self.__class__.__name__} (utopian={self.utopian.tolist()})'
    
    def save_state(self) -> dict:
        
        obj_dict = {'name': self.__class__.__name__,
                    'state_dict': tensordict_to_dict(self.state_dict()),
                    'targets': [t.save_state() for t in self.targets]}
        
        return obj_dict
    
    @classmethod
    def load_state(cls, obj_dict: dict):
               
        new_obj = cls(targets=[Target.load_state(t_dict) for t_dict in obj_dict['targets']],
                      **obj_dict['state_dict'])
        
        return new_obj


class Bounded_Target(Objective):
    """
    Represents a bounded target objective for multi-output and single-output
    acquisition objectives

    Args:
        bounds (list[tuple[float, float] | None]): The bounds for each target.
            If a bound is None, it is ignored.
        targets (Target | list[Target]): The target or list of targets.
        tau (float, optional): The temperature parameter for the sigmoid function.
            Defaults to 1e-3.

    Attributes:
        lb (torch.Tensor): The lower bounds for the targets.
        ub (torch.Tensor): The upper bounds for the targets.
        tau (torch.Tensor): The temperature parameter for the sigmoid function.
        indices (torch.Tensor): The indices of the non-None bounds.

    """

    def __init__(self,
                 bounds: list[tuple[float, float] | None],
                 targets: Target | list[Target],
                 tau: float = 1e-3) -> None:
        if not isinstance(targets, list):
            targets = [targets]

        if not all(isinstance(t, Target) for t in targets):
            raise TypeError('All elements in the list must be Target objects')

        self.targets = targets

        if len(bounds) != len(targets):
            raise ValueError("Length of bounds and targets must be the same")

        super().__init__(mo=len(targets) > 1)

        # Only store and process the indicated bounds
        indices = []
        lb_t = []
        ub_t = []
        for i, (b, target) in enumerate(zip(bounds, targets)):
            if b is not None:
                lb_t.append(target.transform_f(b[0]).iloc[0])
                ub_t.append(target.transform_f(b[1]).iloc[0])
                indices.append(i)
    
        self.bounds = bounds
        self.indices = torch.tensor(indices)
        self.lb = torch.tensor(lb_t, dtype=TORCH_DTYPE).flatten()
        self.ub = torch.tensor(ub_t, dtype=TORCH_DTYPE).flatten()
        tau = torch.tensor(tau, dtype=TORCH_DTYPE)

        self.register_buffer('tau', tau)

    def __repr__(self):
        return f'{self.__class__.__name__} (indices={self.indices.tolist()}, bounds={self.bounds})'

    def forward(self,
                samples: Tensor,
                X: Tensor | None = None) -> Tensor:
        
        approx_lb = torch.sigmoid((samples[..., self.indices] - self.lb) / self.tau)
        approx_ub = torch.sigmoid((self.ub - samples[..., self.indices]) / self.tau)
        product = approx_lb * approx_ub
        out = samples.to(TORCH_DTYPE)
        out[..., self.indices] = product
        return self.output_shape(out)

    def save_state(self) -> dict:
        
        obj_dict = {'name': self.__class__.__name__,
                    'state_dict': tensordict_to_dict(self.state_dict()),
                    'bounds': self.bounds,
                    'targets': [t.save_state() for t in self.targets]}
        
        return obj_dict
    
    @classmethod
    def load_state(cls, obj_dict: dict):
               
        new_obj = cls(targets=[Target.load_state(t_dict) for t_dict in obj_dict['targets']],
                      bounds=obj_dict['bounds'],
                      **obj_dict['state_dict'])
        
        return new_obj


class Index_Objective(Objective):
    """
    Creates an objective function that returns the single-objective output
    from a multi-output model, at the specified index.
    
    Always a single-output objective.

    Args:
        index (int): The index of the value to be returned.
    """
    def __init__(self,
                 index: int = 0) -> None:
        super().__init__(mo=False)
        self.register_buffer('index', torch.tensor(index))
        
    def __repr__(self):
        return f'{self.__class__.__name__} (index={self.index.item()})'
        
    def forward(self,
                samples: Tensor,
                X: Tensor | None = None) -> Tensor:
        return samples[..., self.index]
