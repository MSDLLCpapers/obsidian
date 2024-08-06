"""Scalarization methods for reducing multi-output to single-output objectives"""

import torch
from torch import Tensor
from .base import Objective
from obsidian.utils import TORCH_DTYPE

# Reference: https://arxiv.org/pdf/1904.05760


class Scalarization(Objective):
    """
    Base scalarization objective, which condenses multiple outputs into a single one
    
    Always a single-output objective.
    """
    
    def __init__(self) -> None:
        super().__init__(mo=False)
        
    def __repr__(self):
        return f'{self.__class__.__name__} (weights={self.weights.tolist()})'


class Scalar_WeightedSum(Scalarization):
    """
    Scalarizes a multi-output response using a weighted sum

    Args:
        weights (list[float]): A list of weights to be applied to the response tensor.
        
    """
    def __init__(self,
                 weights: list[float]):
        super().__init__()
        self.register_buffer('weights', torch.tensor(weights, dtype=TORCH_DTYPE))
    
    def forward(self,
                samples: Tensor,
                X: Tensor | None = None) -> Tensor:
        
        return (self.weights * samples).sum(dim=-1)


class Scalar_WeightedNorm(Scalarization):
    """
    Scalarizes a multi-output response using a weighted norm

    Args:
        weights (list[float]): A list of weights to be applied to the response tensor.
        norm (int or None, optional): The order of vector norm to be used. If None
            is provided, the p-norm will be used
        neg (bool, optional): Whether or not to return the negative norm, which is
            required for maximizing norms based on distance to a target (utopian point).
            
    """
    def __init__(self,
                 weights: list[float],
                 norm: int | None = None,
                 neg: bool = False):
        super().__init__()
        self.register_buffer('weights', torch.tensor(weights, dtype=TORCH_DTYPE))
        
        if norm is None:
            norm = len(weights)
        self.register_buffer('norm', torch.tensor(norm, dtype=torch.long))
        
        self.register_buffer('neg', torch.tensor(neg, dtype=torch.bool))
        self.C = -1 if neg else 1

    def forward(self,
                samples: Tensor,
                X: Tensor | None = None) -> Tensor:
        
        return self.C*(self.weights * samples).norm(self.norm, dim=-1)


class Scalar_Chebyshev(Scalarization):
    """
    Scalarizes a multi-output response using the augmented Chebyshev function.

    The augmented Chebyshev function maximizes the minimum scaled-response in conjunction with a weighted sum.

    Args:
        weights (list[float]): A list of weights to be applied to the response tensor.
        alpha (float, optional): The scaling factor for the weighted sum. Defaults to 0.05.
        augment (bool, optional): Flag indicating whether to perform augmentation. Defaults to True.
            
    """
    def __init__(self,
                 weights: list[float],
                 alpha: float = 0.05,
                 augment: bool = True):
        super().__init__()
        self.register_buffer('weights', torch.tensor(weights, dtype=TORCH_DTYPE))
        self.register_buffer('alpha', torch.tensor(alpha, dtype=TORCH_DTYPE))
        self.register_buffer('augment', torch.tensor(augment, dtype=torch.bool))
        
    def forward(self,
                samples: Tensor,
                X: Tensor | None = None) -> Tensor:
        
        # Augmentation seeks to maximize weighted sum
        delta = self.alpha * (self.weights * samples).sum(dim=-1) if self.augment else 0
        
        # Maximize the smallest weighted outcome
        return (-1 * self.weights * samples).max(dim=-1).values + delta
