"""Custom objective function formulations"""

import torch
from torch import Tensor
from obsidian.parameters import ParamSpace, Param_Continuous
from .base import Objective
from obsidian.surrogates.utils import tensordict_to_dict
from obsidian.utils import TORCH_DTYPE


class Identity_Objective(Objective):
    """
    Dummy multi-output objective function that simply returns the input Y.
    """
    def __init__(self,
                 mo: bool = False) -> None:
        super().__init__(mo)
    
    def forward(self,
                samples: Tensor,
                X: Tensor | None = None) -> Tensor:
        
        return self.output_shape(samples)


class Feature_Objective(Objective):
    """
    Creates an objective function which creates a new outcome as a linear combination of input features.
    
    Always a multi-output objective.

    Args:
        X_space (ParamSpace): The parameter space.
        indices (list[float | int], optional): The indices of the parameters in the real space
            to be used as features. Defaults to [0].
        coeff (list[float | int | list], optional): The coefficients corresponding to each feature.
            Defaults to [0].

    Raises:
        ValueError: If the length of `indices` and `coeff` are not the same.
        TypeError: If the indices for input objectives are not for continuous parameters.
    """
    
    def __init__(self,
                 X_space: ParamSpace,
                 indices: list[float | int],
                 coeff: list[float | int | list]) -> None:
        
        super().__init__(mo=True)
        
        if len(indices) != len(coeff):
            raise ValueError("featureids and coeff must have the same length")
        
        for i in indices:
            if not isinstance(X_space.params[i], Param_Continuous):
                raise TypeError('Indices for input objectives must be for continuous parameters only')
        
        self.register_buffer('coeff', torch.tensor(coeff, dtype=TORCH_DTYPE))
        self.register_buffer('indices', torch.tensor(indices, dtype=torch.long))
        self.X_space = X_space
    
    def __repr__(self):
        return f'{self.__class__.__name__} (indices={self.indices.tolist()}, coeff={self.coeff.tolist()})'

    def forward(self,
                samples: Tensor,
                X: Tensor) -> Tensor:
        # Ydim = s * b * q * m
        # Xdim = b * q * d
        # if q is 1, it is omitted

        X_u_all = []  # Create unscaled X
        for i in self.indices:
            X_u_all.append(torch.tensor(
                self.X_space[i].decode(X[..., i].detach().numpy()),
                dtype=TORCH_DTYPE).unsqueeze(-1))  # Add output dimension
            
        X_u = torch.concat(X_u_all, dim=-1).unsqueeze(0)  # Add sample dimension to X_u
        cX = self.coeff * X_u
        
        # Make sure that the new obejctive matches Y in shape
        # Sum the C*X features as a new output
        feature_obj = cX.sum(dim=-1, keepdim=True).repeat_interleave(samples.shape[0], 0)
        
        total_obj = torch.concat([samples, feature_obj], dim=-1)
        
        return total_obj
    
    def save_state(self) -> dict:
        
        obj_dict = {'name': self.__class__.__name__,
                    'state_dict': tensordict_to_dict(self.state_dict()),
                    'X_space': self.X_space.save_state()}
        
        return obj_dict
    
    @classmethod
    def load_state(cls, obj_dict: dict):
               
        new_obj = cls(ParamSpace.load_state(obj_dict['X_space']),
                      **obj_dict['state_dict'])
        
        return new_obj
