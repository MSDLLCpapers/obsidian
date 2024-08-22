"""Custom objective function formulations"""

from .base import Objective

from obsidian.parameters import ParamSpace, Param_Continuous, Target
from obsidian.utils import tensordict_to_dict
from obsidian.config import TORCH_DTYPE

import torch
from torch import Tensor


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
        """
        Evaluate the objective function on the candidate set samples, X
        """
        return self.output_shape(samples)


class Product_Objective(Objective):
    """
    Objective function that computes the weighted products of other objectives
    
    Args:
        ind (tuple[int]): The indices of objectives to be used in a product
        weights (tuple[float]): The weights corresponding to indexed objectives
        const (float | int): A constant value that can be added to the product
        new_dim (bool, optional): Whether to create a new objective dimension
            from this product. Default is ``False``. Setting to ``True`` will
            make this the only output objective.
        
    """
    def __init__(self,
                 ind: tuple[int],
                 weights: tuple[float],
                 const: float | int = 0,
                 new_dim: bool = True) -> None:
        super().__init__(new_dim)
        # Always MOO if dim is being added, always SOO otherwise
        if len(weights) != len(ind):
            raise ValueError('The length of weights and indices must be the same')
        self.register_buffer('ind', torch.tensor(ind, dtype=torch.int))
        self.register_buffer('weights', torch.tensor(weights, dtype=TORCH_DTYPE))
        self.register_buffer('const', torch.tensor(const, dtype=TORCH_DTYPE))
        self.register_buffer('new_dim', torch.tensor(new_dim, dtype=torch.bool))
        
    def forward(self,
                samples: Tensor,
                X: Tensor | None = None) -> Tensor:
        """
        Evaluate the objective function on the candidate set samples, X
        """
        obj = (self.weights*samples[..., self.ind]).prod(dim=-1, keepdim=True) + self.const
        if self.new_dim:
            return torch.concat((samples, obj), dim=-1)
        else:
            return self.output_shape(obj)


class Divide_Objective(Objective):
    """
    Objective function that computes the weighted quotient of other objectives
    
    (w_num * samples[..., ind_num])/(w_denom * samples[..., ind_denom]) + const
    
    Args:
        ind_num (int): The index of the objective to be used in the numerator
        w_num (float | int, optional): The weights corresponding to numerator
            objective. Defaults to ``1``.
        ind_denom (int): The index of the objective to be used in the denominator
        w_denom (float | int, optional): The weights corresponding to denominator
            objective. Defaults to ``1``.
        const (float | int): A constant value that can be added to the quotient
        new_dim (bool, optional): Whether to create a new objective dimension
            from this quotient. Default is ``False``. Setting to ``True`` will
            make this the only output objective.
        
    """
    def __init__(self,
                 ind_num: int,
                 ind_denom: int,
                 w_num: float | int = 1,
                 w_denom: float | int = 1,
                 const: float | int = 0,
                 new_dim: bool = True) -> None:
        super().__init__(new_dim)
        # Always MOO if dim is being added, always SOO otherwise

        self.register_buffer('ind_num', torch.tensor(ind_num, dtype=torch.int))
        self.register_buffer('w_num', torch.tensor(w_num, dtype=TORCH_DTYPE))
        self.register_buffer('ind_denom', torch.tensor(ind_denom, dtype=torch.int))
        self.register_buffer('w_denom', torch.tensor(w_denom, dtype=TORCH_DTYPE))
        self.register_buffer('const', torch.tensor(const, dtype=TORCH_DTYPE))
        self.register_buffer('new_dim', torch.tensor(new_dim, dtype=torch.bool))
        
    def forward(self,
                samples: Tensor,
                X: Tensor | None = None) -> Tensor:
        """
        Evaluate the objective function on the candidate set samples, X
        """
        w_num = (self.w_num*samples[..., self.ind_num]).unsqueeze(-1)
        w_denom = (self.w_denom*samples[..., self.ind_denom]).unsqueeze(-1)
        obj = w_num/w_denom + self.const
        
        if self.new_dim:
            return torch.concat((samples, obj), dim=-1)
        else:
            return self.output_shape(obj)


class Feature_Objective(Objective):
    """
    Creates an objective function which creates a new outcome as a linear combination of input features.
    
    Always a multi-output objective.

    Args:
        X_space (ParamSpace): The parameter space.
        ind (tuple[int]): The indices of the parameters in the real space
            to be used as features.
        coeff (tuple[float | int], optional): The coefficients corresponding to each feature.
            Defaults to ``[0]``.

    Raises:
        ValueError: If the length of `ind` and `coeff` are not the same.
        TypeError: If the indices for input objectives are not for continuous parameters.
    """
    
    def __init__(self,
                 X_space: ParamSpace,
                 ind: tuple[int],
                 coeff: tuple[float | int]) -> None:
        
        super().__init__(mo=True)
        
        if len(ind) != len(coeff):
            raise ValueError("featureids and coeff must have the same length")
        
        for i in ind:
            if not isinstance(X_space.params[i], Param_Continuous):
                raise TypeError('Indices for input objectives must be for continuous parameters only')
        
        self.register_buffer('coeff', torch.tensor(coeff, dtype=TORCH_DTYPE))
        self.register_buffer('ind', torch.tensor(ind, dtype=torch.long))
        self.X_space = X_space
    
    def __repr__(self):
        """String representation of object"""
        return f'{self.__class__.__name__} (ind={self.ind.tolist()}, coeff={self.coeff.tolist()})'

    def forward(self,
                samples: Tensor,
                X: Tensor) -> Tensor:
        """
        Evaluate the objective function on the candidate set samples, X
        """
        # Ydim = s * b * q * m
        # Xdim = b * q * d
        # if q is 1, it is omitted

        X_u_all = []  # Create unscaled X
        for i in self.ind:
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
        """Saves the objective to a state dictionary"""
        obj_dict = {'name': self.__class__.__name__,
                    'state_dict': tensordict_to_dict(self.state_dict()),
                    'X_space': self.X_space.save_state()}
        
        return obj_dict
    
    @classmethod
    def load_state(cls, obj_dict: dict):
        """Loads the objective from a state dictionary"""
        new_obj = cls(ParamSpace.load_state(obj_dict['X_space']),
                      **obj_dict['state_dict'])
        
        return new_obj


class Utopian_Distance(Objective):
    """
    A custom objective for calculating the distance between an output response
    and a most-desirable, ideal/utopian point.

    Args:
        utopian (tuple[float | int]): A list of values representing the utopian point.
        targets (Target | list[Target]): A single Target object or a list of Target objects.

    Attributes:
        u_t (Tensor): The utopian coordinates in transformed space
    
    Raises:
        TypeError: If the input is not a list of Target objects.
        ValueError: If the length of the utopian point and targets are not the same.

    """

    def __init__(self,
                 utopian: list[float | int],
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
        """
        Evaluate the objective function on the candidate set samples, X
        """
        distance = (-1)*(self.u_t-samples).abs()
        
        return self.output_shape(distance)
    
    def __repr__(self):
        """String representation of object"""
        return f'{self.__class__.__name__} (utopian={self.utopian.tolist()})'
    
    def save_state(self) -> dict:
        """Saves the objective to a state dictionary"""
        obj_dict = {'name': self.__class__.__name__,
                    'state_dict': tensordict_to_dict(self.state_dict()),
                    'targets': [t.save_state() for t in self.targets]}
        
        return obj_dict
    
    @classmethod
    def load_state(cls, obj_dict: dict):
        """Loads the objective from a state dictionary"""
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
            Defaults to ``1e-3``.

    Attributes:
        lb (torch.Tensor): The lower bounds for the targets.
        ub (torch.Tensor): The upper bounds for the targets.
        tau (torch.Tensor): The temperature parameter for the sigmoid function.
        ind (torch.Tensor): The indices of the non-None bounds.

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
        ind = []
        lb_t = []
        ub_t = []
        for i, (b, target) in enumerate(zip(bounds, targets)):
            if b is not None:
                lb_t.append(target.transform_f(b[0]).iloc[0])
                ub_t.append(target.transform_f(b[1]).iloc[0])
                ind.append(i)
    
        self.bounds = bounds
        self.ind = torch.tensor(ind)
        self.lb = torch.tensor(lb_t, dtype=TORCH_DTYPE).flatten()
        self.ub = torch.tensor(ub_t, dtype=TORCH_DTYPE).flatten()
        tau = torch.tensor(tau, dtype=TORCH_DTYPE)

        self.register_buffer('tau', tau)

    def __repr__(self):
        """String representation of object"""
        return f'{self.__class__.__name__} (ind={self.ind.tolist()}, bounds={self.bounds})'

    def forward(self,
                samples: Tensor,
                X: Tensor | None = None) -> Tensor:
        """
        Evaluate the objective function on the candidate set samples, X
        """
        approx_lb = torch.sigmoid((samples[..., self.ind] - self.lb) / self.tau)
        approx_ub = torch.sigmoid((self.ub - samples[..., self.ind]) / self.tau)
        product = approx_lb * approx_ub
        out = samples.to(TORCH_DTYPE)
        out[..., self.ind] = product
        return self.output_shape(out)

    def save_state(self) -> dict:
        """Saves the objective to a state dictionary"""
        obj_dict = {'name': self.__class__.__name__,
                    'state_dict': tensordict_to_dict(self.state_dict()),
                    'bounds': self.bounds,
                    'targets': [t.save_state() for t in self.targets]}
        
        return obj_dict
    
    @classmethod
    def load_state(cls, obj_dict: dict):
        """Loads the objective from a state dictionary"""
        new_obj = cls(targets=[Target.load_state(t_dict) for t_dict in obj_dict['targets']],
                      bounds=obj_dict['bounds'],
                      **obj_dict['state_dict'])
        
        return new_obj


class Index_Objective(Objective):
    """
    Creates an objective function that returns the single-objective output
    from a multi-output model, at the specified index.
    
    Single-objective when index is int; multi-objective if tuple.

    Args:
        ind (int | tuple[int]): The index of the value to be returned.
    """
    def __init__(self,
                 ind: int | tuple[int] = 0) -> None:
        super().__init__(mo=not isinstance(ind, int))
        self.register_buffer('ind', torch.tensor(ind))
        
    def __repr__(self):
        """String representation of object"""
        return f'{self.__class__.__name__} (ind={self.ind.item()})'
        
    def forward(self,
                samples: Tensor,
                X: Tensor | None = None) -> Tensor:
        """
        Evaluate the objective function on the candidate set samples, X
        """
        return samples[..., self.ind]
