"""Constraints on the input features of a parameter space"""

from obsidian.parameters import IParamSpace, Param_Continuous
from obsidian.constraints import Constraint
from obsidian.config import TORCH_DTYPE

import torch
from torch import Tensor
from abc import abstractmethod

# See  https://botorch.org/api/optim.html, optimize_acqf "equality_constraints", "inequality_constraints",
# and "nonlinear_inequality_constraints"


class Input_Constraint(Constraint):
    """
    Input constraint for a given parameter space.

    Note: Saving and loading input constraints is managed by ParamSpace.
        The interface class IParamSpace is used here to avoid circular imports
        with constraints that depend on ParamSpace, which saves/loads constraints.
    """
    def __init__(self,
                 X_space: IParamSpace):
        super().__init__()
        self.X_space = X_space


class Linear_Constraint(Input_Constraint):
    """
    Input constraint for a given parameter space.

    Note: SUM(LHS) <= RHS equivalent to -SUM(LHS) >= -RHS

    Linear constraints must return:
        tuple = (indices = Tensor, coefficients = Tensor, rhs = float)
        where sum_i(X[indices[i]] * coefficients[i]) = rhs (equality) or >= rhs (inequality)

    Attributes:
        X_space (ParamSpace): The parameter space object.
        ind (list[float | int], optional): The indices of the parameters to be constrained.
            Defaults to ``[0]``.
        weights (list[float | int | list], optional): The coefficients for the parameters in the
            constraint equation. Defaults to ``[1]``.
        rhs (int | float, optional): The right-hand side value of the constraint equation.
            Defaults to ``-1``.
        equality (bool, optional): Whether the constraint is an equality (=)  or inequality (>=)
            constraint. Defaults to ``False`` for inequality constraint.

    """
    def __init__(self,
                 X_space: IParamSpace,
                 ind: list[float | int] = [0],
                 weights: list[float | int] = [1],
                 rhs: int | float = -1,
                 equality: bool = False) -> None:
        super().__init__(X_space)

        self.register_buffer('ind', torch.tensor(ind, dtype=torch.int64))
        self.register_buffer('weights', torch.tensor(weights, dtype=TORCH_DTYPE))
        self.register_buffer('rhs', torch.tensor(rhs, dtype=TORCH_DTYPE))
        self.register_buffer('equality', torch.tensor(equality, dtype=torch.bool))

        # Determine which parameters in the decoded space are indicated
        self.ind_t = torch.tensor([X_space.t_map[i] for i in ind],
                                  dtype=torch.int64)
        for i in ind:
            if not isinstance(X_space.params[i], Param_Continuous):
                raise TypeError('Indeces for input constraints must be \
                                for continuous parameters only')

        # We neeed to write the constraints on the encoded space
        # To convert (raw -> encoded) for a continuous param with min-max scaling:
        #      w_t = w * range
        #      rhs_t = rhs - sum(w_t * min)
        weights_t = []
        rhs_t = rhs
        for w, i in zip(weights, ind):
            param_i = X_space.params[i]
            rhs_t -= param_i.min*w
            weights_t.append(w * param_i.range)
        self.weights_t = torch.tensor(weights_t, dtype=TORCH_DTYPE)
        self.rhs_t = torch.tensor(rhs_t, dtype=TORCH_DTYPE)

    def forward(self) -> tuple[Tensor, Tensor, Tensor]:
        """
        Forward method for the input constraint.

        Forms the linear constraint in the tuple that can be handled by BoTorch optimizer.

        Returns:
            tuple[Tensor, Tensor, Tensor]: A tuple containing the indices, coefficients,
                and right-hand side value of the constraint in the encoded space.

        """
        return self.ind_t, self.weights_t, self.rhs_t


class Nonlinear_Constraint(Input_Constraint):
    """
    Abstract class for nonlinear constraints

    Nonlinear inequality constraints must return:
        tuple = (constraint(x) = callable, intra-point = bool)

        where:
            Intra point: callable takes X in (d) and return scalar
            Intra point: when q = 1, or when applying the same constraint to each
                candidate in the batch
            Inter point: callable takes X in (q x d) and returns scalar
            Inter point: When q > 1 and the constraints are applied to an entire
                batch of candidates
        
        Use 1-d tensor of indices for intra point constraint
        Use 2-d tensor of indices for inter point constraint where indices[i] = (k_i, l_i, ...)

    """
    @abstractmethod
    def forward(self) -> tuple[callable, bool]:
        """
        Forward method for the input constraint.

        Returns:
            tuple[callable, bool]: A tuple containing the callable function and a boolean
                indicating whether the constraint is an intra-point constraint.

        """
        pass  # pragma: no cover


class BatchVariance_Constraint(Nonlinear_Constraint):
    """
    Constraint how much one parameter can vary over a batch of optimized points
    
    Attributes:
        X_space (ParamSpace): The parameter space object.
        indices (list[float | int], optional): The indices of the parameters to be constrained.
            Defaults to ``[0]``.
        coeff (list[float | int | list], optional): The coefficients for the parameters in the
            constraint equation. Defaults to ``[1]``.
        rhs (int | float, optional): The right-hand side value of the constraint equation.
            Defaults to ``-1``.

    """
    def __init__(self,
                 X_space: IParamSpace,
                 ind: int,
                 tol: int | float = 0.01) -> None:
        
        super().__init__(X_space)

        self.register_buffer('ind', torch.tensor(ind, dtype=torch.int64))
        self.register_buffer('tol', torch.tensor(tol, dtype=TORCH_DTYPE))

        # Determine which parameters in the decoded space are indicated
        self.ind_t = torch.tensor(X_space.t_map[ind], dtype=torch.int64)
        if not isinstance(X_space.params[ind], Param_Continuous):
            raise TypeError('Indeces for input constraints must be \
                            for continuous parameters only')

    def forward(self):

        def nl_func(X: Tensor) -> Tensor:
            X_range = X.max(dim=0).values - X.min(dim=0).values
            target_range = X_range[self.ind_t]
            return self.tol - target_range
        
        nonlinear_constraint = (nl_func, False)  # False for inter-point

        return nonlinear_constraint
