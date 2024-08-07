"""Constraints on the input features of a parameter space"""

from obsidian.parameters import ParamSpace, Param_Continuous
from obsidian.config import TORCH_DTYPE

import torch
from torch import Tensor

# See  https://botorch.org/api/optim.html, optimize_acqf "equality_constraints", "inequality_constraints",
# and "nonlinear_inequality_constraints"

# Equality = list(tuple) / tuple = {indices = Tensor, coefficients = Tensor, rhs = float} / sum_i(X[indices[i]] * coefficients[i]) = rhs

# Inquality = list(tuple) / tuple = (indices = Tensor, coefficients = Tensor, rhs = float) / sum_i(X[indices[i]] * coefficients[i]) >= rhs
# Use 1-d tensor of indices for intra point constraint
# Use 2-d tensor of indices for inter point constraint where indices[i] = (k_i, l_i, ...)

# Nonlinear inequality = list(tuple) / tuple = (constraint(x) = callable, intra-point = bool)
# Intra piont: callable takes X in (d) and return scalar
# Intra point: when q = 1, or when applying the same constraint to each candidate in the batch
# Inter point: callable takes X in (q x d) and returns scalar
# Inter point: When q > 1 and the constraints are applied to an entire batch


def InConstraint_Generic(X_space: ParamSpace,
                         indices: list[float | int] = [0],
                         coeff: list[float | int | list] = [0],
                         rhs: int | float = -1) -> tuple[Tensor, Tensor, Tensor]:
    """
    Creates an input constraint for a given parameter space.

    Args:
        X_space (ParamSpace): The parameter space object.
        indices (list[float | int], optional): The indices of the parameters to be constrained. Defaults to [0].
        coeff (list[float | int | list], optional): The coefficients for the parameters in the constraint equation. Defaults to [0].
        rhs (int | float, optional): The right-hand side value of the constraint equation. Defaults to -1.

    Returns:
        tuple[Tensor, Tensor, Tensor]: A tuple containing the indices, coefficients, and right-hand side value of the constraint.

    Raises:
        TypeError: If X_space is not an obsidian ParamSpace
        TypeError: If indices or coeff are not a list
        TypeError: If rhs is not numeric
        TypeError: If indices are not continuous parameters

    Notes:
        - The constraint equation is of the form: coeff_X1 * X1 + coeff_X2 * X2 + ... = rhs
        - The indices and coefficients should correspond to continuous parameters in the parameter space.
        - The constraint equation is transformed to the encoded space before returning.

    Example:
        X_space = ParamSpace(...)
        indices = [0, 1]
        coeff = [1, -2]
        rhs = 10
        constraint = InConstraint_Generic(X_space, indices, coeff, rhs)
    """
    
    if not isinstance(X_space, ParamSpace):
        raise TypeError('X_space must be a ParamSpace object')
    if not isinstance(indices, list):
        raise TypeError('Indices must be a list')
    if not isinstance(coeff, list):
        raise TypeError('Coefficients must be a list')
    if not isinstance(rhs, (int, float)):
        raise TypeError('RHS must be a scalar')

    # Determine which parameters in the decoded space are indicated
    i_t = [X_space.tinv_map[i] for i in indices]

    for i in i_t:
        if not isinstance(X_space.params[i], Param_Continuous):
            raise TypeError('Indeces for input constraints must be \
                            for continuous parameters only')

    # We neeed to write the constraints on the encoded space
    # SUM(LHS) <= RHS equivalent to -SUM(LHS) >= -RHS

    c_t = []
    rhs_t = rhs
    for c, i in zip(coeff, i_t):
        param_i = X_space.params[i]
        rhs_t -= param_i.min*c
        c_t.append(c * param_i.range)

    linear_constraint = (torch.tensor(indices),
                         torch.tensor([float(c) for c in c_t], dtype=TORCH_DTYPE),
                         torch.tensor(rhs_t, dtype=TORCH_DTYPE))

    return linear_constraint


def InConstraint_ConstantDim(X_space: ParamSpace,
                             dim: int,
                             tol: int | float = 0.01) -> tuple[callable, bool]:
    """
    Constraint which maintains one parameter at a relatively constant (but still freely optimized)
    value within a set of suggestions. Useful for simplifying operational complexity of large-
    scale experimental optimization (e.g. temperature on a well plate)

    Args:
        X_space (ParamSpace): The parameter space object.
        dim (int): The dimension of the constant parameter.
        tol (int | float, optional): The tolerance value. Defaults to 0.01.

    Returns:
        tuple[callable, bool]: A tuple containing the constraint function and a boolean indicating if it is an inter-point constraint.

    Raises:
        TypeError: If X_space is not an obsidian ParamSpace
        TypeError: If dim is not an integer
        TypeError: If tol is not numeric
        TypeError: If the dimension is not a continuous parameter
        
    """

    if not isinstance(X_space, ParamSpace):
        raise TypeError('X_space must be a ParamSpace object')
    if not isinstance(dim, int):
        raise TypeError('Dimension must be an integer')
    if not isinstance(tol, (int, float)):
        raise TypeError('Tolerance must be a scalar')
    if not isinstance(X_space.params[dim], Param_Continuous):
        raise TypeError('Constant dimension must be a continuous parameter')

    t_dim = X_space.t_map[dim]

    def nl_func(X: Tensor) -> Tensor:
        X_range = X.max(dim=0).values - X.min(dim=0).values
        target_range = X_range[t_dim]
        return tol - target_range

    nonlinear_constraint = (nl_func, False)  # False for inter-point
    return nonlinear_constraint
