"""Optimizer class definition"""

from obsidian.parameters import ParamSpace, Target, Param_Observational
from obsidian.exceptions import UnsupportedError

from botorch.utils.multi_objective.box_decompositions.dominated import DominatedPartitioning
from botorch.utils.multi_objective.pareto import is_non_dominated
from abc import ABC, abstractmethod

import pandas as pd
import numpy as np
import torch
import random
from torch import Tensor


class Optimizer(ABC):
    """
    Base class for obsidian optimizer, which fits a surrogate model to data and suggests
    optimal experiments

    Attributes:
        X_space (ParamSpace): obsidian ParamSpace object representing the allowable space for optimization.
        seed (int | None): Randomization seed for the optimizer and stochastic surrogate models.
        verbose (int): Flag for monitoring and debugging optimization output.

    Raises:
        ValueError: If verbose is not set to 0, 1, 2, or 3.
        TypeError: If X_space is not an obsidian ParamSpace
    """

    def __init__(self,
                 X_space: ParamSpace,
                 seed: int | None = None,
                 verbose: int = 1):
        
        # Verbose selection
        if verbose not in [0, 1, 2, 3]:
            raise ValueError('Verbose option must be 0 (no output), 1 (summary output), \
                             2 (detailed output), or 3 (debugging)')
        self.verbose = verbose

        # Handle randomization seed, considering all 3 sources (torch, random, numpy)
        self.seed = seed
        if self.seed is not None:
            torch.manual_seed(self.seed)
            if not torch.cuda.is_available():
                torch.use_deterministic_algorithms(True)
            np.random.seed(self.seed)
            random.seed(self.seed)

        # Store the parameter space which contains useful reference properties
        if not isinstance(X_space, ParamSpace):
            raise TypeError('X_space must be an obsidian ParamSpace object')
        self.set_X_space(X_space)

    @property
    def X_space(self):
        """
        ParamSpace: The parameter space defining the search space for the optimization.
        """
        return self._X_space
    
    def set_X_space(self, X_space: ParamSpace):
        self._X_space = X_space
        return

    def _fixed_features(self,
                        fixed_var: dict | None = None) -> list:
        """
        Returns a list of fixed features for optimization.

        Args:
            fixed_var (dict): A dictionary of fixed variables and their corresponding settings.
                The dictionary should be in the format: {param_name: param_setting}.

        Returns:
            list: A list of fixed features for optimization.

        Raises:
            TypeError: If fixed_var is not a dictionary.
            NameError: If a parameter in fixed_var is not found in X_space.
        """

        # Set static optimization features (used for fitting, but set to a fixed number during optimization)
        # e.g. Optimizing for max yield at a fixed end-time in a reaction

        fixed_var = {} if fixed_var is None else fixed_var

        fixed_features_list = []

        # Add any observational variables to fixed_var
        if self.X_space.X_obs:
            fixed_obs = {p.name: p.design_point for p in self.X_space if isinstance(p, Param_Observational)}
            fixed_var.update(fixed_obs)

        # Store each validate setting list in a dataframe, then merge by cartesian product to get all valid combos
        df_list = []

        if fixed_var:
            if not isinstance(fixed_var, dict):
                raise TypeError('Fixed variables must be provided as a dict: {param_name:param_setting}')
            for var, level in fixed_var.items():
                if var not in self.X_space.X_names:
                    raise NameError(f'Profile variable {var} not found in {self.X_space.X_names}')
                param_i = next(param for param in self.X_space if param.name == var)
                param_i._validate_value(level)
                df_i = pd.DataFrame({var: [level]})
                df_list.append(df_i)

        # If we have discrete parameters, we need to limit the acquisition function search as such
        # First, get the cartesian product of all of the categorical/ordinal combos
        for x in self.X_space.X_discrete:
            if x.name not in fixed_var.keys():  # Fixed_var should take precedent and lock out other combinations
                df_i = pd.DataFrame({x.name: x.categories})
                df_list.append(df_i)
        
        # Merge by cross
        if df_list:
            df_all = df_list[0]
            for df_i in df_list[1:]:
                df_all = pd.merge(df_all, df_i, how='cross')
            encoded_cross = self.X_space.encode(df_all)

            # Fixed feature list requires names to be the indeces of our columns
            X_t_names = self.X_t_train.columns
            encoded_cross = encoded_cross.rename(columns={col: X_t_names.get_loc(col) for col in encoded_cross.columns})
            fixed_features_list += encoded_cross.to_dict(orient='records')

        return fixed_features_list

    def hypervolume(self,
                    f: Tensor,
                    ref_point: list | None = None,
                    weights: list | None = None,
                    ) -> float:
        """
        Calculates the hypervolume of the given data points.

        Args:
            f (Tensor): The data points to calculate the hypervolume for.
            ref_point (list, optional): The reference point for the hypervolume calculation. Defaults to ``None``.
            weights (list, optional): The weights to apply to each objective. Defaults to ``None``.

        Returns:
            float: The hypervolume value.

        Raises:
            UnsupportedError: If the number of objectives is less than or equal to 1.
        """

        if f.shape[1] <= 1:
            raise UnsupportedError('Cannot calculate hypervolume for single objective')
        if f.ndim != 2:
            raise ValueError('Hypervolume calculation only supported for 2D tensor')

        if ref_point is None:
            ref_point = f.min(dim=0).values
        else:
            if len(ref_point) != f.shape[1]:
                raise ValueError('Reference point must have the same number of objectives as the data')
            # Need to transform ref_point if provided
            ref_point = torch.tensor(ref_point)

        if weights is None:
            weights = torch.ones(size=(f.shape[1],))
        else:
            if len(weights) != f.shape[1]:
                raise ValueError('Weights must have the same number of objectives as the data')
            weights = torch.tensor(weights)

        f_weighted = f * weights
        bd = DominatedPartitioning(ref_point=ref_point, Y=f_weighted)
        hv = bd.compute_hypervolume().item()

        return hv

    def pareto(self,
               f: Tensor) -> list[bool]:
        """
        Determines the Pareto dominance of a given set of solutions.

        Args:
            f (Tensor): The input series containing the solutions.

        Returns:
            list[bool]: A list of boolean values indicating whether each solution is Pareto optimal or not.
        """
        
        if f.shape[1] <= 1:
            raise UnsupportedError('Cannot calculate pareto front for single objective')
        if f.ndim != 2:
            raise ValueError('Pareto front calculation only supported for 2D tensor')

        return is_non_dominated(f).tolist()
    
    def pf_distance(self,
                    y: pd.Series) -> Tensor:
        """
        Calculates the pairwise distance between the given input `y` and the Pareto front.

        Args:
            y (pd.Series): The input data.

        Returns:
            Tensor: The minimum distance between `y` and the Pareto front.
        """
        f = torch.tensor(pd.concat([t.transform_f(y[t.name]) for t in self.target], axis=1).to_numpy())
        pf = self.pareto(torch.tensor(self.f_train.values))
        f_pareto = torch.tensor(self.f_train.values)[pf]
        
        f = f.unsqueeze(1)  # Insert "to_xdim" before "from_xdim"
        f_pareto = f_pareto.unsqueeze(0)  # Insert "from_xdim" before "to_xdim"
        
        pairwise_distance = torch.norm(f - f_pareto, p=2, dim=-1)
        min_distance = pairwise_distance.min(dim=1).values
        
        return min_distance

    @abstractmethod
    def fit(self,
            Z: pd.DataFrame,
            target: Target | list[Target]):
        """Fit the optimizer's surrogate models to data"""
        pass  # pragma: no cover

    @abstractmethod
    def predict(self,
                X: pd.DataFrame,
                return_f_inv: bool = True,
                PI_range: float = 0.7):
        """Predict the optimizer's target(s) at the candidate set X"""
        pass  # pragma: no cover
    
    @abstractmethod
    def suggest(self):
        """Suggest the next optimal experiment(s)"""
        pass  # pragma: no cover
    
    @abstractmethod
    def maximize(self):
        """Maximize the optimizer's target(s)"""
        pass  # pragma: no cover

    @abstractmethod
    def save_state(self):
        """Save the optimizer to a state dictionary"""
        pass  # pragma: no cover
    
    @classmethod
    @abstractmethod
    def load_state(cls,
                   obj_dict: dict):
        """Load the optimizer from a state dictionary"""
        pass  # pragma: no cover
