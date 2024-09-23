"""A collection of parameters jointly defining an operating or optimization space"""

from .base import (
    IParamSpace,
    Parameter
)
from .continuous import (
    Param_Continuous,
    Param_Observational
)
from .discrete import (
    Param_Categorical,
    Param_Ordinal,
    Param_Discrete,
    Param_Discrete_Numeric,
    Task,
    CAT_SEP
)

from .config import param_class_dict
from obsidian.constraints import (
    Input_Constraint,
    Linear_Constraint,
    Nonlinear_Constraint,
    const_class_dict
)

from obsidian.exceptions import UnsupportedError
from obsidian.utils import tensordict_to_dict

import torch
import numpy as np
import pandas as pd


class ParamSpace(IParamSpace):
    """
    Class designed to define the parameter space in which an optimization
    can be conducted.

    Attributes:
        params (tuple[Parameter]): A tuple of Parameter types defining the parameter space.
        X_names (tuple[str]): A tuple of the names of the parameters.
        X_cont (list[str]): A list of the names of the continuous parameters.
        X_obs (list[str]): A list of the names of the observational parameters.
        X_discrete (dict[str, list[str]]): A dictionary mapping the names of the discrete parameters to their categories.
        X_task (dict[str, list[str]]): A dictionary mapping the names of the task parameters to their categories.
        X_min (torch.Tensor): A tensor containing the minimum values of the parameters.
        X_max (torch.Tensor): A tensor containing the maximum values of the parameters.
        X_range (torch.Tensor): A tensor containing the range of the parameters.
        X_static (list[str]): A list of the names of the static parameters.
        X_vary (list[str]): A list of the names of the varying parameters.
        n_dim (int): The number of dimensions in the parameter space.
        n_tdim (int): The total number of dimensions in the parameter space after transformation.
        t_map (dict): A dictionary mapping parameter indices to transformed indices.
        tinv_map (dict): A dictionary mapping transformed indices to parameter indices.
        X_discrete_idx (list[int]): A list of the indices of the discrete parameters.
        X_t_discrete_idx (list[int]): A list of the indices of the transformed discrete parameters.
        X_t_cat_idx (list[int]): A list of the indices of the transformed categorical parameters.
        X_t_task_idx (int): The index of the transformed task parameter.
        search_space (pd.DataFrame): The allowable search space for future optimization.
    
    Raises:
        ValueError: If the X_names are not unique.
        UnsupportedError: If there is more than one task parameter.

    """
    def __init__(self,
                 params: list[Parameter]):
        
        # Convert to immutable dtype to presever order
        self.params = tuple(params)
        self.X_names = tuple([x.name for x in self])
        if len(set(self.X_names)) != len(self.X_names):
            raise ValueError("X_names must be unique.")
        if any([CAT_SEP in x for x in self.X_names]):
            raise ValueError(f"X_names cannot contain '{CAT_SEP}'.")

        # Save as attributes key types that affect usage behavior
        self.X_cont = [x for x in self if isinstance(x, Param_Continuous)
                       and not isinstance(x, Param_Observational)]
        self.X_obs = [x for x in self if isinstance(x, Param_Observational)]
        self.X_discrete = [x for x in self if isinstance(x, Param_Discrete)]
        self.X_task = [x for x in self if isinstance(x, Task)]
        if len(self.X_task) > 1:
            raise UnsupportedError("Only one task parameter is allowed.")
        
        # Locally calculate other types for counting

        X_cat = [x for x in self if isinstance(x, Param_Categorical)]
        X_ord = [x for x in self if isinstance(x, Param_Ordinal)]
        X_disc_num = [x for x in self if isinstance(x, Param_Discrete_Numeric)]
        
        # Extract ranges for scaling
        self.X_min = torch.Tensor([x.min for x in self])
        self.X_max = torch.Tensor([x.max for x in self])
        self.X_range = self.X_max - self.X_min

        # Extract where variables are changing or are fixed/observational
        self.X_static = [x.name for x in self if x.min == x.max or isinstance(x, Param_Observational)]
        self.X_vary = [name for name in self.X_names if name not in self.X_static]

        # Extract number of dimensions (raw input and encoded)
        self.n_dim = len(self.X_names)
        # Total number of dimensions after transform
        # Numerical + Observational + Unique Categories + Ordinal + Discrete_Numeric
        self.n_tdim = int(len(self.X_cont)
                          + len(self.X_obs)
                          + np.sum([len(x.categories) for x in X_cat])
                          + len(X_ord)
                          + len(X_disc_num))
        
        # Assign the inv/transform maps
        self.t_map = self.map_transform()
        self.tinv_map = self.map_inv_transform()

        # Save the locations of discrete variables
        self.X_discrete_idx = [i for i in self.t_map.keys() if self[i] in self.X_discrete]
        self.X_t_discrete_idx = [i_t for i_t, i in self.tinv_map.items() if self[i] in self.X_discrete]
        self.X_t_cat_idx = [i_t for i_t, i in self.tinv_map.items() if self[i] in X_cat]
        if self.X_task:
            self.X_t_task_idx = next(i_t for i_t, i in self.tinv_map.items() if self[i] in self.X_task)
        else:
            self.X_t_task_idx = None
        
        # Set up storage for constraints
        self.linear_constraints = []
        self.nonlinear_constraints = []

        return

    def map_transform(self) -> dict:
        """
        Maps the parameter indices to transformed indices based on the parameter types.

        Returns:
            dict: A dictionary mapping parameter indices to transformed indices.
        """
        t_map = {}
        count = 0
        for i, param_i in enumerate(self.params):
            if isinstance(param_i, Param_Categorical):
                t_map[i] = [count + j for j in range(len(param_i.categories))]
                count += len(param_i.categories) - 1
            else:
                t_map[i] = count
            count += 1

        return t_map
    
    def map_inv_transform(self) -> dict:
        """
        Maps the inverse of the transformed dictionary.

        Returns:
            dict: A dictionary where the keys are the original values and the values are the original keys.
        """
        t_map = self.map_transform()
        tinv_map = {}
        for k, v in t_map.items():
            if isinstance(v, list):
                for i in v:
                    tinv_map[i] = k
            else:
                tinv_map[v] = k

        return tinv_map
    
    # Establish a handler method to transform X based on type
    def _transform(self,
                   X: pd.DataFrame,
                   type: str) -> pd.DataFrame:
        """
        Transforms the input DataFrame `X` based on the specified `type`.

        Args:
            X (pd.DataFrame): The input DataFrame to be transformed.
            type (str): The type of transformation to be applied.

        Returns:
            pd.DataFrame: The transformed DataFrame.

        Raises:
            KeyError: If an unknown column is passed in `X`.
        """
        cols = X.columns
        # Check for column match, but consider that categorical OH encoding will lead to Parameter N_i type names
        if not all(col.split(CAT_SEP)[0] in self.X_names for col in cols):
            raise KeyError('Unknown column passed in X')
        X_t = pd.DataFrame()
        for param_i in self:
            methods = {'unit_map': param_i.unit_map,
                       'unit_demap': param_i.unit_demap,
                       'encode': param_i.encode,
                       'decode': param_i.decode}
            # Include all matches, including OH columns (e.g. Parameter N_i)
            cols = [col for col in X.columns if param_i.name in col]
            # Skip if one of the possible parameters is excluded from the transform operation
            if cols != []:
                X_t_i = pd.DataFrame(methods[type](X[cols].values))
                X_t_i.columns = [param_i.name] if not (
                    isinstance(param_i, Param_Categorical) and type == 'encode') else X_t_i.columns
                X_t = pd.concat([X_t, X_t_i], axis=1)

        if type in ['encode', 'unit_map']:
            X_t = X_t.apply(pd.to_numeric)  # Do not send object dtypes for these transforms

        return X_t

    # Define transformation methods using a handler method and lambdas; more readable on outer layer
    def unit_map(self, X):
        """Map from measured to 0,1 space"""
        return self._transform(X, type='unit_map')
    
    def unit_demap(self, X):
        """Map from 0,1 space to measured space"""
        return self._transform(X, type='unit_demap')
    
    def encode(self, X):
        """Encode parameter to a format that can be used for training"""
        return self._transform(X, type='encode')

    def decode(self, X):
        """Decode parameter from transformed space"""
        return self._transform(X, type='decode')

    @property
    def search_space(self) -> pd.DataFrame:
        """
        Returns the search space for the parameter space.

        Returns:
            pd.DataFrame: A dataframe containing the search space for the parameter space.
        """
        
        # Establish the boundaries in real space
        X_search_t = pd.DataFrame()
        for param in self:
            # For continuous, encode the continuous bounds
            if isinstance(param, Param_Continuous):
                cont_bounds = pd.DataFrame(param.encode([param.search_min, param.search_max]), columns=[param.name])
                X_search_t = pd.concat([X_search_t, cont_bounds], axis=1)
                
            # For discrete, encode the available categories, then log the min-max of encoded columns
            elif isinstance(param, Param_Discrete) and (not isinstance(param, (Param_Ordinal,Param_Discrete_Numeric))):
                # Discrete parameter bounds aren't actually handled here; they are handled in optimizer._fixed_features())
                cat_e = param.encode(param.search_categories)
                disc_bounds = pd.DataFrame(np.vstack([[0]*cat_e.shape[-1], cat_e.max().values]),
                                           columns=cat_e.columns)
                X_search_t = pd.concat([X_search_t, disc_bounds], axis=1)
            elif isinstance(param, (Param_Ordinal,Param_Discrete_Numeric)):
                cat_e = param.encode(param.search_categories)
                cont_bounds = pd.DataFrame([min(cat_e), max(cat_e)], columns=[param.name])
                X_search_t = pd.concat([X_search_t, cont_bounds], axis=1)
        
        return X_search_t

    def open_search(self) -> None:
        """Set the search space to the parameter space"""
        for param in self:
            param.open_search()
    
    def mean(self) -> pd.DataFrame:
        """
        Calculates the mean values for each parameter in the parameter space.

        Returns:
            pd.DataFrame: A DataFrame containing the mean values for each parameter.
        """
        row = {}
        for param_i in self:
            if isinstance(param_i, Param_Continuous):  # Mean of continuous
                row[param_i.name] = param_i.unit_demap([0.5])[0]
            elif isinstance(param_i, Param_Discrete):  # First of discrete
                row[param_i.name] = param_i.categories[0]

        df_mean = pd.DataFrame([row])

        return df_mean

    def constrain_inputs(self,
                         constraint: Input_Constraint) -> None:
        """
        Constrains the input space based on the specified equality, inequality, or nonlinear constraint.

        Args:
            constraint (Input_Constraint): The constraint to be applied to the input space.
        """
        if isinstance(constraint, Linear_Constraint):
            self.linear_constraints.append(constraint)
        elif isinstance(constraint, Nonlinear_Constraint):
            self.nonlinear_constraints.append(constraint)
        
        return

    def clear_constraints(self) -> None:
        """Clears all constraints from the input space."""
        self.linear_constraints = []
        self.nonlinear_constraints = []
        return

    def save_state(self) -> dict:
        """
        Saves the state of the ParamSpace object.

        Returns:
            dict: A dictionary containing the state of the ParamSpace object.
        """
        obj_dict = {}
        for param in self:
            obj_dict[param.name] = {'class': param.__class__.__name__,
                                    'state': param.save_state()}
            
        if getattr(self, 'linear_constraints', []):
            obj_dict['linear_constraints'] = [{'class': const.__class__.__name__,
                                               'state': tensordict_to_dict(const.state_dict())}
                                              for const in self.linear_constraints]
        if getattr(self, 'nonlinear_constraints', []):
            obj_dict['nonlinear_constraints'] = [{'class': const.__class__.__name__,
                                                  'state': tensordict_to_dict(const.state_dict())}
                                                 for const in self.nonlinear_constraints]

        return obj_dict
    
    @classmethod
    def load_state(cls,
                   obj_dict: dict):
        """
        Loads the state of the ParamSpace object from a dictionary.

        Args:
            obj_dict (dict): A dictionary containing the state of the ParamSpace object.

        Returns:
            ParamSpace: A new ParamSpace object with the loaded state.

        """

        params = []
        for param, param_dict in obj_dict.items():
            if 'constraint' not in param:
                param = param_class_dict[param_dict['class']].load_state(param_dict['state'])
                params.append(param)

        new_X_space = cls(params=params)

        if 'linear_constraints' in obj_dict:
            for const_dict in obj_dict['linear_constraints']:
                const = const_class_dict[const_dict['class']](new_X_space, **const_dict['state'])
                new_X_space.constrain_inputs(const)
        if 'nonlinear_constraints' in obj_dict:
            for const_dict in obj_dict['nonlinear_constraints']:
                const = const_class_dict[const_dict['class']](new_X_space, **const_dict['state'])
                new_X_space.constrain_inputs(const)

        return new_X_space
