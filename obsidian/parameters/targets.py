"""Output features to be optimized"""

import pandas as pd
import numpy as np
import torch
from .transforms import f_transform_dict
from numpy.typing import ArrayLike

from obsidian.exceptions import UnfitError


class Target():
    """
    Base class for optimization response targets.
    """

    def __init__(self,
                 name: str,
                 f_transform: str | None = 'Standard',
                 aim: str = 'max'):

        self.name = name
        if aim not in ['min', 'max']:
            raise ValueError('Aim must be either "min" or "max"')
        if aim == 'min':
            self.multiplier = -1
        else:
            self.multiplier = 1
        self.aim = aim
        
        # Ouput scoring, used for transformation OR to create a cost function of multiple outputs/inputs
        if f_transform is not None:
            if f_transform not in f_transform_dict.keys():
                raise KeyError(f'Scoring function must be selected from one of: {f_transform_dict.keys()}')
        else:
            f_transform = 'Identity'
        self.f_transform = f_transform

    def __repr__(self):
        """String representation of object"""
        return f"{self.__class__.__name__}({self.name}, aim={self.aim})"

    def transform_f(self,
                    f: float | int | ArrayLike,
                    inverse=False,
                    fit=False):
        """
        Converts a raw response to an objective function value ("score").
        Cost-penalization and response transformation should be handled here.

        Args:
            f (array-like): The column(s) containing the response values (y)
            inverse (bool, optional): An indicator to perform the inverse transform. Defaults to ``False``.
            fit (bool, optional): An indicator to fit the properties of the transform function. Defaults to ``False``.

        Returns:
            pd.Series: An array of transformed f values matching the responses in Z

        Raises:
            TypeError: If f is not numeric or array-like
            UnfitError: If the transform function is called without being fit first
        """

        if not (isinstance(f, (pd.Series, pd.DataFrame, np.ndarray, list, float, int))
                or torch.is_tensor(f)):
            raise TypeError('f being transformed must be numeric or array-like')
        
        # Convert everything to numpy except Tensors
        if isinstance(f, (float, int)):
            f = np.array([f])
        if isinstance(f, (list)):
            f = np.array(f)
        if isinstance(f, (pd.Series, pd.DataFrame)):
            f = f.values

        if not torch.is_tensor(f):
            # Check that types are valid, then convert to Tensor
            if not all(np.issubdtype(f_i.dtype, np.number) for f_i in f.flatten()):
                raise TypeError('Each element of f being transformed must be numeric')
            f = torch.tensor(f)

        if not fit:
            if not hasattr(self, 'f_transform_func'):
                raise UnfitError('Transform function is being called without being fit first.')

        if f.ndim == 1:
            f = f.reshape(-1, 1)

        if inverse:
            f_obj = self.f_transform_func.inverse(f)
            return pd.Series(f_obj.flatten(), name=self.name) * self.multiplier
        else:
            if fit:
                self.f_transform_func = f_transform_dict[self.f_transform]()
                f_obj = self.f_transform_func(f, fit=True)
                self.f_raw = f  # Save raw data for re-loading and re-fitting state as needed
            else:
                f_obj = self.f_transform_func(f)
            return pd.Series(f_obj.flatten(), name=self.name+' Trans') * self.multiplier
        
    def save_state(self) -> dict:
        """
        Saves the state of the object as a dictionary.

        Returns:
            dict: A dictionary containing the state of the object.
        """
        # Prepare a dictionary to describe the state
        obj_dict = {'init_attrs': {}}

        # Select some optimizer attributes to save directly
        init_attrs = ['name', 'aim', 'f_transform']
        for attr in init_attrs:
            obj_dict['init_attrs'][attr] = getattr(self, attr)

        # If the transformer has been fit, store the raw data so it can be refit upon load
        if hasattr(self, 'f_transform_func'):
            obj_dict['f_raw'] = self.f_raw.tolist()

        return obj_dict

    @classmethod
    def load_state(cls, obj_dict: dict):
        """
        Loads the state of the target object from a dictionary.

        Args:
            cls (class): The class of the target object.
            obj_dict (dict): A dictionary containing the state of the target object.

        Returns:
            The loaded target object.
        """
        new_target = cls(**obj_dict['init_attrs'])

        # If the transformer has been fit before saving, refit it
        f = torch.Tensor(obj_dict['f_raw'])
        new_target.transform_f(f, fit=True)

        return new_target
