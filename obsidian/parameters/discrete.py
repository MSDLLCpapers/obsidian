"""Parameters that can only be sampled at specific values"""

from .base import Parameter
from .utils import transform_with_type

from obsidian.config import CAT_SEP

from numpy.typing import ArrayLike

import pandas as pd
import numpy as np


class Param_Discrete(Parameter):
    """
    Represents a discrete parameter.

    Attributes:
        name (str): The name of the parameter.
        categories (list[str]): The categories of the parameter.

    Properties:
        min (int): The minimum value of the parameter (always 0).
        nc (int): The number of categories.
        max (int): The maximum value of the parameter (nc - 1).

    """

    @transform_with_type
    def unit_map(self, X: np.ndarray):
        """
        Maps the input values to the unit interval [0, 1] based on the categories.

        Parameters:
            X (np.ndarray): The input values to be mapped.

        Returns:
            np.ndarray: Values mapped to the unit interval [0, 1].
        """
        X_str = X.flatten().astype('U11')
        indices = [self.categories.index(x) for x in X_str]
        return (np.array(indices)/self.nc).reshape(X.shape)
    
    @transform_with_type
    def unit_demap(self, X: np.ndarray):
        """
        Maps continuous values to discrete categories.

        Parameters:
            X (np.ndarray): The input values in the interval [0,1]

        Returns:
            np.ndarray: Values mapped back to the real space.
        """

        return np.array(self.categories)[(X.flatten()*self.nc)
                                         .astype('int')].reshape(X.shape).astype('U11')

    @property
    def min(self):
        return 0

    @property
    def nc(self):
        return len(self.categories)
        
    @property
    def max(self):
        return self.nc-1
        
    def _validate_value(self,
                        value: str):
        """
        Validates if the given value is in the list of categories.

        Args:
            value (str): The value to be validated.

        Raises:
            KeyError: If the value is not in the list of categories.
            TypeError: If the value is not a string
        """
        if not isinstance(value, str):
            raise TypeError(f'Value {value} is not a string')
        
        if value not in self.categories:
            raise ValueError(f'Value {value} is not in {self.categories}')

    def __init__(self,
                 name: str,
                 categories: str | list[str]):
        super().__init__(name=name)
        if isinstance(categories, str):
            self.categories = categories.split(',')
        else:
            self.categories = categories
        for c in categories:
            self._validate_value(c)
    
    def __repr__(self):
        return f"{self.__class__.__name__}(name={self.name}, categories={self.categories})"


class Param_Ordinal(Param_Discrete):
    """
    Represents an ordinal parameter; a discrete parameter with an order.
    """
    # Ordinal encoder maps to integers

    def encode(self, X: np.ndarray):
        return self.unit_map(X)
    
    def decode(self, X: np.ndarray):
        return self.unit_demap(X)


class Param_Categorical(Param_Discrete):
    """
    Represents an categorical parameter; a discrete parameter without an order.
    """

    # Categorical encoder maps to one-hot columns
    # No decorator on this, we need to handle dataframes
    def encode(self, X: str | ArrayLike):
        
        X_str = np.array(X).flatten().astype('U11')
        X_cat = pd.Series(X_str).astype(pd.CategoricalDtype(categories=self.categories))
        X_ohe = pd.get_dummies(X_cat, prefix_sep=CAT_SEP, dtype=float, prefix=self.name)

        return X_ohe
    
    def decode(self, X: ArrayLike):
        
        return [self.categories[int(x)] for x in np.array(X).argmax(axis=1)]


class Task(Param_Discrete):
    """
    Represents an task parameter; a discrete parameter indicating a distinct system.
    """
    # Similar to ordinal, but (0,nc-1) instead of (0,1)

    @transform_with_type
    def encode(self, X: np.ndarray):
        return self.unit_map(X)*self.nc
    
    @transform_with_type
    def decode(self, X: np.ndarray):
        return np.array(self.unit_demap(X/self.nc)).astype('U11')


class Param_Discrete_Numeric(Param_Discrete):
    """
    Represents an discrete numeric parameter; an ordinal parameter comprised of numbers.

    Raises:
        TypeError: If the categories are not numbers.
        TypeError: If the categories are not a list of numbers.
    """

    @property
    def min(self):
        return np.array(self.categories).min()

    @property
    def max(self):
        return np.array(self.categories).max()
    
    @property
    def range(self):
        return self.max-self.min

    def __init__(self,
                 name,
                 categories: int | float | list[int | float]):
        if not isinstance(categories, (int, float, list)):
            raise TypeError('Categories must be a number or list of numbers')
        if isinstance(categories, list):
            if not all(isinstance(x, (int, float)) for x in categories):
                raise TypeError('Categories must be a list of numbers')
        
        self.name = name
        self.categories = categories if isinstance(categories, list) else [categories]
        self.categories.sort()
        for c in categories:
            self._validate_value(c)

    def _validate_value(self,
                        value: int | float):
        """
        Validates if the given value is in the list of categories.

        Args:
            value (str): The value to be validated.

        Raises:
            KeyError: If the value is not in the list of categories.
            TypeError: If the value is not a string
        """
        if not isinstance(value, (int, float)):
            raise TypeError(f'Value {value} is not numeric')
        
        if value not in self.categories:
            raise ValueError(f'Value {value} is not in {self.categories}')

    @transform_with_type
    def unit_map(self, X: np.ndarray):
        return (X-self.min)/self.range if self.range != 0 else 0*X

    @transform_with_type
    def unit_demap(self, X: np.ndarray):
        closest_idx = np.abs(np.array(self.categories)
                             - (X.flatten()[..., np.newaxis]*self.range+self.min)).argmin(axis=1)
        return np.array(self.categories)[closest_idx].reshape(X.shape)

    encode = unit_map
    decode = unit_demap
