"""Parameters that can be sampled continuously between a minimum and maximum"""

from .base import Parameter
from .utils import transform_with_type

import warnings
import numpy as np


class Param_Continuous(Parameter):
    """
    Represents a continuous parameter.

    Attributes:
        name (str): The name of the parameter.
        min (int or float): The minimum value of the parameter.
        max (int or float): The maximum value of the parameter.

    Properties:
        range (int): The range of the parameter (max - min).

        """

    @transform_with_type
    def unit_map(self, X: np.ndarray):
        """Map from measured to 0,1 space"""
        return (X-self.min)/self.range if self.range != 0 else 0*X
        
    @transform_with_type
    def unit_demap(self, X: np.ndarray):
        """Map from 0,1 to measured space"""
        return X*self.range+self.min

    encode = unit_map
    
    decode = unit_demap

    @property
    def range(self):
        """The range of the parameter (max - min)"""
        return self.max-self.min
    
    def set_search(self,
                   search_min: int | float,
                   search_max: int | float):
        """
        Set the search space for the parameter

        Args:
            search_min (int or float): The minimum value of the search space.
            search_max (int or float): The maximum value of the search space.
        """
        for val in [search_min, search_max]:
            self._validate_value(val)
            
        self.search_min = search_min
        self.search_max = search_max
    
    def open_search(self):
        """Set the search space to the parameter space"""
        self.set_search(self.min, self.max)
    
    def __repr__(self):
        """String representation of object"""
        return f"{self.__class__.__name__}(name={self.name}, min={self.min}, max={self.max})"

    def _validate_value(self, value: int | float):
        """
        Validates if the given value is within the specified range.

        Args:
            value (int | float): The value to be validated.

        Raises:
            TypeError: If the value is not a number.
            ValueError: If the value is outside of the specified range.
        """
        if not isinstance(value, (int, float)):
            raise TypeError(f'Value {value} is not a number')
        if (value < self.min) or (value > self.max):
            raise ValueError(f'Value {value} is outside of range {self.min} to {self.max}')

    def __init__(self,
                 name: str,
                 min: int | float,
                 max: int | float,
                 search_min: int | float = None,
                 search_max: int | float = None):
        super().__init__(name=name)
        if max < min:
            warnings.warn(f'Minimum value {min} is greater than maximum value {max}. Auto-swapping values.', UserWarning)
            min, max = max, min
        self.min = min
        self.max = max
        for val in [min, max]:
            self._validate_value(val)
        
        # Set the search space to the parameter space by default
        if not search_min:
            search_min = self.min
        if not search_max:
            search_max = self.max
        self.set_search(search_min, search_max)


class Param_Observational(Param_Continuous):
    """
    This is an observational numeric variable that is used for fitting but is not leveraged during optimization

    Attributes:
        name (str): The name of the parameter.
        min (int or float): The minimum value of the parameter.
        max (int or float): The maximum value of the parameter.
        design_point (int or float): The point which the observational value will be locked to during experiment optimization

    """
    def __init__(self,
                 name: str,
                 min: int | float,
                 max: int | float,
                 search_min: int | float = None,
                 search_max: int | float = None,
                 design_point: int | float | None = None):
        super().__init__(name=name, min=min, max=max, search_min=search_min, search_max=search_max)
        self.design_point = design_point = design_point if design_point is not None else max
