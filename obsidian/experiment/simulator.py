"""Simulate virtual experimental data"""

from obsidian.parameters import ParamSpace

from typing import Callable
import pandas as pd
import numpy as np
import warnings


class Simulator:
    """
    Simulator class for generating in-silico responses to requested experiments.

    This class provides functionality to simulate responses to a set of experiments based on a given response function.
    The simulated responses can be subject to error, which is controlled by the `eps` parameter.

    Attributes:
        X_space (ParamSpace): The ParamSpace object representing the allowable space for optimization.
        response_function (Callable): The callable function used to convert experiments to responses.
        name (str or list[str]): Name of the simulated output(s).
        eps (float or list[float]): The simulated error to apply, as the standard deviation of the Standard Normal distribution.
        kwargs (dict): Optional hyperparameters for the response function.

    Raises:
        TypeError: If response_function is not a callable function.
        TypeError: If X_space is not an obsidian ParamSpace object.
        
    """

    def __init__(self,
                 X_space: ParamSpace,
                 response_function: Callable,
                 name: str | list[str] = 'Response',
                 eps: float | list[float] = 0.0,
                 **kwargs):
        
        if not callable(response_function):
            raise TypeError('Response generator must be a callable function')
        if not isinstance(X_space, ParamSpace):
            raise TypeError('X_space must be an obsidian ParamSpace object')
        
        self.X_space = X_space
        self.response_function = response_function
        self.name = name
        self.eps = eps if isinstance(eps, list) else [eps]
        self.kwargs = kwargs
    
    def __repr__(self):
        return f" obsidian Simulator(response_function={self.response_function.__name__}, eps={self.eps})"

    def simulate(self,
                 X_prop: pd.DataFrame) -> np.ndarray:
        """
        Generates a response to a set of experiments.

        Currently, response function only handles strictly numeric values and categories are manually penalized.

        Args:
            X_prop (pd.DataFrame): Proposed experiments to evaluate.


        Returns:
            np.ndarray: Array of response values to experiments.
        """
        # De-map everything into 0,1 based on ranges
        X = self.X_space.unit_map(X_prop).values
        
        y_sim = self.response_function(X)
        
        # Apply error
        # Expand length of eps to match number of outputs
        if len(self.eps) == 1:
            self.eps *= y_sim.ndim
        if y_sim.ndim == 1:
            y_sim = y_sim.reshape(-1, 1)
        for i in range(y_sim.shape[1]):
            rel_error = np.random.normal(loc=1, scale=self.eps[i], size=y_sim.shape[0])
            y_sim[:, i] *= rel_error

        # Handle naming conventions
        y_dims = y_sim.shape[1]
        if isinstance(self.name, list):
            if len(self.name) != y_dims:
                warnings.warn("Number of names does not match the number of dimensions. Using default response names.")
                self.name = [f'{self.name} {i+1}' for i in range(y_dims)]
        else:
            if y_dims == 1:
                self.name = [self.name]
            else:
                self.name = [f'{self.name} {i+1}' for i in range(y_dims)]

        df_sim = pd.DataFrame(y_sim, columns=self.name)

        return df_sim
