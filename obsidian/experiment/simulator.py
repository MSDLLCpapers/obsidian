"""Simulate virtual experimental data"""

from types import ModuleType
from obsidian.parameters import ParamSpace
from obsidian.rng import RNGManager

from typing import Callable
import pandas as pd
import numpy as np
import warnings

from numpy.random import Generator
import obsidian

class Simulator:
    """
    Simulator class for generating in-silico responses to requested experiments.

    This class provides functionality to simulate responses to a set of experiments based on a given response function.
    The simulated responses can be subject to error, which is controlled by the `eps` parameter.

    Attributes:
        X_space (ParamSpace): The ParamSpace object representing the allowable space for optimization.
        response_function (Callable): The callable function used to convert experiments to responses.
        name (str or list[str]): Name of the simulated output(s). Default is ``Response``.
        eps (float or list[float]): The simulated error to apply, as the standard deviation of the Standard
            Normal distribution. Default is ``0``.
        apply_noise (Callable | None): Optional custom function to apply noise to the simulated response. 
            If None, uses default multiplicative Gaussian noise. Must have signature::
        
                apply_noise(X, y_sim, eps, rng) -> y_with_noise
                
            Where X is the input array, y_sim is the noiseless simulation output with shape 
            (n_samples, n_outputs), eps is the noise parameter array with shape (1, n_eps), 
            and rng is the numpy random Generator. Must return array with same shape as y_sim.
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
                 rng: Generator | RNGManager | int | None = None,
                 apply_noise: Callable | None = None,
                 **kwargs):
        
        if not callable(response_function):
            raise TypeError('Response generator must be a callable function')
        if not isinstance(X_space, ParamSpace):
            raise TypeError('X_space must be an obsidian ParamSpace object')
        
        self.X_space = X_space
        self.response_function = response_function
        self.name = name
        # We always expect `y_sim` to be no more than 2D
        # `eps` with more than 1D makes no sense
        # i.e., one error per output dimension or one error for all dimensions, nothing more
        # Always converting eps to a 2D array for simplicity
        # numpy broadcasting will handle the rest
        self.eps = np.atleast_1d(eps)
        if self.eps.ndim > 1:
            raise ValueError(f"eps must be scalar or 1D list/array, got {self.eps.ndim}D array")
        self.eps = self.eps.reshape(1, -1)
        if isinstance(rng, RNGManager):
            self.rng: Generator | ModuleType = rng.np_rng
            self._seed = rng.seed
        elif isinstance(rng, Generator):
            # use provided numpy generator
            self.rng = rng
            self._seed = None
        else:
            self._seed = rng
            if obsidian.USE_OLD_RNG_CONTROL:
                # no random state control here
                self.rng = np.random
            else:
                # use new RNG manager to control random state
                rng = obsidian.create_rng_manager(rng)
                self.rng = rng.np_rng
        if not apply_noise:
            self.apply_noise = self._default_gaussian_noise
        else:
            if not callable(apply_noise):
                raise TypeError('Error function must be a callable function')
            self.apply_noise = apply_noise
        self.kwargs = kwargs

    def __repr__(self):
        """String representation of object"""
        return f" obsidian Simulator(response_function={self.response_function.__name__}, eps={self.eps})"

    def simulate(self,
                 X_prop: pd.DataFrame) -> pd.DataFrame:
        """
        Generates a response to a set of experiments.

        Currently, response function only handles strictly numeric values and categories are manually penalized.

        Args:
            X_prop (pd.DataFrame): Proposed experiments to evaluate.


        Returns:
            pd.DataFrame: DataFrame of response values to experiments.
        """
        # De-map everything into 0,1 based on ranges
        X = self.X_space.unit_map(X_prop).values
        
        y_sim = self.response_function(X)
        
        # Expand length of eps to match number of outputs
        # if len(self.eps) == 1:
        #     self.eps *= y_sim.ndim
        if y_sim.ndim == 1:
            y_sim = y_sim.reshape(-1, 1)

        # Apply noise to the simulated response
        y_sim = self.apply_noise(X, y_sim, self.eps, self.rng) # type: ignore

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

    #TODO: simulator can have its own random state now, so a save and load state method should be implemented in the future.

    @staticmethod 
    def _default_gaussian_noise(X: np.ndarray, y_sim: np.ndarray, eps: np.ndarray, rng: Generator) -> np.ndarray:
        """Default error function - multiplicative Gaussian noise."""
        rel_noise = 1 + eps * rng.normal(size=y_sim.shape)
        return y_sim * rel_noise