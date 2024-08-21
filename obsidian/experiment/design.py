"""Design initial experiments"""

from .utils import factorial_DOE

from obsidian.parameters import ParamSpace
from obsidian.exceptions import UnsupportedError

from botorch.utils.sampling import draw_sobol_samples
from numpy.typing import ArrayLike
from scipy.stats import qmc

import torch
from torch import Tensor
import pandas as pd
import warnings


class ExpDesigner:
    """
    ExpDesigner is a base class for designing experiments in a parameter space.

    Attributes:
        X_space (ParamSpace): The parameter space for the experiment.
        seed (int | None): The randomization seed.
    
    Raises:
        TypeError: If X_space is not an obsidian ParamSpace object.
    """

    def __init__(self,
                 X_space: ParamSpace,
                 seed: int | None = None):
        if not isinstance(X_space, ParamSpace):
            raise TypeError('X_space must be an obsidian ParamSpace object')
        
        self.X_space = X_space
        self.seed = seed

    def __repr__(self):
        """String representation of object"""
        return f"obsidian ExpDesigner(X_space={self.X_space})"

    def initialize(self,
                   m_initial: int | None = None,
                   method: str = 'LHS',
                   sample_custom: Tensor | ArrayLike | None = None) -> pd.DataFrame:
        """
        Initializes the experiment design.

        Args:
            m_initial (int): The number of experiments to initialize.
            method (str, optional): The method to use for initialization. Defaults to ``'LHS'``.
            seed (int | None, optional): The randomization seed. Defaults to ``None``.
            sample_custom (Tensor | ArrayLike | None, optional): Custom samples for initialization. Defaults to ``None``.

        Returns:
            pd.DataFrame: The initialized experiment design.

        Raises:
            KeyError: If method is not one of the supported methods.
            ValueError: If sample_custom is None when method is 'Custom'.
            ValueError: If the number of columns in sample_custom does not match the size of the feature space.
        """
        d = self.X_space.n_dim

        if m_initial is None:
            m_initial = int(d*2)
        m = m_initial
        seed = self.seed

        method_dict = {
            'LHS': lambda d, m: torch.tensor(
                qmc.LatinHypercube(d=d, scramble=False, seed=seed, strength=1, optimization='random-cd').random(n=m)),
            'Random': lambda d, m: torch.rand(size=(m, d)),
            'Sobol': lambda d, m: draw_sobol_samples(
                bounds=torch.tensor([0.0, 1.0]).reshape(2, 1).repeat(1, d), n=m, q=1).squeeze(1),
            'Custom': lambda d, m: torch.tensor(sample_custom),
            'DOE_full': lambda d, m: torch.tensor(factorial_DOE(d=d, n_CP=3, shuffle=True, seed=seed, full=True)),
            'DOE_res4': lambda d, m: torch.tensor(factorial_DOE(d=d, n_CP=3, shuffle=True, seed=seed))
        }
        
        if method not in method_dict.keys():
            raise KeyError(f'Method must be one of {method_dict.keys()}')
        if method == 'Custom':
            if sample_custom is None:
                raise ValueError('Must provide samples for custom')
        if method in ['DOE_full', 'DOE_res4']:
            if self.X_space.X_discrete:
                raise UnsupportedError('DOE methods not currently designed for discrete parameters')

        if seed is not None:
            torch.manual_seed(seed)
            if not torch.cuda.is_available():
                torch.use_deterministic_algorithms(True)
            
        if sample_custom is not None:
            if sample_custom.shape[1] != d:
                raise ValueError('Columns in custom sample do not match size of feature space')

        # Generate [0-1) samples for each parameter
        sample = method_dict[method](d, m)

        m_required = sample.shape[0]
        
        if m_required > m:
            warnings.warn(f'The number of experiments required to initialize the requested design \
                          ({m_required}) exceeds the m_initial specified ({m}). \
                            Proceeding with larger number of experiments.')
        elif m_required < m:
            print(f'The number of initialization experiments ({m}) exceeds the required \
                   number for the requested design ({m_required}). Filling with randomized experiments.')
            excess = m - m_required
            sample_add = torch.rand(size=(excess, d))
            sample = torch.vstack((sample, sample_add))

        sample = pd.DataFrame(sample.numpy(), columns=self.X_space.X_names)
        
        # Reset parameters to 0 which are not allowed to vary in X_space
        for param in self.X_space.X_static:
            sample[param] = 0
            
        # Map samples into parameter space
        X_0 = self.X_space.unit_demap(sample)
                
        return X_0
