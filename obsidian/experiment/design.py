"""Design initial experiments"""

from .utils import *

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

class AdvExpDesigner:
    """
    A class to manage experimental designs that integrate functions for sampling, evaluating, optimizing, and visualizing designs.
    """

    def __init__(self, continuous_params, conditional_subparameters, subparam_mapping=None):
        """
        Initializes the AdvExpDesigner with experimental parameters and optional subparameter mappings.

        :param continuous_params: A dictionary containing the continuous parameters for the design.
        :param conditional_subparameters: A dictionary containing the conditional subparameters for the design.
        :param subparam_mapping: A dictionary for mapping, will be inferred if not provided.
        """
        self.continuous_params = continuous_params
        self.conditional_subparameters = conditional_subparameters
        self.subparam_mapping = subparam_mapping or infer_subparam_mapping(self.conditional_subparameters)
        self.continuous_keys = list(self.continuous_params.keys())
        self.categorical_keys = list(self.conditional_subparameters.keys())
        self.subparam_key = list(self.subparam_mapping.values())[0] if self.subparam_mapping else None
    
    def generate_design(self, seed, n_samples, optimize_categories=True):
        """
        Generates a design by sampling from the given parameter space.

        :param seed: Random seed for reproducibility.
        :param n_samples: Number of samples to generate.
        :param optimize_categories: Whether to optimize categorical assignments to reduce correlation.
        :return: A DataFrame representing the generated sample design.
        """
        return sample_design(
            seed=seed,
            n_samples=n_samples,
            continuous_params=self.continuous_params,
            conditional_subparameters=self.conditional_subparameters,
            subparam_mapping=self.subparam_mapping,
            optimize_categories=optimize_categories
        )



    def evaluate_design(self, design, metrics_to_optimize=None):
        """
        Evaluates the quality of the given design based on specified metrics.
    
        :param design: The design to evaluate.
        :param metrics_to_optimize: List of metrics to evaluate the design.
        :return: A dictionary containing evaluated metrics for the design.
        """
        if metrics_to_optimize is None:
            metrics_to_optimize = [
                'D-optimality',
                'A-optimality',
                'Condition Number',
                'Pairwise Distance CV',
                'Max Continuous Corr',
                'Max Categorical Corr'
            ]
    
        return evaluate_design(
            design=design,
            continuous_keys=self.continuous_keys,
            categorical_keys=self.categorical_keys,
            subparam_mapping=self.subparam_mapping,
            metrics_to_optimize=metrics_to_optimize
        )


    def optimize_design(self, n_trials, n_samples, metrics_to_optimize=None,
                        maximize_metrics=None, seed_start=0, max_workers=None):
        """
        Optimizes the design based on specified metrics and constraints.
    
        :param n_trials: Number of optimization trials.
        :param n_samples: Number of sample points to consider.
        :param metrics_to_optimize: List of metrics to optimize.
        :param maximize_metrics: List of booleans indicating whether to maximize each metric.
        :param seed_start: Starting seed for random number generation.
        :param max_workers: Maximum number of parallel workers for optimization.
        :return: The best design and metrics summary.
        """
        if metrics_to_optimize is None:
            metrics_to_optimize = [
                'D-optimality',
                'A-optimality',
                'Condition Number',
                'Pairwise Distance CV',
                'Max Continuous Corr',
                'Max Categorical Corr'
            ]
        if maximize_metrics is None:
            maximize_metrics = 'D-optimality'
    
        return find_best_design_parallel(
            n=n_trials,
            n_samples=n_samples,
            continuous_params=self.continuous_params,
            conditional_subparameters=self.conditional_subparameters,
            subparam_mapping=self.subparam_mapping,
            metrics_to_optimize=metrics_to_optimize,
            maximize_metrics=maximize_metrics,
            seed_start=seed_start,
            max_workers=max_workers
        )



    def extend_design(self, existing_design, n, seed=None, num_candidates=10, metrics_to_optimize=None, maximize_metrics=None, max_workers=None):
        """
        Extends the existing design by adding more samples.
    
        :param existing_design: The existing design to extend.
        :param n: Number of new samples to add.
        :param seed: Optional random seed for reproducibility.
        :param num_candidates: Number of candidate extensions to evaluate.
        :param metrics_to_optimize: List of metrics to optimize.
        :param maximize_metrics: List of booleans indicating whether to maximize each metric.
        :param max_workers: Number of parallel workers.
        :return: The extended design and a summary of candidate metrics.
        """
        seed_start = seed if seed is not None else 1000
    
        return extend_design(
            existing_design=existing_design,
            n=n,
            continuous_params=self.continuous_params,
            conditional_subparameters=self.conditional_subparameters,
            subparam_mapping=self.subparam_mapping,
            metrics_to_optimize=metrics_to_optimize,
            maximize_metrics=maximize_metrics,
            num_candidates=num_candidates,
            seed_start=seed_start,
            max_workers=max_workers
        )


    def plot_quality_evolution(self, metrics_df):
        """
        Plots the evolution of the quality of the design over time or iterations.

        :param metrics_df: DataFrame containing the metrics data to plot.
        """
        plot_design_quality_evolution(metrics_df)

    def plot_histograms(self, design):
        """
        Plots histograms for the sampled design parameters.

        :param design: The design to visualize.
        """
        plot_design_histograms(
            design=design,
            continuous_keys=self.continuous_keys,
            categorical_keys=self.categorical_keys,
            subparam_mapping=self.subparam_mapping
        )


    def plot_correlation(self, design):
        """
        Plots the correlation matrix of the design's parameters.

        :param design: The design to visualize.
        """
        plot_correlation_matrix(design, self.categorical_keys)

    def plot_pca(self, design, hue=None):
        """
        Performs PCA (Principal Component Analysis) and plots the result.

        :param design: The design to analyze.
        :param hue: The categorical variable to color the data points by.
        """
        plot_pca(design, self.continuous_keys, self.subparam_mapping, hue)


    def plot_mds(self, design, hue=None):
        """
        Performs MDS (Multidimensional Scaling) and plots the result.

        :param design: The design to analyze.
        :param hue: The categorical variable to color the data points by.
        """
        plot_mds(design, self.continuous_keys, self.subparam_mapping, hue)


    def plot_umap(self, design, hue=None):
        """
        Performs UMAP (Uniform Manifold Approximation and Projection) and plots the result.

        :param design: The design to analyze.
        :param hue: The categorical variable to color the data points by.
        """
        plot_umap(design, self.continuous_keys, self.subparam_mapping, hue)


    def compare_frequencies(self, design):
        """
        Compares the empirical frequencies of categorical variables in the design
        with the expected frequencies defined in the conditional subparameters.

        :param design: The design DataFrame to analyze.
        """
        for cat_var in self.categorical_keys:
            level_info = self.conditional_subparameters[cat_var]
            levels = list(level_info.keys())
            expected_freq = np.array([level_info[lvl].get('freq', 1 / len(levels)) for lvl in levels])
            expected_freq /= expected_freq.sum()  # Normalize in case they don't sum to 1

            counts = design[cat_var].value_counts(normalize=True).reindex(levels).fillna(0).values

            print(f"\nCategorical variable: {cat_var}")
            print("Level\tExpected\tEmpirical")
            for lvl, exp_f, emp_f in zip(levels, expected_freq, counts):
                print(f"{lvl}\t{exp_f:.3f}\t\t{emp_f:.3f}")

