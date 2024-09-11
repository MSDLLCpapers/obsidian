"""Bayesian Optimization: Select experiments from the predicted posterior and update the prior"""

from .base import Optimizer

from obsidian.parameters import ParamSpace, Target, Task
from obsidian.surrogates import SurrogateBoTorch, DNN
from obsidian.acquisition import aq_class_dict, aq_defaults, aq_hp_defaults, valid_aqs
from obsidian.surrogates import model_class_dict
from obsidian.objectives import Index_Objective, Objective_Sequence
from obsidian.constraints import Linear_Constraint, Nonlinear_Constraint, Output_Constraint
from obsidian.exceptions import IncompatibleObjectiveError, UnsupportedError, UnfitError, DataWarning
from obsidian.config import TORCH_DTYPE

from botorch.acquisition.objective import MCAcquisitionObjective
from botorch.optim import optimize_acqf, optimize_acqf_mixed
from botorch.sampling import SobolQMCNormalSampler
from botorch.sampling.list_sampler import ListSampler
from botorch.sampling.index_sampler import IndexSampler
from botorch.models.model_list_gp_regression import ModelListGP
from botorch.models.gpytorch import GPyTorchModel
from botorch.models.model import ModelList
from botorch.utils.sampling import draw_sobol_samples
from botorch.utils.multi_objective.box_decompositions.non_dominated import NondominatedPartitioning

import torch
from torch import Tensor
import pandas as pd
import numpy as np
import warnings


class BayesianOptimizer(Optimizer):
    """
    BayesianOptimizer is a class that implements a Bayesian optimization algorithm.

    This class is used to optimize a given function by iteratively selecting the next set of input parameters
    based on the results of previous evaluations. It uses a surrogate model to approximate the underlying function
    and an acquisition function to determine the next set of parameters to evaluate.

    Args:
        X_space (ParamSpace): The parameter space defining the search space for the optimization.
        surrogate (str | dict | list[str] | list[dict], optional): The surrogate model(s) to use.
            It can be a string representing a single model type, a dictionary specifying multiple model types
            with their hyperparameters, or a list of strings or dictionaries.
            
            Defaults to ``'GP'``. Options are as follows:
            
            - ``'GP'``: Gaussian Process with default settings (Matern Kernel, Gamma covariance priors)
            - ``'MixedGP'``: GP with mixed parameter types (continuous, categorical). Will be re-selected
              by default if 'GP' is selected and input space is mixed.
            - ``'DKL'``: GP with a NN feature-extractor (deep kernel learning)
            - ``'GPflat'``: GP without priors. May result in optimization instability, but removes bias
              for special situations.
            - ``'GPprior'``: GP with custom priors on the mean, likelihood, and covariance
            - ``'MTGP'``: Multi-task GP for multi-output optimization. Will be re-selected by default
              if 'GP' is selected and the input space contains Task parameters.
            - ``'DNN'``: Dropout neural network. Uses MC sampling to mask neurons during training and
              to estimate uncertainty.
            
            
        seed (int | None, optional): The random seed to use. Defaults to ``None``.
        verbose (int, optional): The verbosity level. Defaults to ``1``.

    Attributes:
        surrogate_type (list[str]): The shorthand name of each surrogate model.
        surrogate_hps (list[dict]): The hyperparameters for each surrogate model.
        is_fit (bool): Indicates whether the surrogate model has been fit to data.

    Raises:
        TypeError: If the surrogate argument is not a string, dict, or list of str/dict.
        ValueError: If the surrogate dictionary contains more than one surrogate model type.
        KeyError: If the surrogate model is not selected from the available models.
        ValueError: If the number of responses does not match the number of specified surrogate

    """
    def __init__(self,
                 X_space: ParamSpace,
                 surrogate: str | dict | list[str] | list[dict] = 'GP',
                 seed: int | None = None,
                 verbose: int = 1):
       
        super().__init__(X_space=X_space, seed=seed, verbose=verbose)

        self.surrogate_type = []  # Shorthand name as str (as provided)
        self.surrogate_hps = []  # Hyperparameters

        # Surrogate model selection
        if not isinstance(surrogate, (str, list, dict)):
            raise TypeError('Surrogate argument must be a string, list of strings \
                            dict of {surrogate: {hypers}} or list of dicts')
        
        def _load_surrogate_str(surrogate_str):
            # Just a string = use the same model type for each ycol that might appear later
            self.surrogate_type.append(surrogate_str)
            self.surrogate_hps.append(dict())

        def _load_surrogate_dict(surrogate_dict):
            if len(surrogate_dict) != 1:
                raise ValueError('Surrogate dictionary must contain only one surrogate model type')
            # Dictionary of dictionaries = hyperparameters may be included
            for surrogate_str, surrogate_hps in surrogate_dict.items():
                if not isinstance(surrogate_hps, dict):
                    raise TypeError('Surrogate dictionary must be a nested dictionary providing hyperparameters')
                self.surrogate_type.append(surrogate_str)
                self.surrogate_hps.append(surrogate_hps)

        if isinstance(surrogate, str):
            _load_surrogate_str(surrogate)
        elif isinstance(surrogate, dict):
            _load_surrogate_dict(surrogate)
        elif isinstance(surrogate, list):
            for surrogate_i in surrogate:
                if isinstance(surrogate_i, str):
                    _load_surrogate_str(surrogate_i)
                elif isinstance(surrogate_i, dict):
                    _load_surrogate_dict(surrogate_i)
                else:
                    raise ValueError('Surrogate argument must be a string, dict, or list of str/dict')
                
        for surrogate_str in self.surrogate_type:
            if surrogate_str not in model_class_dict.keys():
                raise KeyError(f'Surrogate model must be selected from one of: {model_class_dict.keys()}')

        return
    
    @property
    def is_fit(self):
        """
        Check if all surrogate mdoels in optimizer are fit

        Returns:
            bool: True if the optimizer is fit, False otherwise.
        """
        if hasattr(self, 'surrogate'):
            return all(model.is_fit for model in self.surrogate)
        else:
            return False

    def _validate_target(self,
                         target: Target | list[Target] | None = None):
        """
        Validates the target input for the optimization process.

        Args:
            target (Target | list[Target] | None, optional): The target object or a list of target objects to be validated.
                If None, the target object specified during the initialization of the optimizer will be used.
                Defaults to ``None``.

        Raises:
            TypeError: If the target is not a Target object or a list of Target objects.

        Returns:
            list[Target]: List of validated target(s)
        """
        if target is None:
            if not hasattr(self, 'target'):
                raise TypeError('Target must be a Target object or a list of Target objects')
            else:
                target = list(self.target)
        else:
            if not isinstance(target, (Target, list)):
                raise TypeError('Target must be a Target object or a list of Target objects')
            elif isinstance(target, Target):
                target = [target]
            elif isinstance(target, list):
                if not all(isinstance(t, Target) for t in target):
                    raise TypeError('Each item in target must be a Target object')
        return target

    def fit(self,
            Z: pd.DataFrame,
            target: Target | list[Target]):
        """
        Fits the BO surrogate model to data.

        Args:
            Z (pd.DataFrame): Total dataset including inputs (X) and response values (y)
            target (Target or list of Target): The responses (y) to be used for optimization,
                packed into a Target object or list thereof

        Returns:
            None. Updates the model in self.surrogate

        Raises:
            NameError: If the target is not present in the data.
            ValueError: If the number of responses does not match the number of specified surrogate models.
        """
    
        self.target = tuple(self._validate_target(target))
        self.y_names = tuple([t.name for t in self.target])
        self.n_response = len(self.target)
        
        for t in self.target:
            if t.name not in Z.columns:
                raise NameError(f"Specified target {t.name} is not present in data")

        # For multi-response, specifying one model type is OK; as this will be used for all responses
        if self.n_response > 1:
            if len(self.surrogate_type) == 1:
                self.surrogate_type *= self.n_response
                self.surrogate_hps *= self.n_response
            else:
                if self.n_response != len(self.surrogate_type):
                    raise ValueError('Number of responses does not match the number \
                                     of specified surrogate models')

        # Filter out NaN by X
        X_names = list(self.X_space.X_names)
        Z_valid = Z.copy().dropna(subset=X_names)

        # Unpack X data
        self.X_train = Z_valid[X_names]
        self.X_t_train = self.X_space.encode(self.X_train)
        # Note: Z is allowed to contain columns that are neither ycols or Xcols; these will get ignored
        # Accessing the list(tuple(names)) will enforce that the order of the columns is preserved before fitting

        # Converty y (response) data to f (target) data
        self.y_train = pd.concat([Z_valid[t.name] for t in self.target], axis=1)
        self.f_train = pd.concat([t.transform_f(Z_valid[t.name], fit=True) for t in self.target], axis=1)

        # Extract the X which achieves the best sum_f
        self.X_best_f_idx = self.f_train.sum(axis=1).idxmax()
        self.X_best_f = self.X_train.iloc[self.X_best_f_idx, :].to_frame().T

        # Instantiate and fit the model(s)
        self.surrogate = []
        for i in range(self.n_response):
            self.surrogate.append(
                SurrogateBoTorch(model_type=self.surrogate_type[i], seed=self.seed,
                                 verbose=self.verbose >= 2, hps=self.surrogate_hps[i]))
            
            # Handle response NaN values on a response-by-response basis
            f_train_i = self.f_train.iloc[:, i]
            nan_indices = np.where(f_train_i.isna().values)[0]
            X_t_train_valid = self.X_t_train.drop(nan_indices)
            f_train_i_valid = f_train_i.drop(nan_indices)
            if X_t_train_valid.shape[0] < 1:
                raise ValueError(f'No valid data points for response {self.y_names[i]}')
            if f_train_i_valid.shape[0] < 1:
                raise ValueError(f'No valid response data points for response {self.y_names[i]}')
            
            # Fit the model for each response
            self.surrogate[i].fit(X_t_train_valid, f_train_i_valid,
                                  cat_dims=self.X_space.X_t_cat_idx, task_feature=self.X_space.X_t_task_idx)
            
            if self.verbose >= 1:
                print(f'{self.surrogate_type[i]} model has been fit to data'
                      + f' with an R2-train-score of: {self.surrogate[i].r2_score:.3g}'
                      + (f' and a training-loss of: {self.surrogate[i].loss:.3g}' if self.verbose >= 2 else '')
                      + ' for response: {self.y_names[i]}')
        return
    
    def save_state(self) -> dict:
        """
        Saves the parameters of the Bayesian Optimizer so that they can be reloaded without fitting.

        Returns:
            dict: A dictionary containing the fit parameters for later loading.
        
        Raises:
            UnfitError: If the surrogate model has not been fit before saving the optimizer.
        """

        if not self.is_fit:
            raise UnfitError('Surrogate model must be fit before saving optimizer')

        # Prepare a dictionary to describe the state
        config_save = {'opt_attrs': {},
                       'X_space': self.X_space.save_state(),
                       'surrogate_spec': [{func: hps} for func, hps in zip(self.surrogate_type, self.surrogate_hps)],
                       'target': [t.save_state() for t in self.target]}

        # Select some optimizer attributes to save directly
        opt_attrs = ['X_train', 'y_train',
                     'y_names', 'n_response',
                     'seed', 'X_best_f_idx', 'X_best_f']
        
        for attr in opt_attrs:
            if isinstance(getattr(self, attr), (pd.Series, pd.DataFrame)):
                config_save['opt_attrs'][attr] = getattr(self, attr).to_dict()
            else:
                config_save['opt_attrs'][attr] = getattr(self, attr)

        # Unpack the fit parameters of each surrogate model, if present
        if self.surrogate:
            model_states = []
            # Save each surrogate model using surrogate.save() methods
            for surrogate in self.surrogate:
                model_states.append(surrogate.save_state())

            config_save['model_states'] = model_states

        return config_save
    
    def __repr__(self):
        return f'BayesianOptimizer(X_space={self.X_space}, surrogate={self.surrogate_type}, target={getattr(self, "target", None)})'

    @classmethod
    def load_state(cls,
                   config_save: dict):
        """
        Loads the parameters of the Bayesian Optimizer from a previously fit optimizer.

        Args:
            config_save (dict): A dictionary containing the fit parameters for later loading.

        Returns:
            None. Updates the parameters of the BayesianOptimizer and its surrogate model.

        Raises:
            ValueError: If the number of saved models does not match the number of named models.
        """

        new_opt = cls(X_space=ParamSpace.load_state(config_save['X_space']),
                      surrogate=config_save['surrogate_spec'])
        new_opt.target = [Target.load_state(t) for t in config_save['target']]

        # Directly unpack all of the entries in opt_attrs
        for k, v in config_save['opt_attrs'].items():
            setattr(new_opt, k, v)
  
        # Unpack and encode/transform the data objects if present
        data_objects = ['X_train', 'y_train', 'X_best_f']
        if all(hasattr(new_opt, attr) for attr in data_objects):
            new_opt.X_train = pd.DataFrame(new_opt.X_train)
            new_opt.X_t_train = new_opt.X_space.encode(new_opt.X_train)
            new_opt.y_train = pd.DataFrame(new_opt.y_train, columns=new_opt.y_names)
            new_opt.X_best_f = pd.DataFrame(new_opt.X_best_f)
            
            f_train = pd.DataFrame()
            for t, y in zip(new_opt.target, new_opt.y_train.columns):
                f = t.transform_f(new_opt.y_train[y], fit=True)
                f_train = pd.concat([f_train, f.to_frame()], axis=1)
            new_opt.f_train = f_train
            
        # Unpack the models and parameteres if present
        if 'model_states' in config_save:
            if len(new_opt.surrogate_type) != len(config_save['model_states']):
                raise ValueError('The number of saved models does not match the number of named models')
            
            # Reload each surrogate model using surrogate.load() methods
            new_opt.surrogate = []
            for obj_dict in config_save['model_states']:
                new_opt.surrogate.append(SurrogateBoTorch.load_state(obj_dict))
        
        return new_opt
    
    def predict(self,
                X: pd.DataFrame,
                return_f_inv: bool = True,
                PI_range: float = 0.7) -> pd.DataFrame:
        """
        Predicts a response over a range of experiments using the surrogate function.

        Args:
            X (pd.DataFrame): Experiments to predict over.
            return_f_inv (bool, optional): Whether or not to return the inverse-transformed objective function,
                which is the raw response (unscored). The default is ``True``. Most internal calls set to ``False`` to handle
                the transformed objective function.
            PI_range (float, optional): The nominal coverage range for the returned prediction interval

        Returns:
            pd.DataFrame: Mean prediction and prediction interval for each response

        Raises:
            TypeError: If the input is not a DataFrame.
            UnfitError: If the surrogate model has not been fit before predicting.
            ValueError: If the prediction interval range is greater than 1.
            NameError: If the input does not contain all of the required predictors from the training set.
        """

        if not isinstance(X, pd.DataFrame):
            raise TypeError('X must be pd.DataFrame')
        if not self.is_fit:
            raise UnfitError('Surrogate model must be fit before predicting')
        if PI_range >= 1:
            raise ValueError('Prediction interval range must be < 1 \
                              (100% coverage of prob. density func.)')
        if not all(col in X.columns for col in self.X_train.columns):
            raise NameError('X for prediction does not contain all of the \
                            required predictors from the training set')
        
        if self.verbose >= 3:
            print(f'Predicting {X.shape[0]} experiments [...]')
        
        X_names = list(self.X_space.X_names)
        X_pred = X[X_names].dropna(subset=X_names)  # Reinforce order and non-nan before proceeding
        nan_indices = np.where(pd.isnull(X[X_names]).any(axis=1))[0].tolist()
        if nan_indices:
            warnings.warn(f'NaN values in X_pred filtered out at indices: {nan_indices}', DataWarning)

        # Scale and encode X
        X_t = self.X_space.encode(X_pred)

        preds = pd.DataFrame()
        for i in range(self.n_response):
            mu, sd = self.surrogate[i].predict(X_t)  # Returns pd.DataFrame/Series objects
            target_i = self.target[i]
            _, lb = self.surrogate[i].predict(X_t, q=(1-PI_range)/2)
            _, ub = self.surrogate[i].predict(X_t, q=1-(1-PI_range)/2)
            name = self.y_names[i]
            if return_f_inv:
                mu = target_i.transform_f(mu, inverse=True).rename(name+' (pred)')
                lb = target_i.transform_f(lb, inverse=True).rename(name+' lb')
                ub = target_i.transform_f(ub, inverse=True).rename(name+' ub')
            else:
                mu = pd.Series(mu, name=name+'_t (pred)')
                lb = pd.Series(lb, name=name+'_t lb')
                ub = pd.Series(ub, name=name+'_t ub')
            predict_i = pd.concat([mu, lb, ub], axis=1)
            preds = pd.concat([preds, predict_i], axis=1)
            
        return preds
       
    def _validate_hypers(self,
                         o_dim: int,
                         acquisition: str | dict,) -> tuple[dict, dict]:
        """
        Validates the acquisition functions and their hyperparameters.

        Args:
            o_dim (int): The output dimensionality of the final objective function
            acquisition (str | dict): Acquisition function name (str) or dictionary
                containing the acquisition function name and its hyperparameters.

        Returns:
            tuple[dict, dict]. Validated acquisition functions and hyperparmeters

        Raises:
            ValueError: If the number of hyperparameters does not match the number of acquisition functions.
            TypeError: If the hyperparameters are not provided as a dictionary.
            ValueError: If an unknown hyperparameter is passed for an acquisition function
            UnsupportedError: If the acquisition function is not supported for the optimization type.
            ValueError: If any hyperparameters are required but not provided.

        """
        # If the item is a string, use default hypers starting with an empty dict
        aq_hps = {}
        if isinstance(acquisition, str):
            aq_str = acquisition
            hps = {}
        # If the item is a dict, the structure is {name: {hypers}}
        else:
            # Validate the hypers if provided
            if len(acquisition.items()) != 1:
                raise ValueError('One dictionary of hyperparameters \
                                    must be provided for each acquisition function')
            aq_str, hps = list(acquisition.items())[0]
            if not isinstance(hps, dict):
                raise TypeError('Hyperparameters must be provided as a dictionary')
            if not all(key in aq_hp_defaults[aq_str].keys() for key in hps.keys()):
                raise ValueError('Unknown hyperpameter amongst {hps.keys()} \
                                    selected for {aq_str}, select from one of \
                                    {aq_hp_defaults[aq_str].keys()}')
            aq_hps.update(hps)

        optim_type = 'single' if o_dim == 1 else 'multi'

        # Validate the acquisition name
        if aq_str not in valid_aqs[optim_type]:
            raise UnsupportedError(f'Each acquisition function must be \
                                    selected from: {valid_aqs[optim_type]}')

        # Fill in empty hypers with defaults, as appropriate
        for key, defaults in aq_hp_defaults[aq_str].items():
            if hps.get(key) is None:
                if not defaults['optional']:
                    raise ValueError(f'Must specify hyperpameter value {key} for {aq_str}')
                if key in ['weights', 'scalarization_weights']:
                    aq_hps[key] = [1] * o_dim
                else:
                    aq_hps[key] = defaults['val']

        return (aq_str, aq_hps)

    def _parse_aq_kwargs(self,
                         aq: str,
                         hps: dict,
                         m_batch: int,
                         target_locs: list[int],
                         X_t_pending: Tensor | None = None,
                         objective: MCAcquisitionObjective | None = None) -> dict:
        """
        Parses the acquisition function keyword arguments based on the selected acquisition
        function and hyperparameters.

        Args:
            aq (str): The name of the acquisition function.
            hps (dict): The hyperparameters for the acquisition function.
            target_locs (list[int]): Indices of trained targets to use for objective function
            X_t_pending (Tensor, optional): Suggested points yet to be run
            objective (GenericMCMultiOutputObjective or GenericMCObjective, optional):
                The objective used foroptimization after calling the target models.
                Default is ``None``.

        Returns:
            dict: The parsed acquisition function keyword arguments.
        """
        
        aq_kwargs = {}

        # Establish baseline X from training and pending
        X_train = torch.tensor(self.X_space.encode(self.X_train).values, dtype=TORCH_DTYPE)
        if X_t_pending is not None:
            X_baseline = torch.concat([X_train, X_t_pending], axis=0)
        else:
            X_baseline = X_train

        # Calculate the performance on baseline X
        f_all = []
        for i in target_locs:
            X_b = pd.DataFrame(X_baseline.numpy(),
                               columns=[col for col in self.X_t_train.columns
                                        if col not in self.X_space.X_task])
            f_i, _ = self.surrogate[i].predict(X_b)
            f_all.append(f_i)
        f_t = torch.stack(f_all, axis=1)

        # If using an objective, want to calculate EI/PI from here
        o = f_t if not objective else objective(f_t.unsqueeze(0), X_baseline).squeeze(0)
        if objective:
            aq_kwargs['objective'] = objective
        
        # Improvement aqs based on inflation or deflation of best point
        if aq in ['EI', 'PI']:
            o_max = o.max(dim=0).values * (1+hps['inflate'])
            aq_kwargs.update({'best_f': o_max})
        
        # UCB based on: mu + sqrt(beta) * sqrt(variance) = mu + sqrt(beta) * sd
        if aq == 'UCB':
            aq_kwargs['beta'] = hps['beta']

        # Noisy aqs require X_train reference
        if aq in ['NEI', 'NEHVI', 'NParEGO']:
            aq_kwargs['X_baseline'] = X_baseline
      
        # Hypervolume requires reference point
        if aq in ['EHVI', 'NEHVI']:

            # The reference point must be in the objective space, by default
            # use the minimum point - 10% of the range
            ref_point = hps['ref_point']
            if ref_point is None:
                max = o.max(dim=0).values
                min = o.min(dim=0).values
                ref_point = min - 0.1 * (max - min)
            else:
                ref_point = torch.tensor(ref_point)

            aq_kwargs['ref_point'] = ref_point
            
        if aq in ['NParEGO', 'NEHVI']:
            aq_kwargs['prune_baseline'] = True
            
        if aq == 'EHVI':
            aq_kwargs['partitioning'] = NondominatedPartitioning(aq_kwargs['ref_point'], Y=o)

        if aq == 'NIPV':
            X_bounds = torch.tensor([[0.0, 1.0]]*self.X_space.n_tdim, dtype=TORCH_DTYPE).T
            qmc_samples = draw_sobol_samples(bounds=X_bounds, n=128, q=m_batch)
            aq_kwargs['mc_points'] = qmc_samples.squeeze(-2)
            aq_kwargs['sampler'] = None
            if objective:
                raise UnsupportedError('NIPV does not support objectives')

        if aq == 'NParEGO':
            w = hps['scalarization_weights']
            if isinstance(w, list):
                w = torch.tensor(w)
                w = w/torch.sum(torch.abs(w))
            aq_kwargs['scalarization_weights'] = w

        return aq_kwargs

    def suggest(self,
                m_batch: int = 1,
                target: Target | list[Target] | None = None,
                acquisition: list[str] | list[dict] = None,
                optim_sequential: bool = True,
                optim_samples: int = 512,
                optim_restarts: int = 10,
                objective: MCAcquisitionObjective | None = None,
                out_constraints: Output_Constraint | list[Output_Constraint] | None = None,
                eq_constraints: Linear_Constraint | list[Linear_Constraint] | None = None,
                ineq_constraints: Linear_Constraint | list[Linear_Constraint] | None = None,
                nleq_constraints: Nonlinear_Constraint | list[Nonlinear_Constraint] | None = None,
                task_index: int = 0,
                fixed_var: dict[str: float | str] | None = None,
                X_pending: pd.DataFrame | None = None,
                eval_pending: pd.DataFrame | None = None) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """
        Suggest future experiments based on a maximization of some acquisition
        function calculated from the expectation of a surrogate model.

        Args:
            m_batch (int, optional): The number of experiments to suggest at once. The default is ``1``.
            target (Target or list of Target, optional): The response(s) to be used for optimization,
            acquisition (list of str or list of dict, optional): Indicator for the desired acquisition function(s).
                A list will propose experiments for each acquisition function based on ``optim_sequential``.
                
                The default is ``['NEI']`` for single-output and ``['NEHVI']`` for multi-output.
                Options are as follows:
                
                - ``'EI'``: Expected Improvement (relative to best of ``y_train``). Accepts hyperparameter
                  ``'inflate'``, a positive or negative float to inflate/deflate the best point for explore/exploit.
                - ``'NEI'``: Noisy Expected Improvement. More robust than ``EI`` and uses all of ``y_train``,
                  but accepts no hyperparameters.
                - ``'PI'``: Probability of Improvement (relative to best of ``y_train``). Accepts hyperparameter
                  ``'inflate'``, a positive or negative float to inflate/deflate the best point for explore/exploit.
                - ``'UCB'``: Upper Confidence Bound. Accepts hyperparameter ``'beta'``, a positive float which sets
                  the number of standard deviations above the mean.
                - ``'SR'``: Simple Regret
                - ``'RS'``: Random Sampling
                - ``'Mean'``: Mean of the posterior distribution (pure exploitation/maximization of objective)
                - ``'SF'``: Space Filling. Requests points that maximize the minimumd distance to ``X_train`` based
                  on Euclidean distance.
                - ``'NIPV'``: Negative Integrated Posterior Variance. Requests the point which most improves the prediction
                  interval for a random selection of points in the design space. Used for active learning.
                - ``'EHVI'``: Expected Hypervolume Improvement. Can accept a ``ref_point``, otherwise a point just
                  below the minimum of ``y_train``.
                - ``'NEHVI'``: Noisy Expected Hypervolume Improvement. Can accept a ``ref_point``, otherwise a point
                  just below the minimum of ``y_train``.
                - ``'NParEGO'``: Noisy Pareto Efficient Global Optimization. Can accept ``scalarization_weights``, a
                  list of weights for each objective.
                
            optim_sequential (bool, optional): Whether or not to optimize batch designs sequentially
                (by fantasy) or simultaneously. Default is ``True``.
            optim_samples (int, optional): The number of samples to use for quasi Monte Carlo sampling
                of the acquisition function. Also used for initializing the acquisition optimizer.
                The default value is ``512``.
            optim_restarts (int, optional): The number of restarts to use in the global optimization
                of the acquisition function. The default value is ``10``.
            objective (MCAcquisitionObjective, optional): The objective function to be used for optimization.
                The default is ``None``.
            out_constraints (Output_Constraint | list[Output_Constraint], optional): An output constraint, or a list
                thereof, restricting the search space by outcomes. The default is ``None``.
            eq_constraints (Linear_Constraint | list[Linear_Constraint], optional): A linear constraint, or a list
                thereof, restricting the search space by equality (=). The default is ``None``.
            ineq_constraints (Linear_Constraint | list[Linear_Constraint], optional):  A linear constraint, or a list
                thereof, restricting the search space by inequality (>=). The default is ``None``.
            nleq_constraints (Nonlinear_Constraint | list[Nonlinear_Constraint], optional):  A nonlinear constraint,
                or a list thereof, restricting the search space by nonlinear feasibility. The default is ``None``.
            task_index (int, optional): The index of the task to optimize for multi-task models. The default is ``0``.
            fixed_var (dict(str:float), optional): Name of a variable and setting, over which the
                suggestion should be fixed. Default values is ``None``
            X_pending (pd.DataFrame, optional): Experiments that are expected to be run before the next optimal set
            eval_pending (pd.DataFrame, optional): Acquisition values associated with X_pending

        Returns:
            tuple[pd.DataFrame, pd.DataFrame] = (X_suggest, eval_suggest)
                X_suggest (pd.DataFrame): Experiment matrix of real input variables,
                    selected by optimizer.
                eval_suggest (pd.DataFrame): Mean results (response, prediction interval, f(response), obj
                    function for
                    each suggested experiment.
        
        Raises:
            UnfitError: If the surrogate model has not been fit before suggesting new experiments.
            TypeError: If the target is not a Target object or a list of Target objects.
            IncorrectObjectiveError: If the objective does not successfully execute on a sample.
            TypeError: If the acquisition is not a list of strings or dictionaries.
            UnsupportedError: If the provided acquisition function does not support output constraints.

        """

        if not self.is_fit:
            raise UnfitError('Surrogate model must be fit before suggesting new experiments')
            
        if self.verbose >= 2:
            print(f'Optimizing {m_batch} experiments [...]')
        
        # Use indexing to handle if suggestions are made for a subset of fit targets/surrogates
        target = self._validate_target(target)
        target_locs = [self.y_names.index(t.name) for t in target]
        
        # Select the model(s) to use for optimization
        model_list = [one_surrogate.torch_model for i, one_surrogate in enumerate(self.surrogate) if i in target_locs]
        if all(isinstance(m, GPyTorchModel) for m in model_list):
            model = ModelListGP(*model_list)
        else:
            model = ModelList(*model_list)

        # Make sure that model/objective outputs match the input requirements for aqs
        # In order to determine the number of outputs, considering objectives,
        # just call the objective on a random sample, and check the output dims
        if objective:
            try:
                X_sample = self.X_train.iloc[0, :].to_frame().T
                eval_suggest = self.evaluate(X_sample, target=target, objective=objective)
                o_dim = len([col for col in eval_suggest.columns if 'Objective' in col])
            except Exception:
                raise IncompatibleObjectiveError('Objective(s) did not successfully execute on sample')
        else:
            o_dim = len(target_locs)

        optim_type = 'single' if o_dim == 1 else 'multi'

        # Default if no aq method is provided
        if not acquisition:
            acquisition = [aq_defaults[optim_type]]

        # Type check for acquisition
        if not isinstance(acquisition, list):
            raise TypeError('acquisition must be a list of strings or dictionaries')
        if not all(isinstance(item, (str, dict)) for item in acquisition):
            raise TypeError('Each item in acquisition list must be either a string or a dictionary')
        
        # Compute static variable inputs
        fixed_features_list = self._fixed_features(fixed_var)
        
        # Set up the sampler, for MC-based optimization of acquisition functions
        if not isinstance(model, ModelListGP):
            samplers = []
            for m in model.models:
                if isinstance(m, DNN):
                    sampler_i = IndexSampler(sample_shape=torch.Size([optim_samples]), seed=self.seed)
                else:
                    sampler_i = SobolQMCNormalSampler(sample_shape=torch.Size([optim_samples]), seed=self.seed)
                samplers.append(sampler_i)
            sampler = ListSampler(*samplers)
        else:
            sampler = SobolQMCNormalSampler(sample_shape=torch.Size([optim_samples]), seed=self.seed)
            
        # Calculate search bounds for optimization
        X_bounds = torch.tensor(self.X_space.search_space.values, dtype=TORCH_DTYPE)
        
        # Set up master lists to hold the candidates from multi-acquisition results
        candidates_all = []
        eval_suggest = pd.DataFrame()

        # Incorporate previously suggested X values, if provided
        if X_pending is not None:
            m_pending = X_pending.shape[0]
            candidates_pending = torch.tensor(self.X_space.encode(X_pending).values)
            candidates_all.append(candidates_pending)
            X_t_pending = torch.concat(candidates_all)
            if eval_pending is None:
                eval_suggest['aq Method'] = ['User Provided']*m_pending
            else:
                eval_suggest = eval_pending
        else:
            X_t_pending = None

        #  Select the task to optimize for a multi-task model
        if self.X_space.X_task:
            if objective is not None:
                objective = Objective_Sequence([Index_Objective(task_index), objective])
            task_param = next(x for x in self.X_space if isinstance(x, Task))
            task_name = task_param.name
            task_value = task_param.encode([task_param.categories[task_index]])

        # Proceed with the optimization over each set of acquisition/hypers
        for aq_i in acquisition:
            # Extract acq function names and custom hyperparameters from the 'acquisition' list in config
            aq_str, aq_hps = self._validate_hypers(o_dim, aq_i)

            # Use aq_kwargs so that extra unnecessary ones in hps get removed for certain aq funcs
            aq_kwargs = {'model': model, 'sampler': sampler, 'X_pending': X_t_pending}
            
            aq_kwargs.update(self._parse_aq_kwargs(aq_str, aq_hps, m_batch, target_locs, X_t_pending, objective))

            # Raise errors related to certain constraints
            if aq_str in ['UCB', 'Mean', 'TS', 'SF', 'SR', 'NIPV']:
                if out_constraints is not None:
                    raise UnsupportedError('Provided aquisition function does not support output constraints')
            else:
                if out_constraints and not isinstance(out_constraints, list):
                    out_constraints = [out_constraints]
                aq_kwargs['constraints'] = [c.forward(scale=objective is None)
                                            for c in out_constraints] if out_constraints else None

            # If NoneType, coerce to list
            if not eq_constraints:
                eq_constraints = []
            if not ineq_constraints:
                ineq_constraints = []
            if not nleq_constraints:
                nleq_constraints = []

            # Coerce input constraints to lists
            if not isinstance(eq_constraints, list):
                eq_constraints = [eq_constraints]
            if not isinstance(ineq_constraints, list):
                ineq_constraints = [ineq_constraints]
            if not isinstance(nleq_constraints, list):
                nleq_constraints = [nleq_constraints]

            # Append X_space constraints
            if getattr(self.X_space, 'linear_constraints', []):
                for c in self.X_space.linear_constraints:
                    if c.equality:
                        eq_constraints.append(c)
                    else:
                        ineq_constraints.append(c)
            if getattr(self.X_space, 'nonlinear_constraints', []):
                nleq_constraints += self.X_space.nonlinear_constraints

            # Input constraints are used by optim_acqf and friends
            optim_kwargs = {'equality_constraints': [c() for c in eq_constraints] if eq_constraints else None,
                            'inequality_constraints': [c() for c in ineq_constraints] if ineq_constraints else None,
                            'nonlinear_inequality_constraints': [c() for c in nleq_constraints] if nleq_constraints else None}
            
            optim_options = {}  # Can optionally specify batch_limit or max_iter
            
            # If nonlinear constraints are used, BoTorch doesn't provide an ic_generator
            # Must provide manual samples = just use random initialization
            if nleq_constraints:
                X_ic = torch.ones((optim_samples, 1 if fixed_features_list else m_batch, self.X_space.n_tdim))*torch.rand(1)
                optim_kwargs['batch_initial_conditions'] = X_ic
                if fixed_features_list:
                    raise UnsupportedError('Nonlinear constraints are not supported with discrete features.')
            
            # Hypervolume aqs fail with X_t_pending when optim_sequential=True
            if aq_str in ['NEHVI', 'EHVI']:
                optim_sequential = False

            # If it's random search, no need to do optimization; Otherwise, initialize the aq function and optimize
            if aq_str == 'RS':
                candidates = torch.rand((m_batch, self.X_space.n_tdim), dtype=TORCH_DTYPE)
            else:
                aq_func = aq_class_dict[aq_str](**aq_kwargs).to(TORCH_DTYPE)
                
                # If there are any discrete values, we must used the mixed integer optimization
                if fixed_features_list:
                    candidates, _ = optimize_acqf_mixed(acq_function=aq_func, bounds=X_bounds,
                                                        fixed_features_list=fixed_features_list,
                                                        q=m_batch,  # Always sequential
                                                        num_restarts=optim_restarts, raw_samples=optim_samples,
                                                        options=optim_options,
                                                        **optim_kwargs)
                else:
                    candidates, _ = optimize_acqf(acq_function=aq_func, bounds=X_bounds,
                                                  q=m_batch,
                                                  sequential=optim_sequential,
                                                  num_restarts=optim_restarts, raw_samples=optim_samples,
                                                  options=optim_options,
                                                  **optim_kwargs)
            
            if self.verbose >= 2:
                print(f'Optimized {aq_str} acquisition function successfully')
            
            candidates_i = self.X_space.decode(
                pd.DataFrame(candidates.detach().cpu().numpy(),
                             columns=[col for col in self.X_t_train.columns if col not in self.X_space.X_task]))
            
            if X_t_pending is not None:
                X_t_pending_i = X_t_pending
            else:
                X_t_pending_i = None
            
            eval_i = self.evaluate(candidates_i, X_t_pending_i,
                                   target=target, acquisition=aq_i, objective=objective, eval_aq=True)
            eval_suggest = pd.concat([eval_suggest, eval_i], axis=0).reset_index(drop=True)

            candidates_all.append(candidates)
            X_t_pending = torch.concat(candidates_all)
                       
        candidates_all = pd.DataFrame(X_t_pending.detach().cpu().numpy(),
                                      columns=[col for col in self.X_t_train.columns
                                               if col not in self.X_space.X_task])
    
        if self.X_space.X_task:
            candidates_all[task_name] = task_value
        
        # Arrange suggested experiments into a single dataframe, and compute predictions
        X_suggest = self.X_space.decode(candidates_all)
        
        return X_suggest, eval_suggest
    
    def evaluate(self,
                 X_suggest: pd.DataFrame,
                 X_t_pending: Tensor | None = None,
                 target: Target | list[Target] | None = None,
                 acquisition: str | dict = None,
                 objective: MCAcquisitionObjective | None = None,
                 eval_aq: bool = False) -> pd.DataFrame:
        """
        Args:
            X_suggest (pd.DataFrame): Experiment matrix of real input variables, selected by optimizer.
            X_t_pending (Tensor): Suggested experiments yet to be run
            target (Target or list of Target, optional): The response(s) to be used for optimization,
            acquisition (str | dict, optional): Acquisition function name (str) or dictionary
                containing the acquisition function name and its hyperparameters.
            objective (MCAcquisitionObjective, optional): The objective function to be used for optimization.
                The default is ``None``.
            eval_aq (bool, optional): Whether or not to also evaluate the aq function. The default is ``False``.
        
        Returns:
            pd.DataFrame: Response prediction, pred interval, transformed mean, aq value,
                and objective function evaluation(s)

        """
        
        if not self.is_fit:
            raise UnfitError('Surrogate model must be fit before evaluating new experiments')
                
        # Use indexing to handle if suggestions are made for a subset of fit targets/surrogates
        target = self._validate_target(target)
        target_locs = [self.y_names.index(t.name) for t in target]

        # Begin evaluation with y_predict with pred interval
        eval_suggest = self.predict(X_suggest)
        X_t = torch.tensor(self.X_space.encode(X_suggest).values, dtype=TORCH_DTYPE)
        X_t_train = torch.tensor(self.X_space.encode(self.X_train).values, dtype=TORCH_DTYPE)

        # Evaluate f_predict on new and pending points
        f_all = []
        f_train_all = []
        f_pending_all = []
        for loc in target_locs:
            # Predict: new suggestions
            t_model = self.surrogate[loc]
            mu_i, _ = t_model.predict(self.X_space.encode(X_suggest))
            f_all.append(mu_i.unsqueeze(1))
            eval_suggest[f'f({self.target[loc].name})'] = mu_i

            # Training data on select targets
            mu_i_train, _ = t_model.predict(self.X_space.encode(self.X_train))
            f_train_all.append(mu_i_train.unsqueeze(1))

            # Pending points if provided
            if X_t_pending is not None:
                mu_i_pending, _ = t_model.predict(pd.DataFrame(X_t_pending,
                                                               columns=[col for col in self.X_t_train.columns
                                                                        if col not in self.X_space.X_task]))
                f_pending_all.append(mu_i_pending.unsqueeze(1))

        f_t = torch.concat(f_all, dim=1)
        f_t_train = torch.concat(f_train_all, dim=1)
        if X_t_pending is not None:
            f_t_pending = torch.concat(f_pending_all, dim=1)

        if objective:
            # Convert f_predict to tensor and evaluate objective
            # Must add sample dimension to f_t
            o = objective(f_t.unsqueeze(0), X_t).squeeze(0)
            if o.ndim < 2:
                o = o.unsqueeze(1)  # Rearrange into m x o

            # Store multiple objectives if applicable
            for o_i in range(o.shape[-1]):
                eval_suggest[f'Objective {o_i+1}'] = o[:, o_i].detach().cpu().numpy()

            # Also calculate for training and pending data (for total hv and pareto calcs)
            o_train = objective(f_t_train.unsqueeze(0), X_t_train).squeeze(0)
            if o_train.ndim < 2:
                o_train = o_train.unsqueeze(1)
            if X_t_pending is not None:
                o_pending = objective(f_t_pending.unsqueeze(0), X_t_pending).squeeze(0)
                if o_pending.ndim < 2:
                    o_pending = o_pending.unsqueeze(1)

            # Calculate output dimensionality and evaluate acquisition
            o_dim = o.shape[-1]
            
        else:
            o_dim = len(target_locs)

        optim_type = 'single' if o_dim == 1 else 'multi'
        
        if eval_aq:
            # Default if no aq method is provided
            if not acquisition:
                acquisition = [aq_defaults[optim_type]]

            if not isinstance(acquisition, (str, dict)):
                raise TypeError('Acquisition must be either a string or a dictionary')
            
            model_list = [one_surrogate.torch_model for i, one_surrogate in enumerate(self.surrogate) if i in target_locs]
            if all(isinstance(m, GPyTorchModel) for m in model_list):
                model = ModelListGP(*model_list)
            else:
                model = ModelList(*model_list)
            
            # Extract acq function names and custom hyperparameters from the 'acquisition' list in config
            aq_str, aq_hps = self._validate_hypers(o_dim, acquisition)

            # Use aq_kwargs so that extra unnecessary ones in hps get removed for certain aq funcs
            aq_kwargs = {'model': model, 'sampler': None, 'X_pending': X_t_pending}
                       
            aq_kwargs.update(self._parse_aq_kwargs(aq_str, aq_hps, X_suggest.shape[0], target_locs, X_t_pending, objective))
                
            # If it's random search, no need to evaluate aq
            if aq_str == 'RS':
                a_joint = torch.tensor([float('nan')]).repeat(X_t.shape[0]).unsqueeze(1)
            else:
                aq_func = aq_class_dict[aq_str](**aq_kwargs)

                # Evaluate acquisition on individual samples, then jointly
                a = []
                for x_i in X_t:
                    a_i = aq_func(x_i.unsqueeze(0))
                    a.append(a_i.detach().cpu())
                a = torch.concat(a).unsqueeze(1)
                a_joint = aq_func(X_t).repeat(X_t.shape[0]).unsqueeze(1)  # Rearrange into m x 1

                eval_suggest['aq Value'] = a.numpy()
                
            eval_suggest['aq Value (joint)'] = a_joint.detach().cpu().numpy()
            eval_suggest['aq Method'] = [aq_str]*X_t.shape[0]

        # For multi-output evaluations, calculate pareto and hv considering objectives
        if o_dim > 1:
            if objective is None:
                o = f_t
                o_list = [f_t, f_t_train]
                if X_t_pending is not None:
                    o_list.append(f_t_pending)
            else:
                o_list = [o, o_train]
                if X_t_pending is not None:
                    o_list.append(o_pending)
            o_all = torch.concat(o_list, dim=0)

            hv = self.hypervolume(o_all)
            eval_suggest['Expected Hypervolume (joint)'] = hv
            pf = self.pareto(o_all)
            eval_suggest['Expected Pareto'] = pf[-o.shape[0]:]

        return eval_suggest

    def maximize(self,
                 optim_samples=1026,
                 optim_restarts=50,
                 fixed_var: dict[str: float | str] | None = None) -> tuple[pd.DataFrame, pd.DataFrame]:
        """
        Predicts the conditions which return the maximum response value within the parameter space.

        Args:
            optim_samples (int): The number of samples to be used for optimization. Default is ``1026``.
            optim_restarts (int): The number of restarts for the optimization process. Default is ``50``.
            fixed_var (dict(str:float), optional): Name of a variable and setting, over which the
                            suggestion should be fixed. Default values is ``None``
        Returns:
            tuple[pd.DataFrame, pd.DataFrame] = (X_suggest, eval_suggest)
                X_suggest (pd.DataFrame): Experiment matrix of real input variables,
                    selected by optimizer.
                y_suggest (pd.DataFrame): Mean results and prediction interval for
                    each suggested experiment.
        """
        
        X_suggest = pd.DataFrame()
        eval_suggest = pd.DataFrame()

        for target in self.target:
            X_suggest_i, eval_suggest_i = self.suggest(
                m_batch=1, acquisition=['Mean'], optim_samples=optim_samples, optim_restarts=optim_restarts, target=target)
            X_suggest = pd.concat([X_suggest, X_suggest_i], axis=0)
            eval_suggest = pd.concat([eval_suggest, eval_suggest_i], axis=0)
        
        return X_suggest, eval_suggest
