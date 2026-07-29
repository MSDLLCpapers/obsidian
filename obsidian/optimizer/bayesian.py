"""Bayesian Optimization: Select experiments from the predicted posterior and update the prior"""

from typing import Any
import obsidian
from obsidian.acquisition.utils import ParserContext
from obsidian.rng import RNGManager, validate_seed
from obsidian.utils import TaskType
from .base import Optimizer

from obsidian.parameters import ParamSpace, Target, Task
from obsidian.surrogates import SurrogateBoTorch, EnsembleModel
from obsidian.acquisition import registry, aq_defaults, unconstrainable_aqs
from obsidian.surrogates import model_class_dict
from obsidian.objectives import Index_Objective, Objective_Sequence
from obsidian.constraints import Linear_Constraint, Nonlinear_Constraint, Output_Constraint
from obsidian.exceptions import IncompatibleObjectiveError, UnsupportedError, UnfitError, DataWarning, OptimizerWarning
from obsidian.config import TORCH_DTYPE

from botorch.acquisition.objective import MCAcquisitionObjective
from botorch.optim import optimize_acqf, optimize_acqf_mixed
from botorch.sampling import SobolQMCNormalSampler
from botorch.sampling.list_sampler import ListSampler
from botorch.sampling.index_sampler import IndexSampler
from botorch.models.model_list_gp_regression import ModelListGP
from botorch.models.gpytorch import GPyTorchModel
from botorch.models.model import ModelList, Model

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
                 task: TaskType | str = TaskType.OPTIMIZATION,
                 seed: int | None = None,
                 rng: RNGManager | None = None,
                 fix_random_state: bool = True,
                 verbose: int = 1):
       
        super().__init__(
            X_space=X_space, task=task, seed=seed, rng=rng, fix_random_state=fix_random_state, verbose=verbose
        )

        self.surrogate_type = []  # Shorthand name as str (as provided)
        self.surrogate_hps = []  # Hyperparameters

        self.aq_args: dict[str, dict[str, Any]] = {}

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

    def _fit(self, Z: pd.DataFrame, target: Target | list[Target], fit_options: dict | None = None):
        """
        Fits the BO surrogate model to data. The user-facing ``fit`` method is inherited from the base Optimizer class,
        which handles RNG control and then calls this method to perform the actual fitting.

        Args:
            Z (pd.DataFrame): Total dataset including inputs (X) and response values (y)
            target (Target or list of Target): The responses (y) to be used for optimization,
                packed into a Target object or list thereof
            fit_options (dict, optional): Additional options to customize the fitting process. Refer to the model's `fit` method for details.

        Returns:
            None. Updates the model in self.surrogate

        Raises:
            NameError: If the target is not present in the data.
            ValueError: If the number of responses does not match the number of specified surrogate models.
        """
        fit_options = fit_options or {}

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
            model = SurrogateBoTorch(
                model_type=self.surrogate_type[i],
                seed=self.seed,
                verbose=self.verbose >= 2,
                hps=self.surrogate_hps[i],
            )

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
            model.fit(
                X_t_train_valid,
                f_train_i_valid,
                cat_dims=self.X_space.X_t_cat_idx,
                task_feature=self.X_space.X_t_task_idx,
                **fit_options,
            )

            if self.verbose >= 1:
                print(f'{self.surrogate_type[i]} model has been fit to data'
                      + f' with an R2-train-score of: {model.r2_score:.3g}'
                      + (f' and a training-loss of: {model.loss:.3g}' if self.verbose >= 2 else '')
                      + f' for response: {self.y_names[i]}')
            self.surrogate.append(model)


    def save_state(self) -> dict:
        """
        Saves the parameters of the Bayesian Optimizer so that they can be reloaded.

        This method can save both fitted and unfitted optimizers:

        - **Unfitted**: Saves configuration only (X_space, surrogate spec, RNG state)
        - **Fitted**: Saves configuration + training data + fitted models

        Returns:
            dict: A dictionary containing the optimizer state for later loading.

        Examples:
            >>> # Save unfitted optimizer
            >>> opt = BayesianOptimizer(X_space, seed=42)
            >>> state = opt.save_state()  # Works without fitting

            >>> # Load and then fit
            >>> opt2 = BayesianOptimizer.load_state(state)
            >>> opt2.fit(data, target=target)
        """
        # if not self.is_fit:
        #     raise UnfitError('Surrogate model must be fit before saving optimizer')

        # Prepare a dictionary to describe the state
        config_save = {'opt_attrs': {},
                       'X_space': self.X_space.save_state(),
                       'surrogate_spec': [{func: hps} for func, hps in zip(self.surrogate_type, self.surrogate_hps)],
                       'is_fitted': self.is_fit}

        # Log if saving unfitted optimizer
        if not self.is_fit and self.verbose:
            print("Saving unfitted optimizer (configuration only)")

        # Always save basic attributes
        config_save['opt_attrs']['seed'] = self.seed
        config_save['opt_attrs']['task'] = self.task.value

        # Save RNG state if new RNG control is enabled
        if not obsidian.USE_OLD_RNG_CONTROL:
            config_save['rng_state'] = self.rng.save_state()
            config_save['fix_random_state'] = self.fix_random_state
            if self.model_generator:
                config_save['model_generator_state'] = self.model_generator.bit_generator.state

        # Conditionally save fit-dependent attributes
        if self.is_fit:
            config_save['target'] = [t.save_state() for t in self.target]

            # Select optimizer attributes to save
            fit_attrs = ['X_train', 'y_train', 'y_names', 'n_response', 'X_best_f_idx', 'X_best_f']

            for attr in fit_attrs:
                if isinstance(getattr(self, attr), (pd.Series, pd.DataFrame)):
                    config_save['opt_attrs'][attr] = getattr(self, attr).to_dict()
                else:
                    config_save['opt_attrs'][attr] = getattr(self, attr)

            # Save surrogate model states
            model_states = []
            for surrogate in self.surrogate:
                model_states.append(surrogate.save_state())
            config_save['model_states'] = model_states

        return config_save
    
    def __repr__(self):
        return f'BayesianOptimizer(X_space={self.X_space}, surrogate={self.surrogate_type}, target={getattr(self, "target", None)})'

    @classmethod
    def load_state(cls, config_save: dict):
        """
        Loads the parameters of the Bayesian Optimizer from a saved state.

        Can load both fitted and unfitted optimizers.

        Args:
            config_save (dict): A dictionary containing the saved state.

        Returns:
            BayesianOptimizer: Loaded optimizer instance

        Raises:
            ValueError: If the number of saved models does not match the number of named models.
        """
        # Check for is_fitted flag and handle backward compatibility
        if 'is_fitted' in config_save:
            is_fitted = config_save['is_fitted']
        else:
            # Legacy save files: infer from presence of model_states
            is_fitted = 'model_states' in config_save
            if is_fitted:
                # Info message for legacy fitted saves (not a warning, just informational)
                print("Loading state saved with older version (missing 'is_fitted' flag)")

        # Warn when loading unfitted optimizer
        if not is_fitted:
            warnings.warn(
                "Loading unfitted optimizer - call fit() before making predictions",
                UserWarning,
                stacklevel=2
            )

        # Restore RNG state if saved
        rng = None
        fix_random_state = True  # Default
        if 'rng_state' in config_save:
            rng = RNGManager.load_state(config_save['rng_state'])
            fix_random_state = config_save.get('fix_random_state', True)

        seed = config_save['opt_attrs'].get('seed', None)

        new_opt = cls(X_space=ParamSpace.load_state(config_save['X_space']),
                      surrogate=config_save['surrogate_spec'],
                      seed=seed,
                      rng=rng,
                      fix_random_state=fix_random_state)
        if 'model_generator_state' in config_save:
            new_opt.model_generator = np.random.default_rng()
            new_opt.model_generator.bit_generator.state = config_save['model_generator_state']

        # Load target if present (fitted optimizer)
        if 'target' in config_save:
            new_opt.target = tuple([Target.load_state(t) for t in config_save['target']])

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

        new_opt.task = TaskType.from_value(new_opt.task)

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

    def _build_parser_context(
            self,
            m_batch: int,
            X_t_pending: torch.Tensor | None,
            target: list[Target],
            target_locs: list[int],
            objective: MCAcquisitionObjective | None = None
        ):

        # Establish baseline X from training and pending
        X_train = torch.tensor(self.X_space.encode(self.X_train).values, dtype=TORCH_DTYPE)
        if X_t_pending is not None:
            X_baseline = torch.concat([X_train, X_t_pending], axis=0) # type: ignore
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
        # Create parser context
        context: ParserContext = {
            "f_t": f_t,
            "X_baseline": X_baseline,
            "m_batch": m_batch,
            "n_dim": self.X_space.n_tdim,
            "target": target,
            "objective": objective,
            "n_obs": self.X_train.shape[0]
        }
        return context

    def _parse_aq_kwargs(
            self,
            aq_name: str,
            hps: dict[str, Any],
            model: ModelList | ModelListGP,
            sampler: ListSampler | SobolQMCNormalSampler | None,
            X_t_pending: torch.Tensor | None,
            objective: MCAcquisitionObjective | None,
            o_dim: int,
            target: list[Target],
            target_locs: list[int],
            m_batch: int,
            aq_kwargs: dict[str, Any] = {}
            ):
        """A wrapper function to validate and parse acquisition function hyperparameters."""
        # Use aq_kwargs so that extra unnecessary ones in hps get removed for certain aq funcs
        aq_kwargs.update({'model': model, 'sampler': sampler, 'X_pending': X_t_pending, "objective": objective})
        # Extract acq function names and custom hyperparameters from the 'acquisition' list in config
        aq_kwargs, aq_hps = registry.validate_hyperparameters(self.task, o_dim, aq_name, hps, aq_kwargs)
        context = self._build_parser_context(m_batch, X_t_pending, target, target_locs, objective)
        aq_kwargs = registry.parse_hyperparameters(aq_name, aq_kwargs, aq_hps, context)
        return aq_kwargs

    def _setup_model_and_objective(
            self,
            target: Target | list[Target] | None,
            objective: MCAcquisitionObjective | None
        ) -> tuple[ModelList | ModelListGP, int, list[int], list[Target]]:
        """
        Set up the model list and validate objective dimensions.
        
        Args:
            target: Target or list of Target objects
            objective: MCAcquisitionObjective or None
        
        Returns:
            tuple: (model, o_dim, target_locs, target)
        
        Raises:
            IncompatibleObjectiveError: If the objective does not successfully execute on a sample.
        """
        target = self._validate_target(target)
        target_locs = [self.y_names.index(t.name) for t in target]
        
        model_list = [one_surrogate.torch_model for i, one_surrogate in enumerate(self.surrogate) 
                      if i in target_locs]
        if all(isinstance(m, GPyTorchModel) for m in model_list):
            model = ModelListGP(*model_list)
        else:
            model = ModelList(*model_list)
        
        # Determine output dimensions
        if objective:
            try:
                X_sample = self.X_train.iloc[0, :].to_frame().T
                eval_suggest = self.evaluate(X_sample, target=target, objective=objective)
                o_dim = len([col for col in eval_suggest.columns if 'Objective' in col])
            except Exception:
                raise IncompatibleObjectiveError('Objective(s) did not successfully execute on sample')
        else:
            o_dim = len(target_locs)
        
        return model, o_dim, target_locs, target

    @property
    def non_tracking_targets(self) -> list[Target]:
        """Targets eligible for suggestion (all non-tracking targets)."""
        return [t for t in self.target if not t.tracking_only]

    def _validate_suggestion_target(
            self,
            target: Target | list[Target] | None = None,
        ) -> list[Target]:
        """Validate and normalize targets for suggestion-only operations."""
        if target is None:
            non_tracking_targets = self.non_tracking_targets
            if not non_tracking_targets:
                raise UnsupportedError('No suggestible targets available: all fitted targets are tracking-only')
            return non_tracking_targets

        target = self._validate_target(target)
        fitted_targets = {t.name: t for t in self.target}
        selected_targets: list[Target] = []
        warned_tracking = set()

        for t in target:
            if t.name not in fitted_targets:
                raise NameError(f'Specified target {t.name} is not present in fitted targets')

            target_fit = fitted_targets[t.name]
            if target_fit.tracking_only and t.name not in warned_tracking:
                warnings.warn(
                    f'Target {t.name} is tracking-only and was explicitly requested for suggestion.',
                    UserWarning,
                    stacklevel=3,
                )
                warned_tracking.add(t.name)
            selected_targets.append(target_fit)

        return selected_targets

    def _setup_constraints(
            self,
            eq_constraints: Linear_Constraint | list[Linear_Constraint] | None,
            ineq_constraints: Linear_Constraint | list[Linear_Constraint] | None,
            nleq_constraints: Nonlinear_Constraint | list[Nonlinear_Constraint] | None
        ) -> tuple[list[Linear_Constraint], list[Linear_Constraint], list[Nonlinear_Constraint]]:
        """
        Consolidate and validate all constraints.
        
        Args:
            eq_constraints: Equality constraints (single or list)
            ineq_constraints: Inequality constraints (single or list)
            nleq_constraints: Nonlinear constraints (single or list)
        
        Returns:
            tuple: (eq_constraints, ineq_constraints, nleq_constraints) as lists
        """
        # Coerce to lists
        if not eq_constraints:
            eq_constraints = []
        if not ineq_constraints:
            ineq_constraints = []
        if not nleq_constraints:
            nleq_constraints = []
        
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
        
        return eq_constraints, ineq_constraints, nleq_constraints

    def _optimize_single_acquisition(
        self,
        aq_i: str | dict[str, Any],
        model: ModelList | ModelListGP,
        sampler: ListSampler | SobolQMCNormalSampler,
        X_t_pending: Tensor | None,
        objective: MCAcquisitionObjective | None,
        o_dim: int,
        target: list[Target],
        target_locs: list[int],
        m_batch: int,
        manual_seed: int | None,
        out_constraints: Output_Constraint | list[Output_Constraint] | None,
        eq_constraints: list[Linear_Constraint],
        ineq_constraints: list[Linear_Constraint],
        nleq_constraints: list[Nonlinear_Constraint],
        fixed_var: dict[str, float | str] | None,
        optim_sequential: bool,
        optim_samples: int,
        optim_restarts: int,
        optim_options: dict | None,
    ) -> tuple[Tensor, pd.DataFrame, pd.DataFrame]:
        """
        Optimize a single acquisition function with given constraints.
        
        Args:
            aq_i: Acquisition function specification (string or dict)
            model: Surrogate model
            sampler: MC sampler
            X_t_pending: Pending experiments (torch tensor)
            objective: Objective function
            o_dim: Output dimension
            target: Target or list of targets
            target_locs: Target locations in y_names
            m_batch: Batch size
            out_constraints: Output constraints
            eq_constraints: Equality constraints
            ineq_constraints: Inequality constraints
            nleq_constraints: Nonlinear constraints
            fixed_var: Fixed variables dict
            optim_sequential: Whether to optimize sequentially
            optim_samples: Number of optimization samples
            optim_restarts: Number of optimization restarts
            optim_options: Additional optimization options
        
        Returns:
            tuple: (candidates_tensor, candidates_df, eval_df)
        
        Raises:
            UnsupportedError: If acquisition function doesn't support constraints or
                             if nonlinear constraints are used with discrete features
        """
        aq_str, aq_hps = self._normalize_aq_input(aq_i)
        
        # Validation for constraints
        aq_kwargs = {}
        if aq_str in unconstrainable_aqs:
            if out_constraints is not None:
                raise UnsupportedError(
                    f"Acquisition function '{aq_str}' does not support output constraints"
                )
        else:
            if out_constraints and not isinstance(out_constraints, list):
                out_constraints = [out_constraints]
            aq_kwargs['constraints'] = [c.forward(scale=objective is None)
                                        for c in out_constraints] if out_constraints else None
        
        aq_kwargs = self._parse_aq_kwargs(aq_str, aq_hps, model, sampler, X_t_pending, 
                                         objective, o_dim, target, target_locs, m_batch, aq_kwargs)
        
        # Cache parsed acquisition arguments
        self.aq_args[aq_str] = aq_kwargs
        
        # Compute fixed features
        fixed_features_list = self._fixed_features(fixed_var)
        if len(fixed_features_list) > 25:
            warnings.warn(f'The combinations of discrete features is large at {len(fixed_features_list)}.'
                          + ' Optimization will proceed very slowly due to the combinatorial explosion.'
                          + ' Recommend reducing the number of discrete parameters used.', OptimizerWarning)
        
        # Setup optimization kwargs
        X_bounds = torch.tensor(self.X_space.search_space.values, dtype=TORCH_DTYPE)
        optim_kwargs: dict[str, Any] = {
            "equality_constraints": [c() for c in eq_constraints] if eq_constraints else None,
            "inequality_constraints": [c() for c in ineq_constraints] if ineq_constraints else None,
            "nonlinear_inequality_constraints": [c() for c in nleq_constraints] if nleq_constraints else None,
        }

        # Check if nonlinear constraints require manual initial conditions
        if nleq_constraints and fixed_features_list:
            raise UnsupportedError('Nonlinear constraints are not supported with discrete features.')

        # Hypervolume aqs special handling
        if aq_str in ['NEHVI', 'EHVI']:
            if optim_sequential and X_t_pending is not None:
                warnings.warn('Hypervolume aqs with X_pending require joint optimization. '
                             'Setting optim_sequential to False', UserWarning)
                optim_sequential = False

        # Nonlinear constraints with batch_initial_conditions require joint optimization
        if nleq_constraints and optim_sequential:
            warnings.warn('Nonlinear constraints require joint optimization. \
                           Setting optim_sequential to False', UserWarning)
            optim_sequential = False
        
        # Optimize
        if aq_str == 'RS':
            def acqf_wrapper(fixed_features_list):  # type: ignore
                """Generate random candidates with optional fixed features"""
                aq_func = registry.instantiate_acquisition(aq_str, **aq_kwargs).to(TORCH_DTYPE)
                candidates = aq_func()

                # Apply fixed features if specified
                if fixed_features_list:
                    # RS samples uniformly; with discrete features, use first combination
                    if len(fixed_features_list) > 1:
                        warnings.warn('RS with discrete features will sample from first feature combination')
                    fixed_features = fixed_features_list[0]
                    for idx, val in fixed_features.items():
                        candidates[:, idx] = val

                return candidates, None
        else:
            def acqf_wrapper(fixed_features_list):
                aq_func = registry.instantiate_acquisition(aq_str, **aq_kwargs).to(TORCH_DTYPE)
                # If nonlinear constraints are used, generate initial conditions inside wrapper
                # so they use the seeded RNG state for deterministic behavior
                if nleq_constraints:
                    X_ic = torch.ones((optim_samples, 1 if fixed_features_list else m_batch, self.X_space.n_tdim))*torch.rand(1, dtype=TORCH_DTYPE)
                    optim_kwargs['batch_initial_conditions'] = X_ic

                if fixed_features_list:
                    optim_func = optimize_acqf_mixed
                    optim_kwargs["fixed_features_list"] = fixed_features_list
                else:
                    optim_func = optimize_acqf
                    optim_kwargs["sequential"] = optim_sequential

                candidates, acq_values = optim_func(
                    acq_function=aq_func,
                    bounds=X_bounds,
                    q=m_batch,
                    num_restarts=optim_restarts,
                    raw_samples=optim_samples,
                    options=optim_options,
                    **optim_kwargs,
                )
                return candidates, acq_values

        # Resolve the RNG seed once so the optimization and the subsequent acquisition
        # evaluation share it. This keeps the reported aq value consistent with the
        # optimized candidate, and advances model_generator at most once per suggest.
        # In old RNG-control mode there is no model_generator; _rng_wrapper is a no-op
        # there, so simply pass manual_seed through unchanged.
        resolved_seed = manual_seed if obsidian.USE_OLD_RNG_CONTROL else self._resolve_rng_seed(manual_seed)

        # Wrap with RNG control for deterministic behavior
        wrapped_func = self._rng_wrapper(acqf_wrapper, resolved_seed)
        candidates, _ = wrapped_func(fixed_features_list)

        if self.verbose >= 2:
            print(f'Optimized {aq_str} acquisition function successfully')

        # Decode candidates
        candidates_df = self.X_space.decode(
            pd.DataFrame(candidates.detach().cpu().numpy(),
                        columns=[col for col in self.X_t_train.columns
                                if col not in self.X_space.X_task]))

        # Evaluate reusing the same resolved seed for a reproducible aq value
        eval_df = self.evaluate(candidates_df, X_t_pending,
                               target=target, acquisition=aq_i,
                               objective=objective, eval_aq=True,
                               manual_seed=resolved_seed)
        
        return candidates, candidates_df, eval_df

    def suggest(self,
                m_batch: int = 1,
                target: Target | list[Target] | None = None,
                acquisition: list[str] | list[dict] | None = None,
                optim_sequential: bool = True,
                optim_samples: int = 512,
                optim_restarts: int = 10,
                optim_options: dict | None = None,
                manual_seed: int | None = None,
                objective: MCAcquisitionObjective | None = None,
                out_constraints: Output_Constraint | list[Output_Constraint] | None = None,
                eq_constraints: Linear_Constraint | list[Linear_Constraint] | None = None,
                ineq_constraints: Linear_Constraint | list[Linear_Constraint] | None = None,
                nleq_constraints: Nonlinear_Constraint | list[Nonlinear_Constraint] | None = None,
                task_index: int = 0,
                fixed_var: dict[str, float | str] | None = None,
                X_pending: pd.DataFrame | None = None,
                eval_pending: pd.DataFrame | None = None,
                ) -> tuple[pd.DataFrame, pd.DataFrame]:
        """
        Suggest future experiments based on a maximization of some acquisition
        function calculated from the expectation of a surrogate model.

        Args:
            m_batch (int, optional): The number of experiments to suggest at once. The default is ``1``.
            target (Target or list of Target, optional): The response(s) to be used for optimization.
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
            optim_options (dict, optional): Options to pass to the optimization routine directly. Refer to BoTorch's `optimize_acqf` function family, `gen_candidates_scipy`, `gen_candidates_torch`, and `scipy.optimize.minimize` for possible options.
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
                    function for each suggested experiment.
        
        Raises:
            UnfitError: If the surrogate model has not been fit before suggesting new experiments.
            TypeError: If the target is not a Target object or a list of Target objects.
            IncorrectObjectiveError: If the objective does not successfully execute on a sample.
            TypeError: If the acquisition is not a list of strings or dictionaries.
            UnsupportedError: If the provided acquisition function does not support output constraints.
        """
        
        if not self.is_fit:
            raise UnfitError('Surrogate model must be fit before suggesting new experiments')

        validate_seed(manual_seed)

        if self.verbose >= 2:
            print(f'Optimizing {m_batch} experiments [...]')

        target = self._validate_suggestion_target(target)

        # Setup model and objective
        model, o_dim, target_locs, target = self._setup_model_and_objective(target, objective)
        optim_type = 'single' if o_dim == 1 else 'multi'
        
        # Default acquisition
        if not acquisition:
            acquisition = [aq_defaults[self.task.value][optim_type]]
        
        # Type check
        if not isinstance(acquisition, list):
            raise TypeError('acquisition must be a list of strings or dictionaries')
        if not all(isinstance(item, (str, dict)) for item in acquisition):
            raise TypeError('Each item in acquisition list must be either a string or a dictionary')
        
        # Setup constraints
        eq_constraints, ineq_constraints, nleq_constraints = self._setup_constraints(
            eq_constraints, ineq_constraints, nleq_constraints)
        
        # Setup sampler
        sampler = self._setup_sampler(model, optim_samples, self.seed)
        
        # Handle pending experiments
        candidates_all = []
        eval_suggest = pd.DataFrame()
        
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
        
        # Handle task index
        task_name = None
        task_value = None
        if self.X_space.X_task:
            if objective is not None:
                objective = Objective_Sequence([Index_Objective(task_index), objective])  # type: ignore
            task_param = next(x for x in self.X_space if isinstance(x, Task))
            task_name = task_param.name
            task_value = task_param.encode(np.array([task_param.categories[task_index]]))
        
        # Reset parsed acquisition arguments
        self.aq_args = {}
        
        # OPTIMIZATION LOOP
        for aq_i in acquisition:
            candidates, candidates_df, eval_df = self._optimize_single_acquisition(
                aq_i=aq_i,
                model=model,
                sampler=sampler,
                X_t_pending=X_t_pending,
                objective=objective,
                o_dim=o_dim,
                target=target,
                target_locs=target_locs,
                m_batch=m_batch,
                manual_seed=manual_seed,
                out_constraints=out_constraints,
                eq_constraints=eq_constraints,
                ineq_constraints=ineq_constraints,
                nleq_constraints=nleq_constraints,
                fixed_var=fixed_var,
                optim_sequential=optim_sequential,
                optim_samples=optim_samples,
                optim_restarts=optim_restarts,
                optim_options=optim_options,
            )
            
            eval_suggest = pd.concat([eval_suggest, eval_df], axis=0).reset_index(drop=True)
            candidates_all.append(candidates)
            X_t_pending = torch.concat(candidates_all)
        
        # Finalize results
        candidates_all = pd.DataFrame(torch.concat(candidates_all).detach().cpu().numpy(),
                                      columns=[col for col in self.X_t_train.columns
                                              if col not in self.X_space.X_task])
        
        if self.X_space.X_task:
            candidates_all[task_name] = task_value
        
        X_suggest = self.X_space.decode(candidates_all)
        
        return X_suggest, eval_suggest
    
    def evaluate(self,
                 X_suggest: pd.DataFrame,
                 X_t_pending: Tensor | None = None,
                 target: Target | list[Target] | None = None,
                 acquisition: str | dict | None = None,
                 objective: MCAcquisitionObjective | None = None,
                 eval_aq: bool = False,
                 manual_seed: int | None = None) -> pd.DataFrame:
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
            manual_seed (int | None, optional): Seed applied to the acquisition evaluation (when
                ``eval_aq`` is True) so the reported aq value is reproducible and consistent with the
                optimized candidate. If ``None``, a seed is resolved from the optimizer's RNG state.
                Defaults to ``None``.
        
        Returns:
            pd.DataFrame: Response prediction, pred interval, transformed mean, aq value,
                and objective function evaluation(s)

        """
        
        if not self.is_fit:
            raise UnfitError('Surrogate model must be fit before evaluating new experiments')

        validate_seed(manual_seed)

        # Use indexing to handle if suggestions are made for a subset of fit targets/surrogates
        target = self._validate_target(target)
        target_locs = [self.y_names.index(t.name) for t in target]

        # Begin evaluation with y_predict with pred interval
        eval_suggest = self.predict(X_suggest)
        X_t = torch.tensor(self.X_space.encode(X_suggest).values, dtype=TORCH_DTYPE)
        X_t_train = torch.tensor(self.X_space.encode(self.X_train).values, dtype=TORCH_DTYPE)

        # Compute f_tensors for new, training, and pending points
        f_t = self._compute_f_tensors(X_suggest, target_locs)
        f_t_train = self._compute_f_tensors(self.X_train, target_locs)
        
        if X_t_pending is not None:
            X_pending_df = pd.DataFrame(X_t_pending.detach().cpu().numpy(),
                                        columns=[col for col in self.X_t_train.columns
                                                 if col not in self.X_space.X_task])
            f_t_pending = self._compute_f_tensors(X_pending_df, target_locs)
        
        # Add standardized predictions to eval_suggest
        for i, loc in enumerate(target_locs):
            eval_suggest[f'{self.target[loc].name} Standardized'] = f_t[:, i].detach().cpu().numpy()

        # Evaluate objectives
        o = self._evaluate_objectives(f_t, X_t, objective)
        o_train = self._evaluate_objectives(f_t_train, X_t_train, objective)
        if X_t_pending is not None:
            o_pending = self._evaluate_objectives(f_t_pending, X_t_pending, objective)
        
        # Store objective values and determine dimensionality
        if objective:
            for o_i in range(o.shape[-1]):
                eval_suggest[f'Objective {o_i+1}'] = o[:, o_i].detach().cpu().numpy()
            o_dim = o.shape[-1]
        else:
            o_dim = len(target_locs)

        optim_type = 'single' if o_dim == 1 else 'multi'
        
        if eval_aq:
            # Default if no aq method is provided
            if not acquisition:
                acquisition = aq_defaults[self.task.value][optim_type]

            if not isinstance(acquisition, (str, dict)):
                raise TypeError('Acquisition must be either a string or a dictionary')

            aq_str, aq_hps = self._normalize_aq_input(acquisition)

            if aq_str not in self.aq_args:
                model_list = [one_surrogate.torch_model for i, one_surrogate in enumerate(self.surrogate) if i in target_locs]
                if all(isinstance(m, GPyTorchModel) for m in model_list):
                    model = ModelListGP(*model_list)
                else:
                    model = ModelList(*model_list)
                aq_kwargs = {'model': model, 'sampler': None, 'X_pending': X_t_pending}
                aq_kwargs = self._parse_aq_kwargs(aq_str, aq_hps, model, None, X_t_pending, None, o_dim, target, target_locs, 1)     
            else:
                aq_kwargs = self.aq_args[aq_str]
            
            # Evaluate acquisition function under RNG control, mirroring the optimization
            # step, so any stochastic acquisition (e.g. randomized straddle beta sampling,
            # unseeded samplers) yields a reproducible value consistent with the candidate.
            eval_acq = self._rng_wrapper(self._evaluate_acquisition, manual_seed)
            a, a_joint = eval_acq(X_t, aq_str, aq_kwargs)
            
            if aq_str != 'RS':
                eval_suggest['aq Value'] = a.numpy()
            
            eval_suggest['aq Value (joint)'] = a_joint.detach().cpu().numpy() if a_joint is not None else float('nan')
            eval_suggest['aq Method'] = [aq_str]*X_t.shape[0]

        # For multi-output evaluations, calculate pareto and hv considering objectives
        if o_dim > 1:
            if objective is None:
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
                 acquisition=['Mean'],
                 optim_samples=1026,
                 optim_restarts=50,
                 fixed_var: dict[str, float | str] | None = None) -> tuple[pd.DataFrame, pd.DataFrame]:
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

        # Maximize intentionally evaluates all targets, so suppress tracking-only warnings in suggest
        with warnings.catch_warnings():
            warnings.filterwarnings(
                'ignore',
                message=r'Target .* is tracking-only and was explicitly requested for suggestion\.',
                category=UserWarning,
            )
            for target in self.target:
                X_suggest_i, eval_suggest_i = self.suggest(
                    m_batch=1, acquisition=acquisition, optim_samples=optim_samples, optim_restarts=optim_restarts,
                    target=target, fixed_var=fixed_var)
                X_suggest = pd.concat([X_suggest, X_suggest_i], axis=0)
                eval_suggest = pd.concat([eval_suggest, eval_suggest_i], axis=0)
        
        return X_suggest, eval_suggest

    def _compute_f_tensors(self, X_df: pd.DataFrame, target_locs: list[int]) -> Tensor:
        """
        Compute f (transformed objective) tensors for given X data.
        
        Args:
            X_df (pd.DataFrame): Input data in original space
            target_locs (list[int]): Indices of targets to compute
        
        Returns:
            torch.Tensor: Concatenated f predictions with shape (n_samples, n_targets)
        """
        f_all = []
        for loc in target_locs:
            t_model = self.surrogate[loc]
            mu_i, _ = t_model.predict(self.X_space.encode(X_df))
            f_all.append(mu_i.unsqueeze(1))
        
        return torch.concat(f_all, dim=1)

    def _evaluate_objectives(self, f_tensors: Tensor, X_tensors: Tensor, objective: MCAcquisitionObjective | None) -> Tensor:
        """
        Evaluate objective function(s) on f tensors.
        
        Args:
            f_tensors (torch.Tensor): Transformed objective values (n x m)
            X_tensors (torch.Tensor): Input tensors in encoded space (n x d)
            objective (MCAcquisitionObjective): Objective function
        
        Returns:
            torch.Tensor: Objective values with shape (n x o_dim)
        """
        if objective is None:
            return f_tensors
        
        # Evaluate objective with sample dimension
        o = objective(f_tensors.unsqueeze(0), X_tensors).squeeze(0)
        if o.ndim < 2:
            o = o.unsqueeze(1)  # Reshape to (n x 1) if 1D
        
        return o

    def _evaluate_acquisition(self, X_t: Tensor, aq_str: str, aq_kwargs: dict[str, Any]) -> tuple[Tensor, Tensor | None]:
        """
        Evaluate acquisition function on candidates.
        
        Args:
            X_t (torch.Tensor): Candidate points in encoded space
            aq_str (str): Acquisition function name
            aq_kwargs (dict): Acquisition function arguments
        
        Returns:
            tuple: (individual_values, joint_value) - both are torch Tensors
                individual_values: Acquisition values for each point individually
                joint_value: Acquisition value for all points jointly
        """
        # Random search has no acquisition value
        if aq_str == 'RS':
            return torch.tensor([float('nan')]).repeat(X_t.shape[0]).unsqueeze(1), None
        
        aq_func = registry.instantiate_acquisition(aq_str, **aq_kwargs)
        
        # Evaluate acquisition on individual samples
        a = []
        for x_i in X_t:
            a_i = aq_func(x_i.unsqueeze(0))
            a.append(a_i.detach().cpu())
        a = torch.concat(a).unsqueeze(1)
        
        # Evaluate acquisition jointly
        a_joint = aq_func(X_t).repeat(X_t.shape[0]).unsqueeze(1)
        
        return a, a_joint

    @staticmethod
    def _setup_sampler(model: Model | ModelListGP, optim_samples: int, seed: int | None) -> ListSampler | SobolQMCNormalSampler:
        if not isinstance(model, ModelListGP):
            samplers = []
            for m in model.models:
                if isinstance(m, EnsembleModel):
                    sampler_i = IndexSampler(sample_shape=torch.Size([optim_samples]), seed=seed)
                else:
                    sampler_i = SobolQMCNormalSampler(sample_shape=torch.Size([optim_samples]), seed=seed)
                samplers.append(sampler_i)
            sampler = ListSampler(*samplers)
        else:
            sampler = SobolQMCNormalSampler(sample_shape=torch.Size([optim_samples]), seed=seed)
        return sampler

    @staticmethod
    def _normalize_aq_input(acquisition: str | dict[str, Any]) -> tuple[str, dict[str, Any]]:
        # Parse acquisition input
        if isinstance(acquisition, str):
            aq_name = acquisition
            hps = {}
        elif isinstance(acquisition, dict):
            if len(acquisition) != 1:
                raise ValueError("One dictionary of hyperparameters must be provided for each acquisition function")
            aq_name, hps = next(iter(acquisition.items()))
            if not isinstance(hps, dict):
                raise TypeError("Hyperparameters must be provided as a dictionary")
        else:
            raise TypeError("Acquisition must be a string or a dictionary")
        return aq_name, hps