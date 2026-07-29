"""Campaign class definition"""

from obsidian.parameters import ParamSpace, Target
from obsidian.optimizer import Optimizer, BayesianOptimizer
from obsidian.experiment import ExpDesigner, designer_class_dict
from obsidian.objectives import Objective, Objective_Sequence, obj_class_dict
from obsidian.constraints import Output_Constraint, const_class_dict
from obsidian.exceptions import IncompatibleObjectiveError
from obsidian.utils import tensordict_to_dict, TaskType
from obsidian.rng import RNGManager
import obsidian

import pandas as pd
import torch
import warnings
import traceback


class Campaign():
    """
    Base class for tracking optimization progress and other metrics
    over multiple iterations.

    Args:
        X_space (ParamSpace): The parameter space for the campaign.
        target (Target | list[Target]): The target(s) for optimization.
        task (TaskType | str | None, optional): The task type: 'optimization' or 'characterization'.
            If ``None``, the campaign will issue a warning and default to
            ``TaskType.OPTIMIZATION``. Pass ``task='characterization'`` explicitly for
            characterization campaigns. Defaults to ``None``.
        constraints (Output_Constraint | list[Output_Constraint] | None, optional):
            Output constraints for the campaign. Defaults to ``None``.
        optimizer (Optimizer | None, optional): The optimizer to use. If ``None``, a
            :class:`BayesianOptimizer` will be created automatically with default settings.
            To control ``fix_random_state``, create your own optimizer instance and pass it here:

            .. code-block:: python

                # For stochastic variation
                opt = BayesianOptimizer(X_space, seed=123, fix_random_state=False)
                campaign = Campaign(X_space, target, optimizer=opt)

            Defaults to ``None``.
        designer (ExpDesigner | None, optional): The experimental designer for generating initial
            designs. If ``None``, an :class:`ExpDesigner` will be created automatically.
            Defaults to ``None``.
        objective (Objective | None, optional): The objective function for optimization. If ``None``,
            will be created automatically based on targets. Defaults to ``None``.
        seed (int | None, optional): Random seed for reproducibility. If ``None`` and no ``rng``
            is provided, a time-based seed will be generated. This seed is used to initialize the
            RNG for the campaign, optimizer, and designer. Defaults to ``None``.
        rng (RNGManager | None, optional): An existing :class:`RNGManager` instance to share across
            components. If provided, the campaign will use this shared RNG instead of creating its own.
            If both ``rng`` and ``seed`` are provided, ``seed`` is ignored. Defaults to ``None``.

    Attributes:
        X_space (ParamSpace): The parameter space for the campaign.
        data (pd.DataFrame): The data collected during the campaign.
        optimizer (Optimizer): The optimizer used for optimization.
        designer (ExpDesigner): The experimental designer used for experiment design.
        iter (int): The current iteration number.
        seed (int): The seed for random number generation.
        rng (RNGManager): The random number generator manager for the campaign.

    Properties:
        m_exp (int): The number of observations in campaign.data
        y (pd.Series): The response data in campaign.data
        y_names (list): The names of the response data columns
        f (pd.Series): The transformed response data
        o (pd.Series): The objective function evaluated on f
        o_names (list): The names of the objective function columns
        X (pd.DataFrame): The input features of campaign.data
        response_max (float | pd.Series): The maximum for each response
        target (Target | list[Target]): The target(s) for optimization.
        objective (Objective, optional): The objective of the optimization campaign

    Note:
        By default, campaigns use deterministic behavior (``fix_random_state=True`` in the
        optimizer). To enable stochastic variation, create a custom optimizer with
        ``fix_random_state=False`` and pass it to the campaign.

    """

    def __init__(self,
                 X_space: ParamSpace,
                 target: Target | list[Target],
                 task: TaskType | str | None = None,
                 constraints: Output_Constraint | list[Output_Constraint] | None = None,
                 optimizer: Optimizer | None = None,
                 designer: ExpDesigner | None = None,
                 objective: Objective | None = None,
                 seed: int | None = None,
                 rng: RNGManager | None = None
                 ):

        self.set_X_space(X_space)
        self.data = pd.DataFrame()

        if task is None:
            warnings.warn(
                "task not specified. Defaulting to 'optimization'. "
                "Pass task='characterization' explicitly for characterization campaigns.",
                UserWarning,
            )
            task = TaskType.OPTIMIZATION

        self.task = TaskType.from_value(task)
        if obsidian.USE_OLD_RNG_CONTROL:
            optimizer_seed = seed
            designer_seed = seed
            warnings.warn("Using old RNG control. This is deprecated.", UserWarning)
        else:
            if rng is None:
                self.rng = obsidian.create_rng_manager(seed)
                self._owns_rng = True
            else:
                # User provided explicit RNG to share
                self.rng = rng
                self._owns_rng = False
                print(
                    "Campaign is using a shared RNGManager instance. Reproducibility will depend how other objects consume from this RNG."
                )
                if seed is not None:
                    warnings.warn(
                        "Both `rng` and `seed` were provided. The seed parameter will be ignored "
                        "in favor of the seed from `rng`.", UserWarning
                    )
            seed = self.rng.seed

            optimizer_seed = seed
            designer_seed = seed

        if not optimizer:
            optimizer = BayesianOptimizer(
                X_space,
                task=self.task,
                rng=self.rng if not obsidian.USE_OLD_RNG_CONTROL else None,
                seed=optimizer_seed
            )
            self._owns_optimizer = True
        else:
            self._owns_optimizer = False
        self.set_optimizer(optimizer)

        # Sync the optimizer's task to the campaign's. A user-provided optimizer keeps
        # its constructor default (TaskType.OPTIMIZATION) otherwise, which silently
        # selects optimization-task acquisition defaults instead of characterization.
        if self._optimizer.task != self.task:
            if not self._owns_optimizer:
                warnings.warn(
                    f"Provided optimizer has task={self._optimizer.task.value!r}, "
                    f"overriding to match campaign task={self.task.value!r}.",
                    UserWarning,
                )
            self._optimizer.task = self.task

        if not designer:
            designer = ExpDesigner(
                X_space,
                seed=designer_seed
            )
        self.set_designer(designer)
        
        self.set_target(target)
        self.set_objective(objective)
        
        self.output_constraints = None
        self.constrain_outputs(constraints)
        
        # Non-object attributes
        self.iter = 0
        self.seed = seed
        self.version = obsidian.__version__

        # Number of rows in ``self.data`` at the time of the most recent ``fit()``.
        # Used to gate characterization analysis to freshly-fit state, so add_data()
        # and set_objective() don't run expensive Sobol+posterior metrics against a
        # stale optimizer.
        self._last_fit_n_rows: int | None = None

    def add_data(self, df: pd.DataFrame):
        """
        Adds data to the campaign.

        Args:
            Z_i (pd.DataFrame): The data to be added to the campaign.
        
        Raises:
            KeyError: If all X_names are not in the dataset
            KeyError: If all y_names are not in the dataset
        """
        
        if not all(name in df.columns for name in self.X_space.X_names):
            raise KeyError('Input dataset does not contain all of the required parameter names')
        if not all(name in df.columns for name in self.y_names):
            raise KeyError('Input dataset does not contain all of the required response target names')
        
        new_data = df.copy(deep=True)
        
        if 'Iteration' not in new_data.columns:
            new_data['Iteration'] = self.iter
        else:
            self.iter = int(new_data['Iteration'].max())

        self.iter += 1
        self.data = pd.concat([self.data, new_data], axis=0, ignore_index=True)
        self.data.index.name = 'Observation ID'
        self.data.index = self.data.index.astype('int')
        
        if self.optimizer.is_fit:
            self._analyze()

    def clear_data(self):
        """Clears campaign data"""
        self.data = pd.DataFrame()
        self.iter = 0
        self._last_fit_n_rows = None

    @property
    def X_space(self) -> ParamSpace:
        """Campaign ParamSpace"""
        return self._X_space
    
    def set_X_space(self, X_space: ParamSpace):
        """Sets the campaign ParamSpace"""
        self._X_space = X_space

    @property
    def optimizer(self) -> Optimizer:
        """Campaign Optimizer"""
        return self._optimizer
    
    def set_optimizer(self, optimizer: Optimizer):
        """Sets the campaign optimizer"""
        self._optimizer = optimizer
        
    @property
    def designer(self) -> ExpDesigner:
        """Campaign Experimental Designer"""
        return self._designer
    
    def set_designer(self, designer: ExpDesigner):
        """Sets the campaign experiment designer"""
        self._designer = designer

    @property
    def objective(self) -> Objective | None:
        """Campaign Objective function"""
        return self._objective
    
    def _eval_objective(self):
        """Evaluates objective and appends it to campaign data"""
        df_o = self.o
        for col in df_o.columns:
            self.data[col] = df_o[col].values
            self.o_names = [col for col in self.data.columns if 'Objective' in col]
    
    def set_objective(self, objective: Objective | None):
        """(Re)sets the campaign objective function"""
        self._objective = objective
        if not self.data.empty:
            # Remove previous objective evaluations
            self.data = self.data.drop(
                columns=[col for col in self.data.columns if 'Objective' in col]
                )
            if self.optimizer.is_fit:
                self._analyze()
                
    def clear_objective(self):
        """Clears the campaign objective function"""
        self._objective = None
                
    @property
    def target(self):
        """Campaign experimental target(s)"""
        return self._target

    def set_target(self,
                   target: Target | list[Target]):
        """
        Sets the experimental target context for the campaign.

        Args:
            target (Target | list[Target] | None): The target or list of targets to set.

        """
        if isinstance(target, Target):
            self._target = [target]
        else:
            self._target = target
        if all(t.tracking_only for t in self._target):
            warnings.warn(
                "All targets are tracking-only. Campaign will not optimize towards any target by default. "
                "Only use this campaign for analyzing data or informational purposes. "
                "Pass tracking-only targets explicitly to `suggest` to optimize towards them if this is truly intended.",
                UserWarning,
                stacklevel=2,
            )
        self.y_names = [t.name for t in self._target]
        self.n_response = len(self.y_names)

    @property
    def _is_mo(self) -> bool:
        """
        Boolean flag for multi-output
        """
        if self.objective:
            return self.objective._is_mo
        else:
            return self.n_response > 1

    @property
    def _is_characterization(self) -> bool:
        """
        Boolean flag for characterization task
        """
        return self.task == TaskType.CHARACTERIZATION
    
    @property
    def m_exp(self) -> int:
        """
        Number of observations in training data
        """
        return self.data.shape[0]

    @property
    def y(self) -> pd.Series | pd.DataFrame:
        """
        Experimental response data

        """
        if not self.data.empty:
            return self.data[self.y_names]
        else:
            return None

    @property
    def response_max(self) -> float | pd.Series:
        """
        Maximum response data in training set
        """
        return self.y.max()

    @property
    def f(self) -> pd.Series | pd.DataFrame:
        """
        Experimental response data, in transformed space
        """
        f = pd.concat([t.transform_f(self.y[t.name]) for t in self.target], axis=1)
        return f

    @property
    def o(self) -> pd.Series | pd.DataFrame:
        """
        Objective function evaluated on f
        """
        if self.objective:
            try:
                x = self.X_space.encode(self.X).values
                o = self.objective(torch.tensor(self.f.values).unsqueeze(0),
                                   X=torch.tensor(x)).squeeze(0)
                if o.ndim < 2:
                    o = o.unsqueeze(1)  # Rearrange into m x o
                return pd.DataFrame(o.detach().cpu().numpy(),
                                    columns=[f'Objective {o_i+1}' for o_i in range(o.shape[1])])
            except Exception:
                raise IncompatibleObjectiveError('Objective(s) did not successfully execute on sample')
        else:
            return None
    
    @property
    def out(self) -> pd.Series | pd.DataFrame:
        """
        Returns the objective function as appropriate, else the response data
        """
        if self.objective and self.optimizer.is_fit:
            return self.o
        else:
            return self.y
    
    @property
    def X_best(self) -> pd.DataFrame:
        """
        Best performing X values
        """
        best_idx = self.out.idxmax().values
        
        X_best = self.X.iloc[best_idx, :]
        if isinstance(X_best, pd.Series):
            X_best = X_best .to_frame().T
        
        return X_best
    
    @property
    def X(self) -> pd.DataFrame:
        """
        Feature columns of the training data
        """
        return self.data[list(self.X_space.X_names)]
            
    def __repr__(self):
        """String representation of object"""
        return f"obsidian Campaign for {getattr(self,'y_names', None)}; {getattr(self,'m_exp', 0)} observations"

    def initialize(self, **design_kwargs):
        """
        Maps ExpDesigner.initialize method
        """
        return self.designer.initialize(**design_kwargs)

    def fit(self, fit_options: dict | None = None):
        """
        Maps Optimizer.fit method

        Raises:
            ValueError: If no data has been registered to the campaign
        """
        fit_options = fit_options or {}

        if self.m_exp <= 0:
            raise ValueError('Must register data before fitting')

        self.optimizer.fit(self.data, target=self.target, fit_options=fit_options)
        self._last_fit_n_rows = len(self.data)
        self._analyze()

    def suggest(self, **optim_kwargs):
        """
        Maps Optimizer.suggest method
        """
        if self.optimizer.is_fit:
            try:
                # In case X_space has changed, re-set the optimizer X_space
                self.optimizer.set_X_space(self.X_space)
                # Use campaign attributes as defaults; caller-supplied values take precedence
                if "objective" not in optim_kwargs:
                    optim_kwargs["objective"] = self.objective
                if "out_constraints" not in optim_kwargs:
                    optim_kwargs["out_constraints"] = self.output_constraints
                X, eval = self.optimizer.suggest(**optim_kwargs)
                return (X, eval)
            except Exception as e:
                warnings.warn(f'Optimization failed: {e}')
                # print full traceback for debugging verbosity
                if self.optimizer.verbose > 2:
                    print("Stack trace:", traceback.format_exc())
                return None
        else:
            warnings.warn('Optimizer is not fit to data. Suggesting initial experiments.', UserWarning)
            X0 = self.initialize()
            return X0

    def evaluate(self, X_suggest: pd.DataFrame):
        """
        Maps Optimizer.evaluate method
        """
        return self.optimizer.evaluate(X_suggest, objective=self.objective)

    def evaluate_characterization(self, X: pd.DataFrame | int | None = None,
                                  PI_range: float = 0.7) -> dict:
        """
        Evaluate characterization metrics on specified points.

        Args:
            X (pd.DataFrame | int | None): Points to evaluate (pd.DataFrame),
                number of Sobol samples (int), or None to use training data
            PI_range (float): Prediction interval coverage (0.7 or 0.95)

        Returns:
            dict: Per-target and joint classification fractions

        Raises:
            ValueError: If campaign has no thresholds set
        """
        from obsidian.campaign.characterization import CharacterizationEvaluator

        if not self._has_thresholds():
            raise ValueError("Campaign must have at least one target with a threshold set")

        if X is None:
            X = self.X

        evaluator = CharacterizationEvaluator(self)
        return evaluator.classify_points(X, PI_range=PI_range)

    def score_against_ground_truth(self, X: pd.DataFrame,
                                   y_true,
                                   PI_range: float = 0.7) -> dict:
        """
        Score campaign predictions against ground truth (for benchmarking).

        Args:
            X (pd.DataFrame): Points to evaluate
            y_true (np.ndarray): Ground truth values, shape (n_points, n_targets)
            PI_range (float): Prediction interval coverage

        Returns:
            dict: Jaccard scores and confusion matrices

        Raises:
            ValueError: If campaign has no thresholds set
        """
        from obsidian.campaign.characterization import CharacterizationEvaluator
        import numpy as np

        if not self._has_thresholds():
            raise ValueError("Campaign must have at least one target with a threshold set")

        # Convert y_true to numpy if needed
        if not isinstance(y_true, np.ndarray):
            y_true = np.array(y_true)

        evaluator = CharacterizationEvaluator(self)
        return evaluator.evaluate_with_ground_truth(X, y_true, PI_range=PI_range)

    def _profile_hv(self):
        """
        Calculate and assign the hypervolume values to each iteration in the data.

        Returns:
            None
        """
        iters = self.data['Iteration'].unique()
        hv = {}
        
        for i in iters:
            iter_index = self.data.query(f'Iteration <= {i}').index
            out_iter = self.out.loc[iter_index, :]
            out_iter = torch.tensor(out_iter.values)
            hv[i] = self.optimizer.hypervolume(out_iter)
        
        self.data['Hypervolume (iter)'] = self.data.apply(lambda x: hv[x['Iteration']], axis=1)
        self.data['Pareto Front'] = self.optimizer.pareto(torch.tensor(self.out.values))

    def _profile_max(self):
        """
        Calculate the maximum values achieved for targets at each iteration

        Returns:
            None
        """

        # Remove previous max-profiling
        self.data = self.data.drop(
            columns=[col for col in self.data.columns if '(max) (iter)' in col]
        )

        for out in self.out.columns:
            self.data[out+' (max) (iter)'] = self.data.apply(
                lambda x: self.data.query(f'Iteration<={x["Iteration"]}')[out].max(), axis=1
            )

        return

    def _has_thresholds(self) -> bool:
        """
        Check if any active target has a threshold set (indicates characterization campaign).

        Tracking-only targets are excluded because characterization evaluation requires
        at least one non-tracking target with a threshold; including tracking-only
        targets here would incorrectly trigger characterization analysis for campaigns
        that have no actionable thresholds.

        Returns:
            bool: True if at least one non-tracking target has a threshold
        """
        return any(
            (not t.tracking_only) and (t.threshold is not None)
            for t in self.target
        )

    def _analyze_characterization(self):
        """
        Compute and add characterization classification metrics to campaign.data.

        Adds percentage metrics (pass %, fail %, classified %) at 70% and 95% CI
        for each target with a threshold, and joint metrics if multiple targets.

        Returns:
            None
        """
        from obsidian.campaign.characterization import CharacterizationEvaluator

        try:
            evaluator = CharacterizationEvaluator(self, seed=self.seed)
            N = evaluator.plan_sample_size(pilot_ratio=0.1, epsilon=0.01, z=1.96, max_samples=20000)
            summary = evaluator.summarize_confidence(N)

            current_iter = self.iter - 1
            idx = self.data.index[self.data["Iteration"] == current_iter]

            def _set(col, value):
                if col not in self.data.columns:
                    self.data[col] = float("nan")
                self.data.loc[idx, col] = value

            for name, row in summary.items():
                _set(f"Characterization {name} Pass % (mean)", row["pass_mean"] * 100)
                for ci_label, suffix in (("70% CI", "70"), ("95% CI", "95")):
                    for key in ["pass", "fail", "classified"]:
                        _set(f"Characterization {name} {key.capitalize()} % ({ci_label})", row[f"{key}_{suffix}"] * 100)

        except Exception as e:
            warnings.warn(f"Characterization analysis failed: {e}", UserWarning)
    
    def _analyze(self):
        """
        Analyzes the campaign data for practical optimization performance metrics

        Returns:
            None
        """
        if self.objective:
            self._eval_objective()

        # Skip response max profiling for characterization campaigns (not meaningful for characterization)
        # Explicitly determined by the task type
        # Optimization campaigns always get max-profiling
        # Characterization campaigns always skip it regardless of threshold configuration.
        if not self._is_characterization:
            self._profile_max()

        if self._is_mo:
            self._profile_hv()
        else:
            # Remove previous HV-profiling
            self.data = self.data.drop(
                columns=[col for col in self.data.columns
                         if 'Hypervolume' in col or 'Pareto' in col]
            )

        # Characterization analysis
        # Only meaningful for characterization campaigns that also have at least one active threshold.
        # Additionally, only run when the optimizer was fit against the current dataset —
        # otherwise add_data()/set_objective() would populate metrics from a stale model.
        if self._is_characterization:
            if self._has_thresholds():
                if self._last_fit_n_rows == len(self.data):
                    self._analyze_characterization()
                # else: silently skip; the next fit() will repopulate metrics.
            else:
                warnings.warn(
                    "Campaign is set to characterization task, but no thresholds defined in `campaign.target`. "
                    "Set thresholds for `targets` rather than passing threshold values to `suggest`. "
                    "Skipping characterization analysis. ",
                    UserWarning,
                    stacklevel=2,
                )

    def constrain_outputs(self,
                          constraints: Output_Constraint | list[Output_Constraint] | None) -> None:
        """
        Sets optional output constraints for the campaign.
        """
        if constraints is not None:
            if isinstance(constraints, Output_Constraint):
                constraints = [constraints]
            self.output_constraints = constraints

    def clear_output_constraints(self):
        """Clears output constraints"""
        self.output_constraints = None

    def save_state(self) -> dict:
        """
        Saves the state of the Campaign object as a dictionary.

        Returns:
            dict: A dictionary containing the saved state of the Campaign object.
        """
        obj_dict = {}
        obj_dict['X_space'] = self.X_space.save_state()
        obj_dict['optimizer'] = self.optimizer.save_state()
        obj_dict['data'] = self.data.to_dict()
        obj_dict['target'] = [t.save_state() for t in self.target]
        if self.objective:
            obj_dict['objective'] = self.objective.save_state()
        obj_dict['seed'] = self.seed
        obj_dict['task'] = self.task.value

        # Preserve designer state (seed for basic designer and advanced configurations for advanced designer)
        obj_dict['designer'] = self.designer.save_state()


        # Save RNG state for reproducibility
        if hasattr(self, 'rng'):
            obj_dict['rng_state'] = self.rng.save_state()
            obj_dict['owns_rng'] = getattr(self, '_owns_rng', False)
            obj_dict['owns_optimizer'] = getattr(self, '_owns_optimizer', False)
        else:
            obj_dict['rng_state'] = None
            obj_dict['owns_rng'] = False
            obj_dict['owns_optimizer'] = False

        if getattr(self, 'output_constraints', None):
            obj_dict['output_constraints'] = [{'class': const.__class__.__name__,
                                               'state': tensordict_to_dict(const.state_dict())}
                                              for const in self.output_constraints]

        return obj_dict
    
    @classmethod
    def load_state(cls,
                   obj_dict: dict):
        """
        Loads the state of the campaign from a dictionary.

        Args:
            cls (Campaign): The class object.
            obj_dict (dict): A dictionary containing the campaign state.

        Returns:
            Campaign: A new campaign object with the loaded state.
        """
        
        if 'objective' in obj_dict:
            if obj_dict['objective']['name'] == 'Objective_Sequence':
                new_objective = Objective_Sequence.load_state(obj_dict['objective'])
            else:
                obj_class = obj_class_dict[obj_dict['objective']['name']]
                new_objective = obj_class.load_state(obj_dict['objective'])
        else:
            new_objective = None

        # Restore RNG state if saved
        rng = None
        seed=obj_dict['seed']
        if 'rng_state' in obj_dict and obj_dict['rng_state'] is not None:
            rng = RNGManager.load_state(obj_dict['rng_state'])
        else:
            msg = "Loading a legacy campaign save.\nA new RNG manager object will be created to control randomness.\n"
            if seed is None:
                msg += "A random seed will be assigned due to seed is none."
            else:
                msg += f"Seed {seed} will be used to initialize the new RNG manager."
            msg += "\nNote that due to the differences in random states by design, campaign results will be different.\nTo fully recover legacy behavior, set `obsidian.USE_OLD_RNG_CONTROL = True` before loading."
            warnings.warn(msg, UserWarning)

        X_space = ParamSpace.load_state(obj_dict['X_space'])

        # Reconstruct the designer if one was saved.
        # Old state dicts without a 'designer' key fall back to the default ExpDesigner
        designer = None
        designer_state = obj_dict.get('designer')
        if designer_state is not None:
            designer_cls = designer_class_dict.get(designer_state.get('name'))
            if designer_cls is None:
                warnings.warn(
                    f"Unknown designer class '{designer_state.get('name')}' in saved "
                    "state; falling back to default ExpDesigner.",
                    UserWarning,
                )
            else:
                # Reuse the already-reconstructed X_space and the campaign's
                # designer-specific seed to avoid re-parsing the parameter space.
                designer = designer_cls.load_state(designer_state, X_space=X_space, seed=designer_state.get('seed'))

        new_campaign = cls(X_space=X_space,
                           target=[Target.load_state(t_dict) for t_dict in obj_dict['target']],
                           # Default legacy saves to optimization explicitly
                           task=obj_dict.get('task', TaskType.OPTIMIZATION.value),
                           optimizer=BayesianOptimizer.load_state(obj_dict['optimizer']),
                           designer=designer,
                           objective=new_objective,
                           seed=seed,
                           rng=rng)
        new_campaign.data = pd.DataFrame(obj_dict['data'])
        new_campaign.data.index = new_campaign.data.index.astype('int')


        # Handle empty data
        if len(new_campaign.data) > 0 and 'Iteration' in new_campaign.data.columns:
            new_campaign.iter = new_campaign.data['Iteration'].astype('int').max()
        else:
            new_campaign.iter = 0

        # Restore owns_rng flag (gets overwritten during __init__ when rng is passed)
        if 'owns_rng' in obj_dict:
            new_campaign._owns_rng = obj_dict['owns_rng']

        # Restore owns_optimizer flag and sync RNG if Campaign owns the optimizer
        if 'owns_optimizer' in obj_dict:
            new_campaign._owns_optimizer = obj_dict['owns_optimizer']
            # If Campaign created the optimizer originally, restore RNG sharing
            if new_campaign._owns_optimizer and hasattr(new_campaign, 'rng') and hasattr(new_campaign.optimizer, 'rng'):
                new_campaign.optimizer.rng = new_campaign.rng

        if 'output_constraints' in obj_dict:
            all_constraints = [
                const_class_dict[const_dict['class']](new_campaign.target, **const_dict['state'])
                for const_dict in obj_dict['output_constraints']
            ]
            new_campaign.constrain_outputs(all_constraints)

        return new_campaign

    def copy(self):
        """
        Creates a deep copy of the Campaign object.

        A shortcut for saving and then loading the state. The presence of the torch objects prevents a direct deepcopy.

        Returns:
            Campaign: A deep copy of the Campaign object.
        """
        return self.__class__.load_state(self.save_state())
