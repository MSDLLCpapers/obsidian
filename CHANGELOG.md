# Changelog

## [1.0.0]

### Added

- **Advanced Experimental Designer** (`AdvExpDesigner`) — a new design engine
  that drops into a `Campaign` as `designer=`:
  - Continuous factors with configurable spacing
    (linear/geometric/logarithmic/explicit levels) and non-uniform **sampling
    bias**, drawn via stratified LHS
  - **Conditional subparameters** — hierarchical/nested factors (e.g. buffer
    type → pH range) with per-level frequency weights
  - **D-optimal design optimization** with parallel candidate search, seven
    quality metrics (D-/A-optimality, condition number, space-filling,
    continuous/categorical/mixed correlation), and category-correlation
    minimization for orthogonal categoricals
  - `extend_design()` for design augmentation / sequential DOE
  - Built-in visualization: histograms, mixed-correlation heatmap,
    quality-evolution bars, and PCA / MDS / UMAP embeddings
- **Characterization tasks** — a new campaign task type
  (`task="characterization"`) for level-set estimation: mapping which regions of
  the parameter space *pass* a threshold, rather than maximizing a response:
  - `threshold` attribute on `Target`; task-aware acquisition defaults on
    `Campaign`/`BayesianOptimizer`
  - Boundary-seeking "straddle" acquisitions: `STR`, `RANDSTR` (single-target)
    and `MSTR`, `JAREX` (multi-target)
  - `CharacterizationEvaluator` — posterior pass/fail classification, a 4-level
    confidence ladder (mean/70%/95% CI), joint multi-target metrics, sample-size
    planning, ground-truth scoring (confusion matrix / Jaccard), and
    largest-feasible-hypercube search
  - `plot_2d_response_map` (pass/fail, confidence, and continuous modes)
- **Unified RNG control** (`RNGManager`) — coordinates numpy, torch, and Python
  `random` behind a single seed, serialized into campaign/optimizer state so a
  saved-and-loaded campaign reproduces exactly. One `RNGManager` can be shared
  across a campaign, optimizer, and designer; `obsidian.USE_OLD_RNG_CONTROL`
  restores legacy global-seeding behavior.
- **Custom acquisition function registry** — acquisition functions are now
  defined as rich registry entries (implementation + hyperparameter parser +
  modalities + task types + constraint support). New
  `acquisition_function_register(...)` lets users plug in their own;
  `RandomSampling` (`RS`) is now a real implementation.
- **Classical response-surface & n-level DOE** — `central_composite_DOE` (CCD,
  with rotatable/faced/custom alpha and unit-cube inscribing) and
  `factorial_DOE_n_level` (3+ level factorials); CCD exposed via
  `ExpDesigner.initialize(method="CCD")`.
- **Restartable designers** — both `ExpDesigner` and `AdvExpDesigner` gain
  `save_state()`/`load_state()`, persisted with the campaign.
- Multi-response plotting: `response_id`/`response_ids` on
  parity/factor/surface/progress plots, reference-point overlay and custom
  objective on `factor_plot`, confidence bands and raw-data overlay on
  `surface_plot`, and `X_suggest` overlay on `optim_progress`.
- Support for **Python 3.10–3.12**.

### Modified

- **More robust surrogate fitting** — multi-restart GP fitting with best-of-N
  selection is now the default (`max_attempts=5`); `fit()` exposes
  optimizer/multi-start controls and prior-sampling initialization.
- Optimizers can be saved/loaded **unfitted** (config + RNG only); `save_state`
  no longer requires a fitted model.
- `suggest()` gains `optim_options` (passed through to BoTorch/scipy) and
  per-call `manual_seed`; `fit()` gains `fit_options`; `maximize()` accepts an
  `acquisition` argument; Random Search now respects fixed discrete features.
- `evaluate()` now emits a standardized-prediction column per target.

### Fixed

- **Logit transform** — fit now runs only on training data (not on every call,
  which broke for single-row inputs); degenerate/constant/all-NaN data is
  handled gracefully instead of producing NaN/inf, out-of-`[0,1]` inputs raise a
  clear error, and unfitted scalers raise `UnfitError`.

### Removed

- **Dash web app** (`obsidian/dash`, `app.py`) and the `[app]` extra — Obsidian
  is now a library package.

## [0.8.6]

### Added

- Improved methods for fitting PyTorch surrogates, including auto-stopping by
  parameter value norm

### Modified

- Greatly reduced the number of samples for DNN posterior, speeding up
  optimization
- Stabilized the mean estimate of ensemble surrogates by avoiding resampling
- Disabled root caching for ensemble surrogates during optimization
- Increased the maximum length of a category name to 32 characters
- Bug fix for incorrect symmetry in correlation calculation of calc_ofat_ranges
- OFAT range calcs and plots now respect minimum targets and not just maximum

## [0.8.5]

### Added

- More optional outputs for verbose settings
- Parameters in ParamSpace can also be indexed by name
- Parameters now have search_space property, to modify the optimizer search
  space from the full space
- Continuous parameters have search_min/search_max; Discrete parameters have
  search_categories
- Constraints are now defined by Constraint class
- Input constraints can now be included in ParamSpace, and serialized from there
- Output constraints can now be included in Campaign, and serialized from there
- New interface class IParamSpace to address circular import issues between
  ParamSpace and Constraint

### Modified

- Optimizer and Campaign X_space attributes are now assigned using setter
- Optimizer.maximize() appropriately recognizes fixed_var argument

### Removed

- Torch device references and options (GPU compatibility may be re-added)

## [0.8.4]

### Added

- Campaign X_best method
- Optimizer X_best_f attribute(s)
- Sequence of colors "color_list" to branding
- Informative hoverdata for MDS plot
- Created Product_Objective and Divide_Objective

### Modified

- Switched all usages of X_ref = X_space.mean() to optimizer.X_best_f
- Refactored mpl "visualize_inputs" as plotly "visualize_inputs" for better
  interactivity
- Text formatting for some plotly hoverdata

## [0.8.3]

### Added

- Default values for NParEGO scalarization_weights
- SHAP PDP ICE plots now work with categorical values
- Added scikit-learn to dependencies, for MDS
- Added MDS plot

### Modified

- SHAP PDP ICE plots must now have color and x-axis indices that are distinct

## [0.8.2]

### Added

- Project metadata properly captured on PyPI based on changes in pyproject.toml

## [0.8.1]

### Modified

- Fixed infobar on dash app
- Better handling of X_space on dash app
- Bug fixes for optim_progress
- Improved color and axes of parity_plot

## [0.8.0]

### Added

- Major improvements to testing and numerous small bug fixes to improve code
  robustness
- Code coverage > 90%
- New method for asserting equivalence of state_dicts during serialization

### Modified

- Objective PyTests separated
- Constraint PyTests separated

## [0.7.13]

### Added

- Campaign.Explainer now added to PyTests
- Docstrings and typing to Explainer methods
- Campaign.out property to dynamically capture measured responses "y" or
  objectives as appropriate
- Campaign.evaluate method to map optimizer.evaluate method
- DNN to PyTests

### Modified

- Fixed SHAP explainer analysis and visualization functions
- Changed SHAP visualization colors to use obsidian branding
- Moved sensitivity method from campaign.analysis to campaign.explainer
- Moved Explainer testing from optimizer pytests to campaign pytests
- Generalized plotting function MOO_results and renamed optim_progress
- Campaign analysis and plotting methods fixed for multi-response campaigns
- Greatly increased the number of samples used for DNNPosterior, increasing the
  stability of posterior predictions

### Removed

- Removed code chunks regarding unused optional inputs to PDP ICE function
  imported from SHAP GitHub

## [0.7.12]

### Added

- More informative docstrings for optimizer.bayesian, optimizer.predict, to
  explain choices of surrogate models and aq_funcs

### Modified

- Renamed aq_func hyperparameter "Xi_f" to "inflate"
- Moved default aq_func choices for single/multi into aq_defaults of
  acquisition.config
- Fixed and improved campaign analysis methods

## [0.7.11]

### Added

- Documentation improvements

### Modified

- Naming conventions for modules (config, utils, base)
- Import order convention for modules

## [0.7.10]

### Added

- First (working) release on PyPI

## [0.7.6]

### Added

- Added DNN surrogate model using EnsemblePosterior. Requires PosteriorList and
  IndexSampler during optimization
- Added EHVI and NIPV aq_funcs
- Added discarding of NaN X value and Y values
- Target transforms now ignore NaN values
- Quantile prediction to optimizer.predict() and surrogate.predict() to better
  suit non-GP models and non-normal distributions

### Modified

- Generalized surrogate_botorch fitting for models which are not GPs
- Generalized torch.dtype using global variable for import
  obsidian.utils.TORCH_DTYPE
- Improved some tensor.repeat() using tensor.repeat_interleave()
- Switched EI, NEI, and qNEHVI to Log-EI versions based on BoTorch warning
- Dropped "q" from qNEHVI / qNParEGO names
- Simplified surrogate model loading in BO_optimizer
- Changed name of surrogate.args/kwargs to surrogate.hps
- Corrected typing in target transforms

### Removed

- Thompson sampling aq_func

## [0.7.5]

### Modified

- Removed explainer from campaign attributes
- Updated "features objectives" to operate on real space instead of scaled
  space; virtually no speed difference and tensor grads not needed
- Improvements to campaign object and explainer object features
- Switched all GenericMC and GenericMOMC objectives to custom objectives to
  simplify SOO vs MOO, and enable serialization
- Added set_objective to campaign, and objective serialization
- Bug fix for surrogate.save_state() with train_Y
- Marked GPflat + categorical space test as an expected fail in pytest
- Minor bug fixes and test improvements to improve coverage

## [0.7.4]

### Modified

- Bug fix for qMean, made sure objective is used
- Set up basic validation tests for parameters package

## [0.7.3]

### Modified

- Moved plotting dependencies to main build
- Updated Contributing section
- Minor modifications to main Readme
- Moved jupyterlab from docs to dev group

### Removed

- docs/readme.md; moved contents to CONTRIBUTING

## [0.7.2]

### Added

- Added imported members to sphinx-autodoc results using __all__ at module level
- Included readme on main docs page, removed redundant link

### Modified

- Various cosmetic changes to autodoc and autosummary options
- Shortened object names on TOC tree for readability
- Removed autosummary from package and subpackage-level autodoc. Automated only
  at module level with templates
- Split subpackages into regular and "shallow" templates, to expose different
  meaningful levels on the toctree
- Updated sphinx theme with more preferred navigation features

### Removed

- Unused dependencies from pyproject.toml

## [0.7.1]

### Modified

- Evaluate now does not calculate aq_func by default, to speed up evaluation of
  objective
- Bug fixes for optimizer.evaluate() under fringe tests
- Updated license to GPLv3 on Readme
- Removed Merck references and updated branding

## [0.7.0]

### Added

- optimizer.evaluate() to handle y_predict, f_predict, o_predict, and a_predict
  independently of optimize.suggest()

### Modified

- Regardless of q_batch, evaluate aq functions on each sample individually, then
  as a joint sample
- Made sure that X_baseline accepted X_pending
- Made sure that ref_point and f_best in aq_hps are now based on objectives, not
  raw responses

### Removed

- Removed o_dim and optim_type from optimizer.attrs to better support stateless
  operation
- No longer calculate hypervolume or pareto front on raw responses; only after
  considering objectives
- Removed y_pareto from optimizer.attrs
- Temporarily removed campaign._analyze due to bugs with_hv

## [0.6.11]

### Added

- Added plotting module to test coverage
- Reloading/fitting f_train in optimizer.load_state
- Added default_campaign.json to test directory, for faster testing using
  pre-fit objects
- Added matplotlib plotting library
- Can now provide X_pending to suggest, so that users can manually do iterative
  suggestions

### Modified

- Various plotting bug fixes
- Ensure that paramspace encode/unit_map return double dtypes

## [0.6.10]

### Added

- Added __getitem__ to ParamSpace to enable X_space[idx]

### Modified

- Overhaul of all encode/decode and map/demap functions using a decorator to
  handle robust typing
- Bug fix for SHAP_explain resulting from object dtypes with new encode
  functions

## [0.6.9]

### Added

- Added notebook tutorials to docs
- __repr__ method for Target

### Modified

- Bug fixes in factor_plot with X_ref provided
- Fixed bug where categorical encoding wouldn't work if all categories were
  numerical strings
- Changed categorical OH-encode separator from _to ^ and protected against usage
  in X_names

## [0.6.8]

### Modified

- Dash app updates with module refactoring
- Minor refactoring of param_space discrete handling
- Replaced assert statements with appropriate Python base exceptions, added
  custom obsidian exceptions as necessary
- Moved benchmark module to obsidian.experiment

### Removed

- Removed deprecated examples

## [0.6.7]

### Added

- Dev capabilities for document generation using sphinx
- Module docstrings

## [0.6.6]

### Modified

- Overhaul of class and method docstrings

### Removed

- Removed parameter.type and replaced with isinstance checking and
  __class__.__name__ loading

## [0.6.5]

### Added

- Custom exception handling
- Composite objectives
- Utopian point support for SOO and MOO
- Bounded target support for SOO and MOO
- Bounding for only selected targets in multi-output scenarios

### Modified

- Completed refactored weighted MOO to separate out component parts of
  scalarization, utopian distance, norming, and bounding
- Greatly simplified the structure of aq_kwargs based on the above

### Removed

- weightedMOO acquisition function
- Dynamic reference point utility

## [0.6.4]

### Added

- More custom objectives

### Modified

- Switched default single-objective aq from EI to NEI

### Removed

- Simplex sampling of weights for weightedMOO; if weights aren't provided, even
  weights are used

## [0.6.3]

### Modified

- Moved "explain" functionality to base optimizer
- Added backup exceptions to catch fit_gpytorch
- Enabled multi-output objectives to expand single-response models (e.g. using X
  features)
- Enabled single-output objectives to condense multi-response models (e.g.
  scalarization)
- Updated demo notebooks
- Fixed error with calling campaign._profile_max after every fit

### Removed

- Redundant objective formulations
- Removed GPFlex model, which is now redundant with multi-output objective

## [0.6.2]

### Modified

- Updated all plotly plotting methods for new optimizer methods
- Updated typing and enabled multi-objective on custom aq functions

## [0.6.1]

### Modified

- Added new features to campaign object
- Moved "seed" kwarg to ExpDesigner.__init__, consistent with BayesianOptimizer
- Added hypervolume and pareto calculations (incl. pf_distance) to base
  optimizer
- Fixed bug with target_transforms sharing hyperparameters because of bad
  initialization

## [0.6.0]

### Added

- Task parameters and multi-task learning
- Added "index" objective to select tasks for optimization in multi-task
  learning
- Fixed bugs related to torch dtype mismatches

## [0.5.7]

### Modified

- Fixed cat_dims specification for surrogate models so that they do not include
  ordinal params
- Overhauled f_transform approach to avoid scikit-learn and be more customizable
- Enforced abstractmethods on various classes

## [0.5.7]

### Modified

- Fixed cat_dims specification for surrogate models so that they do not include
  ordinal params
- Overhauled f_transform approach to avoid scikit-learn and be more customizable
- Enforced abstractmethods on various classes

## [0.5.6]

### Modified

- Added utopian point subtraction to all scalarization methods, made optional
  also
- Moved PI_bounded weightedMOO to its own objective called "boundedMOO"
- Fixed error where default hyperparameters were being written back to objects
  outside of the optimizer
- Fixed bugs with new parameter types in experiment design modules
- Added pytest parametrization to improve scope of tests
- Added pytest attributes (slow, fast) to manage speed

## [0.5.5]

### Modified

- Fixed bug with _fixed_features generation when fixed_var is specified
- Updated all MOO custom aq functions to match current BoTorch patterns

### Removed

- PI_bounded acquisition function, due to various issues. MOO_weighted with
  PI_bounded+weights scalarization does work

## [0.5.4]

### Modified

- Added Param_Discrete_Numeric class (parent Param_Discrete)
- Added Param_Observational subclass to parent Continuous class which can be
  used for fitting but avoid optimization

## [0.5.3]

### Modified

- Implemented checks and validation to enforce the order of X, y, and targets
  across ParamSpace, Optimizer, Surrogate as appropriate
  - Note: Only Optimizer can handle extraneous or re-ordered  columns, but they
    will be processed before passing to Surrogate

## [0.5.2]

### Modified

- Fixed input/output constraints
- Added de/transformation to output constraints (applied to target samples)
- Added de/transformation to input constraints (applied to coefficients and RHS)
- Implemented a de/transform map for ParamSpace in order to handle constraints
  in the encoded input space
- Implemented a non-linear input constraint which keeps the range of one
  dimension in a joint optimization < 1% of the max-min

## [0.5.1]

### Added

- Constrained multi-objective example notebook
- Allowed optimizer.suggest() on a subset of fit responses
- Enabled optimizer.maximize() for multi-response models based on the above
- Added custom multi-output objective class
- Fixed constraints specification, as we were using output constraints. Added
  input constraints to TODO
- Added parameter name to lb/ub labels in optimizer.predict() to avoid index
  issues

### Modified

- Custom constraints are now specified as a constructor, so that parameters can
  be added and a callable is returned

## [0.5.0]

### Added

- New object-oriented design for several classes: `Campaign` `Parameter`
  `ParamSpace` `Target` `Objective` and `Constraint`
