# Changelog

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
- Major improvements to testing and numerous small bug fixes to improve code robustness
- Code coverage > 90%
- New method for asserting equivalence of state_dicts during serialization

### Modified
- Objective PyTests separated
- Constraint PyTests separated

## [0.7.13]
### Added
- Campaign.Explainer now added to PyTests
- Docstrings and typing to Explainer methods
- Campaign.out property to dynamically capture measured responses "y" or objectives as appropriate
- Campaign.evaluate method to map optimizer.evaluate method
- DNN to PyTests

### Modified
- Fixed SHAP explainer analysis and visualization functions
- Changed SHAP visualization colors to use obsidian branding
- Moved sensitivity method from campaign.analysis to campaign.explainer
- Moved Explainer testing from optimizer pytests to campaign pytests
- Generalized plotting function MOO_results and renamed optim_progress
- Campaign analysis and plotting methods fixed for 
- Greatly increased the number of samples used for DNNPosterior, increasing the stability of posterior predictions

### Removed
- Removed code chunks regarding unused optional inputs to PDP ICE function imported from SHAP GitHub

## [0.7.12]
### Added
- More informative docstrings for optimizer.bayesian, optimizer.predict, to explain choices of surrogate models and aq_funcs

### Modified
- Renamed aq_func hyperparmeter "Xi_f" to "inflate"
- Moved default aq_func choices for single/multi into aq_defaults of acquisition.config
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
- Added DNN surrogate model using EnsemblePosterior. Requries PosteriorList and IndexSampler during optimization
- Added EHVI and NIPV aq_funcs
- Added discarding of NaN X value and Y values
- Target transforms now ignore NaN values
- Quantile prediction to optimizer.predict() and surrogate.predict() to better suit non-GP models nad non-normal distributions

### Modified
- Generalized surrogate_botorch fitting for models which are not GPs
- Generalzed torch.dtype using global variable for import obsidian.utils.TORCH_DTYPE
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
- Updated "features objectives" to operate on real space instead of scaled space; virtually no speed difference and tensor grads not needed
- Improvements to campaign object and explainer object features
- Switched all GenericMC and GenericMOMC objectives to custom objectives to simplify SOO vs MOO, and enable serialization
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
- Removed autosummary from package and subpackage-level autodoc. Automated only at module level with templates
- Split subpackages into regular and "shallow" templates, to expose different meaningful levels on the toctree
- Updated sphinx theme with more preferred navigation features

### Removed
- Unused dependencies from pyproject.toml

## [0.7.1]
### Modified
- Evaluate now does not calculate aq_func by default, to speed up evaluation of objective
- Bug fixes for optimizer.evaluate() under fringe tests
- Updated license to GPLv3 on Readme
- Removed Merck references and updated branding

## [0.7.0]
### Added
- optimizer.evaluate() to handle y_predict, f_predict, o_predict, and a_predict independently of optimize.suggest()

### Modified
- Regardless of q_batch, evaluate aq functions on each sample individually, then as a joint sample
- Made sure that X_baseline accepted X_pending
- Made sure that ref_point and f_best in aq_hps are now based on objectives, not raw responses

### Removed
- Removed o_dim and optim_type from optimizer.attrs to better support stateless operation
- No longer calculate hypervolume or pareto front on raw responses; only after considering objectives
- Removed y_pareto from optimizer.attrs
- Temporarily removed campaign._analyze due to bugs with _hv

## [0.6.11]
### Added
- Added plotting module to test coverage
- Reloading/fitting f_train in optimizer.load_state
- Added default_campaign.json to test directory, for faster testing using pre-fit objects
- Added matplotlib plotting library
- Can now provide X_pending to suggest, so that users can manually do iterative suggestions

### Modified
- Various plotting bug fixes
- Ensure that paramspace encode/unit_map return double dtypes

## [0.6.10]
### Added
- Added __getitem__ to ParamSpace to enable X_space[idx]

### Modified
- Overhaul of all encode/decode and map/demap functions using a decorator to handle robust typing
- Bug fix for SHAP_explain resulting from object dtypes with new encode functions

## [0.6.9]
### Added
- Added notebook tutorials to docs
- __repr__ method for Target

### Modified
- Bug fixes in factor_plot with X_ref provided
- Fixed bug where categorical encoding wouldn't work if all categories were numerical strings
- Changed categorical OH-encode separator from _ to ^ and protected against usage in X_names

## [0.6.8]
### Modified
- Dash app updates with module refactoring
- Minor refactoring of param_space discrete handling
- Replaced assert statements with appropriate Python base exceptions, added custom obsidian exceptions as necessary
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
- Removed parameter.type and replaced with isinstance checking and __class__.__name__ loading

## [0.6.5]
### Added
- Custom exception handling
- Composite objectives
- Utopian point support for SOO and MOO
- Bounded target support for SOO and MOO
- Bounding for only selected targets in multi-output scenarios

### Modified
- Completed refactored weighted MOO to separate out component parts of scalarization, utopian distance, norming, and bounding
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
- Simplex sampling of weights for weightedMOO; if weights aren't provided, even weights are used

## [0.6.3]
### Modified
- Moved "explain" functionality to base optimizer
- Added backup exceptions to catch fit_gpytorch
- Enabled multi-output objectives to expand single-response models (e.g. using X features)
- Enabled single-output objectives to condense multi-response models (e.g. scalarization)
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
- Added hypervolume and pareto calculations (incl. pf_distance) to base optimizer
- Fixed bug with target_transforms sharing hyperparameters because of bad initialization

## [0.6.0]
### Added
- Task parameters and multi-task learning
- Added "index" objective to select tasks for optimization in multi-task learning
- Fixed bugs related to torch dtype mismatches

## [0.5.7]
### Modified
- Fixed cat_dims specification for surrogate models so that they do not include ordinal params
- Overhauled f_transform approach to avoid scikit-learn and be more customizable
- Enforced abstractmethods on various classes

## [0.5.7]
### Modified
- Fixed cat_dims specification for surrogate models so that they do not include ordinal params
- Overhauled f_transform approach to avoid scikit-learn and be more customizable
- Enforced abstractmethods on various classes

## [0.5.6]
### Modified
- Added utopian point subtraction to all scalarization methods, made optional also
- Moved PI_bounded weightedMOO to its own objective called "boundedMOO"
- Fixed error where default hyperparameters were being written back to objects outside of the optimizer
- Fixed bugs with new parameter types in experiment design modules
- Added pytest parametrization to improve scope of tests
- Added pytest attributes (slow, fast) to manage speed

## [0.5.5]
### Modified
- Fixed bug with _fixed_features generation when fixed_var is specified
- Updated all MOO custom aq functions to match current BoTorch patterns

### Removed
- PI_bounded acquisition function, due to various issues. MOO_weighted with PI_bounded+weights scalarization does work

## [0.5.4]
### Modified
- Added Param_Discrete_Numeric class (parent Param_Discrete)
- Added Param_Observational subclass to parent Continuous class which can be used for fitting but avoid optimization

## [0.5.3]
### Modified
- Implemented checks and validation to enforce the order of X, y, and targets across ParamSpace, Optimizer, Surrogate as appropriate
    - Note: Only Optimizer can handle extraneous or re-ordered  columns, but they will be processed before passing to Surrogate

## [0.5.2]
### Modified
- Fixed input/output constraints
- Added de/transformation to output constraints (applied to target samples)
- Added de/transformation to input constraints (applied to coefficients and RHS)
- Implemented a de/transform map for ParamSpace in order to handle constraints in the encoded input space
- Implemented a non-linear input constraint which keeps the range of one dimension in a joint optimization < 1% of the max-min

## [0.5.1]
### Added
- Constrained multi-objective example notebook
- Allowed optimizer.suggest() on a subset of fit responses
- Enabled optimizer.maximize() for multi-response models based on the above
- Added custom multi-output objective class
- Fixed constraints specification, as we were using output constraints. Added input constraints to TODO
- Added parameter name to lb/ub labels in optimizer.predict() to avoid index issues

### Modified
- Custom constraints are now specified as a constructer, so that parameters can be added and a callable is returned

## [0.5.0]
### Added
- New object-oriented design for several classes: `Campaign` `Parameter` `ParamSpace` `Target` `Objective` and `Constraint`

