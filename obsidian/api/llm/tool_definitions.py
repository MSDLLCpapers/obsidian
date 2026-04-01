"""
OpenAI function calling tool definitions for Obsidian REST API.

This module contains function definitions for all Obsidian API endpoints
in OpenAI function calling format. These definitions enable LLM agents to
interact with the Obsidian optimization API programmatically.
"""

# Tool definitions in OpenAI function calling format
# Each definition maps to one REST API endpoint

TOOL_DEFINITIONS = {
    # ============================================================================
    # Session Management Functions (4)
    # ============================================================================
    "create_optimization_session": {
        "name": "create_optimization_session",
        "description": (
            """Create a new Bayesian optimization session.

This initializes a new optimization campaign with specified parameters and objectives.
The session will be in 'configured' status after creation.

Use when:
- Starting a new optimization campaign
- Defining parameter space and optimization targets
- Need to track experiments systematically

Returns a unique session_id to use in subsequent operations.

Limitations:
- Must provide at least one parameter and one target
- Categorical/ordinal parameters need at least 2 categories
- Session persists in ~/.obsidian/sessions/ until deleted"""
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "name": {
                    "type": "string",
                    "description": "Human-readable name for this session (optional, auto-generated if not provided)",
                },
                "parameters": {
                    "type": "array",
                    "description": "List of optimization parameters (must have at least one)",
                    "items": {
                        "type": "object",
                        "properties": {
                            "type": {
                                "type": "string",
                                "enum": ["continuous", "categorical", "ordinal", "observational"],
                                "description": "Parameter type",
                            },
                            "name": {"type": "string", "description": "Parameter name"},
                            "min": {
                                "type": "number",
                                "description": "Minimum value (for continuous/observational only)",
                            },
                            "max": {
                                "type": "number",
                                "description": "Maximum value (for continuous/observational only)",
                            },
                            "categories": {
                                "type": "array",
                                "items": {"type": "string"},
                                "description": "Categories (for categorical/ordinal, min 2 required)",
                            },
                            "search_min": {
                                "type": "number",
                                "description": "Search space minimum (continuous only, optional)",
                            },
                            "search_max": {
                                "type": "number",
                                "description": "Search space maximum (continuous only, optional)",
                            },
                            "search_categories": {
                                "type": "array",
                                "items": {"type": "string"},
                                "description": "Search space categories (categorical/ordinal only, optional)",
                            },
                        },
                        "required": ["type", "name"],
                    },
                },
                "targets": {
                    "type": "array",
                    "description": "List of optimization targets/objectives (must have at least one)",
                    "items": {
                        "type": "object",
                        "properties": {
                            "name": {"type": "string", "description": "Target/response variable name"},
                            "aim": {
                                "type": "string",
                                "enum": ["min", "max"],
                                "description": "Optimization direction (default: 'max')",
                            },
                            "f_transform": {
                                "type": "string",
                                "description": (
                                    "Transformation function ('Standard', 'Identity', or null, default: 'Standard')"
                                ),
                            },
                        },
                        "required": ["name"],
                    },
                },
                "seed": {"type": "integer", "description": "Random seed for reproducibility (optional)"},
                "surrogate": {
                    "type": ["string", "array", "object"],
                    "description": (
                        "Surrogate model type(s). Options:\n"
                        "- String: 'GP' (Gaussian Process, default), 'DNN' (Deep Neural Network), 'DKL' (Deep Kernel Learning), "
                        "'MixedGP' (mixed continuous/categorical), 'GPflat' (GP with flat prior), 'GPprior' (GP with informative prior), 'MTGP' (Multi-task GP)\n"
                        "- Object for hyperparameters: {'DNN': {'p_dropout': 0.2, 'h_width': 32, 'h_layers': 2}}\n"
                        "- Array for multi-output: ['GP', {'DNN': {...}}]\n"
                        "DNN hyperparameters: p_dropout (float, 0-1), h_width (int, hidden layer width), h_layers (int, number of hidden layers)"
                    ),
                    "default": "GP",
                },
            },
            "required": ["parameters", "targets"],
        },
    },
    "list_optimization_sessions": {
        "name": "list_optimization_sessions",
        "description": (
            """List all optimization sessions.

Returns metadata for all sessions, optionally filtered by status.

Use when:
- Discovering available sessions
- Checking session states
- Managing multiple optimization campaigns

Statuses: 'configured', 'initialized', 'fitted', 'suggesting', 'evaluating', 'completed', 'error'"""
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "status_filter": {
                    "type": "string",
                    "description": "Optional filter by status ('configured', 'initialized', 'fitted', etc.)",
                }
            },
        },
    },
    "get_session_details": {
        "name": "get_session_details",
        "description": (
            """Get detailed information about a session.

Returns comprehensive session metadata including parameter names, target names,
experiment count, and current status.

Use when:
- Inspecting session configuration
- Checking parameter/target definitions
- Verifying session state before operations"""
        ),
        "parameters": {
            "type": "object",
            "properties": {"session_id": {"type": "string", "description": "Unique identifier of the session"}},
            "required": ["session_id"],
        },
    },
    "delete_optimization_session": {
        "name": "delete_optimization_session",
        "description": (
            """Delete an optimization session.

Permanently removes the session and all associated data from disk.
This operation cannot be undone.

Use when:
- Cleaning up completed sessions
- Removing test sessions
- Managing storage space

Warning: This deletes all experiment data, model state, and history."""
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "session_id": {"type": "string", "description": "Unique identifier of the session to delete"}
            },
            "required": ["session_id"],
        },
    },
    # ============================================================================
    # Workflow Operation Functions (6)
    # ============================================================================
    "initialize_experiments": {
        "name": "initialize_experiments",
        "description": (
            """Generate initial experiment design.

Creates an initial set of experiments using design of experiments methods (LHS, Random, etc.).
This is typically the first step after creating a session.

Use when:
- Starting a new optimization campaign
- Need initial experiments to explore parameter space
- Before collecting any data

Returns a list of parameter combinations to test experimentally.

Methods: 'LHS' (Latin Hypercube Sampling, default), 'Random'
Typical m_initial: 10-20 experiments"""
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "session_id": {"type": "string", "description": "Unique identifier of the session"},
                "m_initial": {
                    "type": "integer",
                    "description": "Number of initial experiments to generate (default: 10, min: 1)",
                    "default": 10,
                    "minimum": 1,
                },
                "method": {
                    "type": "string",
                    "description": "Design method ('LHS' or 'Random', default: 'LHS')",
                    "default": "LHS",
                },
                "seed": {"type": "integer", "description": "Random seed for reproducibility (optional)"},
            },
            "required": ["session_id"],
        },
    },
    "sample_parameter_space": {
        "name": "sample_parameter_space",
        "description": (
            """Sample random points from parameter space without initializing session.

This is a stateless operation that generates points according to the specified
sampling method WITHOUT changing the session status or storing the points.

Use when:
- Exploring parameter space bounds and structure
- Generating test points for visualization or analysis
- Understanding the design space before committing to initialization
- Creating custom experimental designs for evaluation
- Agent wants to explore surfaces without building full arrays

Key differences from initialize_experiments:
- Does NOT change session status to 'initialized'
- Does NOT store points in the session
- Does NOT affect subsequent workflow operations
- Purely exploratory - no side effects

Methods available:
- 'LHS': Latin Hypercube Sampling (space-filling design)
- 'Random': Uniform random sampling
- 'Sobol': Sobol sequence (quasi-random low-discrepancy)

Returns a DataFrame of sampled points that can be:
- Passed to evaluate_predictions to explore surrogate surfaces
- Used for visualization
- Inspected to understand parameter ranges
- Modified before actual initialization

Typical use: Generate 50-100 points to explore fitted surrogate model."""
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "session_id": {"type": "string", "description": "Unique identifier of the session"},
                "n_points": {
                    "type": "integer",
                    "description": "Number of points to sample (default: 10, min: 1)",
                    "default": 10,
                    "minimum": 1,
                },
                "method": {
                    "type": "string",
                    "description": "Sampling method ('LHS', 'Random', 'Sobol', default: 'LHS')",
                    "default": "LHS",
                    "enum": ["LHS", "Random", "Sobol"],
                },
                "seed": {"type": "integer", "description": "Random seed for reproducibility (optional)"},
            },
            "required": ["session_id"],
        },
    },
    "add_experimental_data": {
        "name": "add_experimental_data",
        "description": (
            """Add experimental results to the session.

Uploads measured response values for experiments. Each data point must include
all parameter values and at least one target value.

Use when:
- Have experimental results to add
- After running suggested or initial experiments
- Incrementally building the dataset

Data format: List of dicts, each containing parameter names and target names as keys.
Example: [{"Temperature": 25, "Pressure": 5, "Yield": 85.2}]"""
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "session_id": {"type": "string", "description": "Unique identifier of the session"},
                "data": {
                    "type": "array",
                    "description": "List of experimental results (must not be empty)",
                    "items": {"type": "object", "description": "Dictionary with parameter and target values"},
                    "minItems": 1,
                },
            },
            "required": ["session_id", "data"],
        },
    },
    "fit_surrogate_model": {
        "name": "fit_surrogate_model",
        "description": (
            """Fit surrogate model to experimental data.

Trains Gaussian Process (GP) or Deep Neural Network (DNN) models on collected data.
This step is required before generating suggestions or making predictions.

Use when:
- Have added experimental data
- Before requesting suggestions
- After collecting a new batch of experiments

The model learns relationships between parameters and responses, enabling
prediction and optimization.

Typical workflow: initialize → add_data → fit → suggest → (repeat)"""
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "session_id": {"type": "string", "description": "Unique identifier of the session"},
                "fit_options": {
                    "type": "object",
                    "description": "Optional fitting configuration (advanced use)",
                    "default": {},
                },
                "verbose": {
                    "type": "integer",
                    "description": "Optimizer verbosity level (0=none, 1=summary, 2=detailed, 3=debug). Default 3 for LLM agents.",
                    "minimum": 0,
                    "maximum": 3,
                    "default": 3,
                },
            },
            "required": ["session_id"],
        },
    },
    "suggest_next_experiments": {
        "name": "suggest_next_experiments",
        "description": (
            """Generate next experiment suggestions using Bayesian optimization.

Uses the fitted surrogate model to suggest promising experiments by optimizing
the acquisition function (e.g., Expected Improvement, Upper Confidence Bound).
This is the core of the Bayesian optimization loop.

Use when:
- Model is fitted (check with get_model_diagnostics)
- Want to generate next batch of experiments
- Continuing an optimization loop

Acquisition functions (single-objective):
- 'NEI': Noisy Expected Improvement (default, robust)
- 'EI': Expected Improvement (classic)
- 'UCB': Upper Confidence Bound (exploration-exploitation balance, use beta hyperparameter)
- 'Mean': Posterior mean (exploitation only)
- 'SR': Simple Regret
- 'NIPV': Negative Integrated Posterior Variance (characterization)

Acquisition functions (multi-objective):
- 'NEHVI': Noisy Expected Hypervolume Improvement (recommended, requires ref_point)
- 'EHVI': Expected Hypervolume Improvement (requires ref_point)
- 'NParEGO': Noisy ParEGO (scalarization-based)

Universal acquisition functions:
- 'RS': Random Sampling (baseline)
- 'SF': Space Filling
- 'Mean': Posterior Mean

Advanced options:
- fixed_var: Lock specific parameters at constant values
- optim_sequential: Control batch optimization strategy
- X_pending: Account for experiments already queued/running

Returns suggested parameter values and optional evaluation metrics.

Limitations:
- Requires at least one model fit
- Suggestions are stochastic unless seed is fixed via session creation"""
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "session_id": {"type": "string", "description": "Unique identifier of the session"},
                "m_batch": {
                    "type": "integer",
                    "description": "Number of experiments to suggest (default: 1, min: 1)",
                    "default": 1,
                    "minimum": 1,
                },
                "acquisition": {
                    "type": "array",
                    "description": (
                        "Acquisition function(s) to use. Options:\n"
                        "- String form (uses defaults): 'NEI', 'EI', 'UCB', 'EHVI', 'NEHVI', 'NParEGO', 'Mean', 'RS', 'SF'\n"
                        "- Object form for hyperparameters: {'UCB': {'beta': 2.0}}, {'NEHVI': {'ref_point': [-100, -50]}}\n"
                        "- Mixed: ['NEI', {'UCB': {'beta': 1.5}}, 'SF']\n"
                        "Common hyperparameters:\n"
                        "  - UCB: beta (float, exploration weight, >1 = more exploration, default: 1.0)\n"
                        "  - EHVI/NEHVI: ref_point (list[float], reference point for hypervolume)\n"
                        "  - NParEGO: scalarization_weights (list[float], scalarization weights)\n"
                        "  - EI/PI: inflate (float, inflation factor, default: 0)\n"
                        "Default: ['NEI']"
                    ),
                    "items": {
                        "oneOf": [
                            {"type": "string"},
                            {"type": "object"}
                        ]
                    },
                    "default": ["NEI"],
                },
                "optim_samples": {
                    "type": "integer",
                    "description": "Optimization samples (higher = better quality, slower, default: 512)",
                    "minimum": 64,
                },
                "optim_restarts": {
                    "type": "integer",
                    "description": "Optimization restarts (higher = better quality, slower, default: 10)",
                    "minimum": 1,
                },
                "manual_seed": {
                    "type": "integer",
                    "description": (
                        "Manual seed for exploration and reproducibility (optional, default: None uses campaign RNG)"
                    ),
                },
                "verbose": {
                    "type": "integer",
                    "description": "Optimizer verbosity level (0=none, 1=summary, 2=detailed, 3=debug). Default 3 for LLM agents.",
                    "minimum": 0,
                    "maximum": 3,
                    "default": 3,
                },
                "fixed_var": {
                    "type": "object",
                    "description": (
                        "Fix specific parameters during optimization (optional). "
                        "Only other parameters will be varied in suggestions. "
                        "Example: {'Temperature': 25.0} locks Temperature at 25.0. "
                        "Useful for sensitivity analysis or when certain parameters must remain constant."
                    ),
                },
                "optim_sequential": {
                    "type": "boolean",
                    "description": (
                        "Sequential (True) vs joint (False) batch optimization. "
                        "True (default): Optimize suggestions one-by-one (faster, good for most cases). "
                        "False: Jointly optimize all batch suggestions together (slower but better batch designs, recommended for m_batch > 1)."
                    ),
                    "default": True,
                },
                "X_pending": {
                    "type": "array",
                    "description": (
                        "Experiments already queued or running (optional). "
                        "Acquisition function will account for these pending points to avoid redundant suggestions. "
                        "Format: list of parameter dicts, e.g., [{'Temperature': 50, 'Pressure': 5}, {'Temperature': 75, 'Pressure': 8}]. "
                        "Useful for parallel experimentation workflows."
                    ),
                    "items": {"type": "object", "description": "Dictionary with parameter names as keys"},
                },
            },
            "required": ["session_id"],
        },
    },
    "evaluate_predictions": {
        "name": "evaluate_predictions",
        "description": (
            """Evaluate (predict) response values at arbitrary parameter points.

Uses the fitted surrogate model to predict responses at specified parameter values.
Optionally returns uncertainty estimates (standard deviation).

Use when:
- Want predictions at specific points
- Exploring parameter space
- Need uncertainty quantification
- Analyzing model predictions

return_std=False (default): Returns mean predictions only
return_std=True: Returns mean predictions + standard deviation for uncertainty

Example use case: Predict yield at Temperature=25, Pressure=5 before running experiment.

Limitations:
- Requires fitted model
- Standard deviation is in transformed space (not original scale)"""
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "session_id": {"type": "string", "description": "Unique identifier of the session"},
                "X": {
                    "type": "array",
                    "description": "Points to evaluate (list of parameter dicts, must not be empty)",
                    "items": {"type": "object", "description": "Dictionary with parameter names as keys"},
                    "minItems": 1,
                },
                "return_std": {
                    "type": "boolean",
                    "description": "If True, return mean + std; if False, return mean only (default: False)",
                    "default": False,
                },
                "verbose": {
                    "type": "integer",
                    "description": "Optimizer verbosity level (0=none, 1=summary, 2=detailed, 3=debug). Default 3 for LLM agents.",
                    "minimum": 0,
                    "maximum": 3,
                    "default": 3,
                },
            },
            "required": ["session_id", "X"],
        },
    },
    # ============================================================================
    # Analysis and Results Functions (5)
    # ============================================================================
    "get_best_results": {
        "name": "get_best_results",
        "description": (
            """Get best results found so far.

Returns the parameter combination with the best observed response value(s).
For multi-objective optimization, returns Pareto-optimal solutions.

Use when:
- Want to know current best parameters
- Summarizing optimization results
- Deciding whether to continue optimization

Returns:
- X_best: Best parameter values
- response_max: Best response value(s)
- n_experiments: Total experiments run

Note: Returns observed best from experimental data, not model predictions."""
        ),
        "parameters": {
            "type": "object",
            "properties": {"session_id": {"type": "string", "description": "Unique identifier of the session"}},
            "required": ["session_id"],
        },
    },
    "get_campaign_data": {
        "name": "get_campaign_data",
        "description": (
            """Export all experimental data from the campaign.

Returns complete dataset with all experiments, parameters, responses, and iteration tracking.
This is the full campaign.data DataFrame converted to JSON.

Use when:
- Need to analyze all experimental data
- Want to visualize optimization progress
- Export data for external analysis
- LLM needs to examine trends and patterns

Returns:
- data: List of all experiments (each with parameters, targets, iteration)
- columns: Column names
- iterations: Unique iteration numbers
- metadata: Parameter names, target names, fit status

Useful for comprehensive data analysis and visualization."""
        ),
        "parameters": {
            "type": "object",
            "properties": {"session_id": {"type": "string", "description": "Unique identifier of the session"}},
            "required": ["session_id"],
        },
    },
    "get_model_diagnostics": {
        "name": "get_model_diagnostics",
        "description": (
            """Get surrogate model quality metrics and diagnostics.

Returns comprehensive model diagnostics including R² scores, loss values,
training data size, and multi-objective metrics (hypervolume, Pareto points).

Use when:
- Assessing model quality
- Deciding if more data is needed
- Debugging poor suggestions
- Understanding model confidence

Key metrics:
- R² score: Model fit quality (1.0 = perfect, <0.5 = poor)
- Loss: Training loss (MLL for GP, MSE for DNN)
- n_training: Number of training points
- hypervolume: Multi-objective convergence metric
- n_pareto_points: Pareto front size

Returns diagnostics per target/response variable.

Limitations:
- R² and loss are None if model not fitted
- Hypervolume requires multi-objective optimization"""
        ),
        "parameters": {
            "type": "object",
            "properties": {"session_id": {"type": "string", "description": "Unique identifier of the session"}},
            "required": ["session_id"],
        },
    },
    "get_optimization_history": {
        "name": "get_optimization_history",
        "description": (
            """Get iteration-by-iteration optimization progress.

Returns per-iteration summary showing how optimization improved over time.
Tracks cumulative best response and mean response per iteration.

Use when:
- Analyzing convergence
- Visualizing optimization progress
- Deciding whether to continue optimization
- Understanding which iterations improved results

Each iteration summary includes:
- n_experiments: Experiments in this iteration
- best_response: Cumulative best for each target
- mean_response: Mean response in this iteration
- hypervolume: Multi-objective progress (if applicable)

Iteration 0 is typically the initial design (LHS/Random).
Subsequent iterations are Bayesian optimization suggestions."""
        ),
        "parameters": {
            "type": "object",
            "properties": {"session_id": {"type": "string", "description": "Unique identifier of the session"}},
            "required": ["session_id"],
        },
    },
    "export_state_dictionary": {
        "name": "export_state_dictionary",
        "description": (
            """Export full internal state dictionary.

Returns the complete state dictionary from campaign.save_state() or optimizer.save_state().
This contains all internal state including model parameters, hyperparameters, RNG state, etc.

Use when:
- Need complete state for debugging
- Want to serialize entire campaign
- Deep inspection of internal state
- Backing up campaign state

object='campaign' (default): Full campaign state (includes optimizer, data, RNG)
object='optimizer': Optimizer state only (model, hyperparameters, training data)

Returns raw state dictionary with all internal representations.

Note: This is advanced functionality for debugging and state inspection."""
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "session_id": {"type": "string", "description": "Unique identifier of the session"},
                "object": {
                    "type": "string",
                    "enum": ["campaign", "optimizer"],
                    "description": "Object type to export ('campaign' or 'optimizer', default: 'campaign')",
                    "default": "campaign",
                },
            },
            "required": ["session_id"],
        },
    },
    # ============================================================================
    # Analysis Functions (2)
    # ============================================================================
    "analyze_shap": {
        "name": "analyze_shap",
        "description": (
            """Compute SHAP (SHapley Additive exPlanations) values for feature importance.

Uses Kernel SHAP to explain which parameters have the strongest effect on the
predicted response. This provides global feature importance and helps understand
which parameters most influence the optimization target.

Use when:
- Model is fitted (check with get_model_diagnostics)
- Need to understand parameter importance
- Deciding which parameters to focus on or fix
- Explaining model predictions to stakeholders

Returns structured data including:
- shap_values_mean: Mean SHAP value for each parameter (positive = increases prediction)
- feature_importance: Parameters ranked by importance (mean absolute SHAP)
- reference_point: Point where SHAP values are computed (default: best point)
- predicted_value: Model prediction at reference point
- expected_value: Baseline/average prediction
- samples: Sample points used for SHAP computation

Note: SHAP values explain predictions, not raw observations. They show how
the model thinks each parameter affects the outcome.

Limitations:
- Requires fitted model
- Computational cost scales with n_samples (default: 100)
- Only explains one target at a time (multi-objective needs separate calls)"""
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "session_id": {"type": "string", "description": "Unique identifier of the session"},
                "target_name": {
                    "type": "string",
                    "description": "Name of the target response to explain (e.g., 'Yield')"
                },
                "n_samples": {
                    "type": "integer",
                    "description": "Number of samples for SHAP computation (10-1000, default: 100)",
                    "default": 100,
                    "minimum": 10,
                    "maximum": 1000
                },
                "reference_point": {
                    "type": "object",
                    "description": "Reference point for SHAP baseline (optional, defaults to best observed point). Dictionary with parameter names as keys.",
                },
                "seed": {
                    "type": "integer",
                    "description": "Random seed for reproducibility (optional)"
                },
            },
            "required": ["session_id", "target_name"],
        },
    },
    "analyze_sensitivity": {
        "name": "analyze_sensitivity",
        "description": (
            """Compute gradient-based sensitivity analysis (dy/dx).

Calculates local sensitivity (numerical gradient) for each parameter at a
reference point. Shows how small changes in each parameter affect the response.
This is faster than SHAP but only provides local information at one point.

Use when:
- Model is fitted (check with get_model_diagnostics)
- Need quick parameter importance ranking
- Analyzing local behavior around a specific point
- Understanding parameter sensitivities at optimal conditions

Returns structured data including:
- sensitivity: dy/dx for each target and parameter (signed values)
- sensitivity_normalized: Absolute sensitivities normalized to [0,1] for comparison
- reference_point: Point where sensitivity is computed (default: best point)
- predictions_at_reference: Model predictions at the reference point

Interpretation:
- Positive sensitivity: Increasing parameter increases response
- Negative sensitivity: Increasing parameter decreases response
- Large absolute value: Parameter has strong local effect
- Near-zero: Parameter has weak local effect

Compared to SHAP:
- Faster (no sampling required)
- Local only (one point vs global average)
- Simpler interpretation (just gradients)
- All targets computed simultaneously

Limitations:
- Requires fitted model with continuous outputs
- Only local information (may not generalize)
- Gradient computed numerically (small perturbation)"""
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "session_id": {"type": "string", "description": "Unique identifier of the session"},
                "reference_point": {
                    "type": "object",
                    "description": "Reference point for sensitivity calculation (optional, defaults to best point). Dictionary with parameter names as keys.",
                },
                "perturbation": {
                    "type": "number",
                    "description": "Perturbation size (dx) for numerical gradient calculation (default: 1e-6)",
                    "default": 1e-6,
                },
            },
            "required": ["session_id"],
        },
    },
    # ============================================================================
    # Informational Functions (1)
    # ============================================================================
    "list_acquisition_functions": {
        "name": "list_acquisition_functions",
        "description": (
            """List all valid acquisition functions with metadata.

Retrieves comprehensive information about all built-in acquisition functions including:
- Name (short code like 'NEI', 'EHVI', 'UCB')
- Modalities (single-objective, multi-objective, or both)
- Task types (optimization, characterization, or both)
- Hyperparameters with defaults and types
- Human-readable descriptions

Use when:
- Need to discover valid acquisition function names dynamically
- Choosing appropriate acquisition function for optimization problem
- Understanding acquisition function capabilities
- Learning hyperparameter options

Returns categorized lists:
- single_objective: Functions for single-objective optimization (e.g., NEI, EI, UCB)
- multi_objective: Functions for multi-objective optimization (e.g., NEHVI, EHVI)
- universal: Functions that work for both (e.g., RS, SF, Mean)

Common acquisition functions:
- NEI: Noisy Expected Improvement (recommended for single-objective)
- NEHVI: Noisy Expected Hypervolume Improvement (recommended for multi-objective)
- UCB: Upper Confidence Bound (exploration-exploitation tradeoff)
- RS: Random Sampling (baseline)

This is particularly useful for LLM agents that need to discover options
dynamically rather than having them hard-coded."""
        ),
        "parameters": {
            "type": "object",
            "properties": {},
            "required": [],
        },
    },
}
