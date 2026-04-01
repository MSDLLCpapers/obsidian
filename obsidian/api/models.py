"""
Pydantic models for HTTP request/response validation.

These models convert between JSON/dict representations and Obsidian objects.
"""

from typing import Any, Dict, List, Literal, Optional, Union
from pydantic import BaseModel, Field, field_validator
from datetime import datetime

from obsidian.parameters import (
    ParamSpace,
    Param_Continuous,
    Param_Categorical,
    Param_Ordinal,
    Param_Observational,
    Target,
)

# ============================================================================
# Parameter Models
# ============================================================================


class ContinuousParameter(BaseModel):
    """Continuous parameter definition."""

    type: Literal["continuous"] = "continuous"
    name: str
    min: float
    max: float
    search_min: Optional[float] = None
    search_max: Optional[float] = None

    def to_obsidian(self) -> Param_Continuous:
        """Convert to Obsidian Param_Continuous."""
        param = Param_Continuous(self.name, self.min, self.max)
        if self.search_min is not None and self.search_max is not None:
            param.set_search(self.search_min, self.search_max)
        return param


class CategoricalParameter(BaseModel):
    """Categorical parameter definition."""

    type: Literal["categorical"] = "categorical"
    name: str
    categories: list[str]
    search_categories: Optional[list[str]] = None

    @field_validator("categories")
    @classmethod
    def check_categories_not_empty(cls, v):
        if len(v) < 2:
            raise ValueError("Categorical parameter must have at least 2 categories")
        return v

    def to_obsidian(self) -> Param_Categorical:
        """Convert to Obsidian Param_Categorical."""
        param = Param_Categorical(self.name, self.categories)
        if self.search_categories:
            param.set_search(self.search_categories)
        return param


class OrdinalParameter(BaseModel):
    """Ordinal parameter definition."""

    type: Literal["ordinal"] = "ordinal"
    name: str
    categories: list[str]  # Order matters!
    search_categories: Optional[list[str]] = None

    @field_validator("categories")
    @classmethod
    def check_categories_not_empty(cls, v):
        if len(v) < 2:
            raise ValueError("Ordinal parameter must have at least 2 categories")
        return v

    def to_obsidian(self) -> Param_Ordinal:
        """Convert to Obsidian Param_Ordinal."""
        param = Param_Ordinal(self.name, self.categories)
        if self.search_categories:
            param.set_search(self.search_categories)
        return param


class ObservationalParameter(BaseModel):
    """Observational parameter definition (not optimized)."""

    type: Literal["observational"] = "observational"
    name: str
    min: float
    max: float

    def to_obsidian(self) -> Param_Observational:
        """Convert to Obsidian Param_Observational."""
        return Param_Observational(self.name, self.min, self.max)


# Union type for any parameter
ParameterDefinition = Union[ContinuousParameter, CategoricalParameter, OrdinalParameter, ObservationalParameter]


# ============================================================================
# Target/Objective Models
# ============================================================================


class TargetDefinition(BaseModel):
    """Optimization target definition."""

    name: str
    aim: Literal["min", "max"] = "max"
    f_transform: Optional[str] = "Standard"

    def to_obsidian(self) -> Target:
        """Convert to Obsidian Target."""
        return Target(name=self.name, aim=self.aim, f_transform=self.f_transform)


# ============================================================================
# Session Models
# ============================================================================


class SessionCreate(BaseModel):
    """Request to create a new session."""

    name: Optional[str] = None
    parameters: list[ParameterDefinition]
    targets: list[TargetDefinition]
    seed: Optional[int] = None
    surrogate: str | dict | list[str | dict] = Field(
        default="GP",
        description="Surrogate model type(s). String: 'GP'|'DNN'|'DKL'|'MixedGP'|'GPflat'|'GPprior'|'MTGP'. "
                    "Dict for hyperparameters: {'DNN': {'p_dropout': 0.2, 'h_width': 32, 'h_layers': 2}}. "
                    "List for multi-output: ['GP', {'DNN': {...}}]"
    )

    @field_validator("parameters")
    @classmethod
    def check_parameters_not_empty(cls, v):
        if len(v) == 0:
            raise ValueError("Must provide at least one parameter")
        return v

    @field_validator("targets")
    @classmethod
    def check_targets_not_empty(cls, v):
        if len(v) == 0:
            raise ValueError("Must provide at least one target")
        return v


class SessionMetadata(BaseModel):
    """Session metadata (lightweight, for listing)."""

    session_id: str
    name: str
    created_at: str  # ISO format datetime
    updated_at: str  # ISO format datetime
    status: str
    n_experiments: int = 0
    n_parameters: int = 0
    n_targets: int = 0


class SessionDetail(SessionMetadata):
    """Detailed session information."""

    parameter_names: list[str] = Field(default_factory=list)
    target_names: list[str] = Field(default_factory=list)


# ============================================================================
# Operation Request/Response Models
# ============================================================================


class InitializeRequest(BaseModel):
    """Request to initialize experiment design."""

    m_initial: int = Field(default=10, ge=1, description="Number of initial experiments")
    method: str = Field(default="LHS", description="Design method (LHS, Random, etc.)")
    seed: Optional[int] = None


class InitializeResponse(BaseModel):
    """Response from initialize operation."""

    suggestions: list[dict[str, Any]]  # List of experiment dicts
    n_experiments: int
    method: str


class SampleRequest(BaseModel):
    """Request to sample points from parameter space."""

    n_points: int = Field(default=10, ge=1, description="Number of points to sample")
    method: str = Field(default="LHS", description="Sampling method (LHS, Random, Sobol, etc.)")
    seed: Optional[int] = None


class SampleResponse(BaseModel):
    """Response from sample operation."""

    samples: list[dict[str, Any]]  # List of sampled points
    n_points: int
    method: str


class DataRequest(BaseModel):
    """Request to add experimental data."""

    data: list[dict[str, Any]]  # List of experiment results

    @field_validator("data")
    @classmethod
    def check_data_not_empty(cls, v):
        if len(v) == 0:
            raise ValueError("Data list cannot be empty")
        return v


class DataResponse(BaseModel):
    """Response from add_data operation."""

    rows_added: int
    total_rows: int


class FitRequest(BaseModel):
    """Request to fit surrogate model."""

    fit_options: dict[str, Any] = Field(default_factory=dict)
    verbose: Optional[int] = Field(
        default=None,
        ge=0,
        le=3,
        description="Optimizer verbosity (0=none, 1=summary, 2=detailed, 3=debug)"
    )


class FitResponse(BaseModel):
    """Response from fit operation."""

    status: str
    message: str = "Model fitted successfully"


class SuggestRequest(BaseModel):
    """Request to suggest next experiments."""

    m_batch: int = Field(default=1, ge=1, description="Number of experiments to suggest")
    acquisition: list[str | dict[str, dict]] = Field(
        default=["NEI"],
        description="Acquisition function(s). String: 'NEI'|'EI'|'UCB'|... "
                    "Dict for hyperparameters: {'UCB': {'beta': 2.0}}, {'EHVI': {'ref_point': [-350, -20]}}"
    )
    optim_samples: Optional[int] = None
    optim_restarts: Optional[int] = None
    manual_seed: Optional[int] = Field(
        default=None, description="Manual seed for exploration (optional, for reproducibility)"
    )
    verbose: Optional[int] = Field(
        default=None,
        ge=0,
        le=3,
        description="Optimizer verbosity (0=none, 1=summary, 2=detailed, 3=debug)"
    )
    fixed_var: Optional[dict[str, Union[float, str]]] = Field(
        default=None,
        description="Fix specific parameters during optimization. E.g., {'Temperature': 25.0, 'Variant': 'A'}"
    )
    optim_sequential: bool = Field(
        default=True,
        description="Sequential (True) vs joint (False) batch optimization. False is better for batch designs but slower."
    )
    X_pending: Optional[list[dict[str, Any]]] = Field(
        default=None,
        description="Experiments already queued/running. Acquisition will account for these."
    )


class SuggestResponse(BaseModel):
    """Response from suggest operation."""

    suggestions: list[dict[str, Any]]  # Suggested experiments
    evaluation: Optional[list[dict[str, Any]]] = None  # Acquisition values and predictions
    n_suggestions: int


class EvaluateRequest(BaseModel):
    """Request to evaluate (predict) on points."""

    X: list[dict[str, Any]]  # Points to evaluate
    return_std: bool = Field(default=False, description="If True, return mean + std; if False, return mean only")
    verbose: Optional[int] = Field(
        default=None,
        ge=0,
        le=3,
        description="Optimizer verbosity (0=none, 1=summary, 2=detailed, 3=debug)"
    )

    @field_validator("X")
    @classmethod
    def check_X_not_empty(cls, v):
        if len(v) == 0:
            raise ValueError("X list cannot be empty")
        return v


class EvaluateResponse(BaseModel):
    """Response from evaluate operation."""

    predictions: list[dict[str, Any]]
    n_points: int


class BestResult(BaseModel):
    """Best results from campaign."""

    X_best: Optional[dict[str, Any]] = None
    response_max: Optional[Union[float, dict[str, float]]] = None
    n_experiments: int
    message: Optional[str] = None


# ============================================================================
# Data Export Models
# ============================================================================


class DataExportResponse(BaseModel):
    """Response from get campaign data operation."""

    data: list[dict[str, Any]]  # Campaign data as list of dicts
    n_rows: int
    n_columns: int
    columns: list[str]
    iterations: list[int]  # Unique iteration numbers
    metadata: dict[str, Any]  # parameter_names, target_names, is_fitted, etc.


# ============================================================================
# Diagnostics Models
# ============================================================================


class SurrogateDiagnostics(BaseModel):
    """Diagnostics for a single surrogate model."""

    response_name: str
    model_type: str
    r2_score: Optional[float] = None
    loss: Optional[float] = None
    n_training: int
    is_fit: bool


class DiagnosticsResponse(BaseModel):
    """Response from diagnostics endpoint."""

    session_id: str
    is_fitted: bool
    n_experiments: int
    n_training_points: int
    n_parameters: int
    n_targets: int
    surrogates: list[SurrogateDiagnostics]
    is_multi_objective: bool
    hypervolume: Optional[float] = None
    n_pareto_points: Optional[int] = None
    current_iteration: Optional[int] = None
    n_iterations: int


# ============================================================================
# History Models
# ============================================================================


class IterationSummary(BaseModel):
    """Summary for a single iteration."""

    iteration: int
    n_experiments: int
    best_response: dict[str, float]  # Cumulative best for each response
    mean_response: dict[str, float]  # Mean in this iteration
    hypervolume: Optional[float] = None
    n_pareto_points: Optional[int] = None


class HistoryResponse(BaseModel):
    """Response from history endpoint."""

    session_id: str
    n_iterations: int
    iterations: list[IterationSummary]
    parameter_names: list[str]
    target_names: list[str]
    is_multi_objective: bool
    total_experiments: int


# ============================================================================
# State Export Models
# ============================================================================


class StateExportResponse(BaseModel):
    """Response from state dictionary export."""

    state: dict[str, Any]  # Full state dictionary from save_state()
    object_type: str  # "campaign" or "optimizer"


# ============================================================================
# Informational Models
# ============================================================================


class AcquisitionFunctionInfo(BaseModel):
    """Information about a single acquisition function."""

    name: str
    modalities: list[str]  # ["single"], ["multi"], or ["single", "multi"]
    task_types: list[str]  # ["optimization"], ["characterization"], or both
    hyperparameters: dict[str, Any]  # {param_name: {default, type, optional}}
    description: str


class AcquisitionFunctionsResponse(BaseModel):
    """Response from acquisition functions listing."""

    functions: list[AcquisitionFunctionInfo]
    count: int
    single_objective: list[str]  # List of names for single-objective functions
    multi_objective: list[str]  # List of names for multi-objective functions
    universal: list[str]  # List of names that work for both


# ============================================================================
# Error Models
# ============================================================================


class ErrorResponse(BaseModel):
    """Standard error response."""

    detail: str
    error_type: str = "error"
    session_id: Optional[str] = None


# ============================================================================
# Helper Functions
# ============================================================================


def build_param_space(parameters: list[ParameterDefinition]) -> ParamSpace:
    """
    Convert list of ParameterDefinition to ParamSpace.

    Args:
        parameters: List of parameter definitions

    Returns:
        ParamSpace object
    """
    obsidian_params = [p.to_obsidian() for p in parameters]
    return ParamSpace(obsidian_params)


def build_targets(targets: list[TargetDefinition]) -> list[Target]:
    """
    Convert list of TargetDefinition to Target objects.

    Args:
        targets: List of target definitions

    Returns:
        List of Target objects
    """
    return [t.to_obsidian() for t in targets]


# ============================================================================
# Analysis Request/Response Models
# ============================================================================


class ShapAnalysisRequest(BaseModel):
    """Request for SHAP analysis."""

    target_name: str = Field(description="Name of the target response to explain")
    n_samples: int = Field(
        default=100,
        ge=10,
        le=1000,
        description="Number of samples to generate for SHAP values (10-1000)"
    )
    reference_point: Optional[dict[str, Any]] = Field(
        default=None,
        description="Reference point for SHAP baseline. If null, uses best point from optimization."
    )
    seed: Optional[int] = Field(
        default=None,
        description="Random seed for reproducibility"
    )


class FeatureImportance(BaseModel):
    """Feature importance entry."""

    parameter: str
    importance: float
    rank: int


class ShapAnalysisResponse(BaseModel):
    """Response from SHAP analysis."""

    target_name: str
    reference_point: dict[str, Any]
    predicted_value: float = Field(description="Model prediction at reference point")
    expected_value: float = Field(description="Expected value (baseline/average prediction)")
    shap_values_mean: dict[str, float] = Field(
        description="Mean SHAP value for each parameter (positive = increases prediction)"
    )
    feature_importance: list[FeatureImportance] = Field(
        description="Parameters ranked by importance (mean absolute SHAP value)"
    )
    n_samples: int
    samples: list[dict[str, Any]] = Field(
        description="Sample of points used for SHAP computation (up to 50 points)"
    )


class SensitivityAnalysisRequest(BaseModel):
    """Request for sensitivity analysis."""

    reference_point: Optional[dict[str, Any]] = Field(
        default=None,
        description="Reference point for sensitivity calculation. If null, uses best point."
    )
    perturbation: float = Field(
        default=1e-6,
        gt=0,
        description="Perturbation size (dx) for numerical gradient calculation"
    )


class SensitivityAnalysisResponse(BaseModel):
    """Response from sensitivity analysis."""

    reference_point: dict[str, Any]
    predictions_at_reference: dict[str, float] = Field(
        description="Model predictions at the reference point"
    )
    sensitivity: dict[str, dict[str, float]] = Field(
        description="Sensitivity values (dy/dx) for each target and parameter"
    )
    sensitivity_normalized: dict[str, dict[str, float]] = Field(
        description="Sensitivity values normalized to [0,1] for comparison"
    )
    perturbation: float
