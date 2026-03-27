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


class FitResponse(BaseModel):
    """Response from fit operation."""

    status: str
    message: str = "Model fitted successfully"


class SuggestRequest(BaseModel):
    """Request to suggest next experiments."""

    m_batch: int = Field(default=1, ge=1, description="Number of experiments to suggest")
    acquisition: list[str] = Field(default=["NEI"], description="Acquisition function(s)")
    optim_samples: Optional[int] = None
    optim_restarts: Optional[int] = None
    manual_seed: Optional[int] = Field(
        default=None, description="Manual seed for exploration (optional, for reproducibility)"
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
