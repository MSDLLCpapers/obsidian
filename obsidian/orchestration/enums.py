"""
Enums and constants for the orchestration layer.
"""

from enum import Enum


class SessionStatus(str, Enum):
    """Status of a campaign session through its lifecycle."""

    CONFIGURED = "configured"  # Session created, campaign configured
    INITIALIZED = "initialized"  # Initial experiments generated
    FITTED = "fitted"  # Surrogate model fitted to data
    SUGGESTING = "suggesting"  # Generating new experiment suggestions
    EVALUATING = "evaluating"  # Evaluating predictions on points
    COMPLETED = "completed"  # Optimization completed
    ERROR = "error"  # Error occurred during operation

    def __str__(self) -> str:
        return self.value
