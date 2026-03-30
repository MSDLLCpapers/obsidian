"""
Informational endpoints for discovery and metadata.

This router provides endpoints that help users (especially LLM agents) discover
valid options, capabilities, and metadata about the Obsidian optimization engine.
"""

from fastapi import APIRouter

from obsidian.acquisition.config import _builtin_acq_func_dict
from obsidian.api.models import AcquisitionFunctionInfo, AcquisitionFunctionsResponse

router = APIRouter()


@router.get("/acquisition-functions", response_model=AcquisitionFunctionsResponse)
def list_acquisition_functions():
    """
    Get list of valid acquisition functions and their metadata.

    Returns information about all built-in acquisition functions including:
    - Name (short code like "NEI", "EHVI")
    - Modalities (single-objective, multi-objective, or both)
    - Task types (optimization, characterization, or both)
    - Hyperparameters and their defaults
    - Human-readable descriptions

    This endpoint is particularly useful for LLM agents that need to discover
    valid acquisition functions dynamically rather than hard-coding them.

    Returns:
        Complete catalog of acquisition functions with metadata
    """
    functions = []

    # Common descriptions for acquisition functions
    descriptions = {
        "EI": "Expected Improvement - Classic BO acquisition for single-objective optimization",
        "NEI": "Noisy Expected Improvement - EI variant for noisy observations (recommended for most cases)",
        "PI": "Probability of Improvement - Probability-based acquisition function",
        "UCB": "Upper Confidence Bound - Balances exploitation and exploration via beta parameter",
        "SR": "Simple Regret - Minimizes regret, focuses on exploitation",
        "NIPV": "Negative Integrated Posterior Variance - Space-filling acquisition for characterization",
        "EHVI": "Expected Hypervolume Improvement - Multi-objective BO acquisition",
        "NEHVI": "Noisy Expected Hypervolume Improvement - EHVI variant for noisy observations (recommended for multi-objective)",
        "NParEGO": "Noisy ParEGO - Multi-objective via scalarization",
        "Mean": "Posterior Mean - Exploit current surrogate model's predictions",
        "RS": "Random Sampling - Uniform random sampling (baseline/control)",
        "SF": "Space Filling - Maximize minimum distance between points",
    }

    for name, config in _builtin_acq_func_dict.items():
        # Extract hyperparameters in simplified format
        hyperparameters = {}
        for hp_name, hp_config in config.get("hyperparameter_defaults", {}).items():
            hyperparameters[hp_name] = {
                "default": hp_config["val"],
                "type": hp_config["dtype"].__name__,
                "optional": hp_config.get("optional", False),
            }

        functions.append(
            AcquisitionFunctionInfo(
                name=name,
                modalities=config["modalities"],
                task_types=config["task_types"],
                hyperparameters=hyperparameters,
                description=descriptions.get(name, ""),
            )
        )

    # Sort by modality then name for better organization
    functions.sort(key=lambda f: (f.modalities[0], f.name))

    return AcquisitionFunctionsResponse(
        functions=functions,
        count=len(functions),
        single_objective=[f.name for f in functions if "single" in f.modalities],
        multi_objective=[f.name for f in functions if "multi" in f.modalities],
        universal=[f.name for f in functions if len(f.modalities) > 1],
    )
