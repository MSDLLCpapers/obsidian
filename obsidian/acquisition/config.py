"""Method pointers and config for acquisition functions"""

# avoid possible naming space collision
import torch

from . import utils as acq_utils
from .botorch import (
    qLogExpectedHypervolumeImprovement,
    qLogExpectedImprovement,
    qLogNoisyExpectedHypervolumeImprovement,
    qLogNoisyExpectedImprovement,
    qLogNParEGO,
    qNegIntegratedPosteriorVariance,
    qProbabilityOfImprovement,
    qSimpleRegret,
    qUpperConfidenceBound,
)
from .characterization import qMultiRandomizedStraddle, qMultiStraddle, qRandomizedStraddle, qStraddle
from .custom import RandomSampling, qMean, qSpaceFill

# TODO: add a short description for each function and nice print at the registry level
_builtin_acq_func_dict = {
    # Single-objective optimization
    "EI": {
        "implementation": qLogExpectedImprovement,
        "hyperparameter_defaults": {"inflate": {"val": 0, "dtype": float, "optional": True}},
        "hyperparameter_parser": acq_utils._ei_hyperparameter_parser,
        "modalities": ["single"],
        "task_types": ["optimization"],
    },
    "NEI": {
        "implementation": qLogNoisyExpectedImprovement,
        "hyperparameter_defaults": {},
        "hyperparameter_parser": acq_utils._noisy_parser,
        "modalities": ["single"],
        "task_types": ["optimization"],
    },
    "PI": {
        "implementation": qProbabilityOfImprovement,
        "hyperparameter_defaults": {"inflate": {"val": 0, "dtype": float, "optional": True}},
        "hyperparameter_parser": acq_utils._ei_hyperparameter_parser,
        "modalities": ["single"],
        "task_types": ["optimization"],
    },
    "UCB": {
        "implementation": qUpperConfidenceBound,
        "hyperparameter_defaults": {"beta": {"val": 1, "dtype": float, "optional": True}},
        "hyperparameter_parser": acq_utils._ucb_hyperparameter_parser,
        "modalities": ["single"],
        "task_types": ["optimization"],
        "output_constraints": False,
    },
    "SR": {
        "implementation": qSimpleRegret,
        "hyperparameter_defaults": {},
        "hyperparameter_parser": acq_utils.default_hyperparameter_parser,
        "modalities": ["single"],
        "task_types": ["optimization"],
        "output_constraints": False,
    },
    "NIPV": {
        "implementation": qNegIntegratedPosteriorVariance,
        "hyperparameter_defaults": {
            "seed": {"val": None, "dtype": int, "optional": True},
            "n_mc_points": {"val": 128, "dtype": int, "optional": True},
        },
        "hyperparameter_parser": acq_utils._nipv_hyperparameter_parser,
        "modalities": ["single"],
        "task_types": ["optimization"],
        "output_constraints": False,
    },
    # Multi-objective optimization
    "EHVI": {
        "implementation": qLogExpectedHypervolumeImprovement,
        "hyperparameter_defaults": {"ref_point": {"val": None, "dtype": list, "optional": True}},
        "hyperparameter_parser": acq_utils._ehvi_hyperparameter_parser,
        "modalities": ["multi"],
        "task_types": ["optimization"],
    },
    "NEHVI": {
        "implementation": qLogNoisyExpectedHypervolumeImprovement,
        "hyperparameter_defaults": {"ref_point": {"val": None, "dtype": list, "optional": True}},
        "hyperparameter_parser": acq_utils._nehvi_hyperparameter_parser,
        "modalities": ["multi"],
        "task_types": ["optimization"],
    },
    "NParEGO": {
        "implementation": qLogNParEGO,
        "hyperparameter_defaults": {"scalarization_weights": {"val": None, "dtype": list, "optional": True}},
        "hyperparameter_parser": acq_utils._nparego_hyperparameter_parser,
        "modalities": ["multi"],
        "task_types": ["optimization"],
    },
    # Single and multi-objective optimization
    "Mean": {
        "implementation": qMean,
        "hyperparameter_defaults": {},
        "hyperparameter_parser": acq_utils.default_hyperparameter_parser,
        "modalities": ["single", "multi"],
        "task_types": ["optimization"],
        "output_constraints": False,
    },
    # Characterization functions
    # Single objective
    "STR": {
        "implementation": qStraddle,
        "hyperparameter_defaults": {
            "threshold": {"val": None, "dtype": float, "optional": True},
            "beta": {"val": 1.96 ** 2, "dtype": float, "optional": True},
        },
        "hyperparameter_parser": qStraddle.parser,
        "modalities": ["single"],
        "task_types": ["characterization"],
        "output_constraints": False,
    },
    "RANDSTR": {
        "implementation": qRandomizedStraddle,
        "hyperparameter_defaults": {
            "threshold": {"val": None, "dtype": float, "optional": True},
            "beta": {"val": None, "dtype": float, "optional": True},
            "batch_size": {"val": 1, "dtype": int, "optional": True},
            "generator": {"val": None, "dtype": torch.Generator, "optional": True},
        },
        "hyperparameter_parser": qRandomizedStraddle.parser,
        "modalities": ["single"],
        "task_types": ["characterization"],
        "output_constraints": False,
    },
    # Multi-objective
    "MSTR": {
        "implementation": qMultiStraddle,
        "hyperparameter_defaults": {
            "threshold": {"val": None, "dtype": list[float], "optional": True},
            "beta": {"val": 1.96 ** 2, "dtype": float, "optional": True},
            "weights": {"val": None, "dtype": list[float], "optional": True},
            "k_decay": {"val": 2.0, "dtype": float, "optional": True},
            "tau": {"val": 0.5, "dtype": float, "optional": True},
            "kernel_reduction": {"val": "softmin", "dtype": str, "optional": True},
        },
        "hyperparameter_parser": qMultiStraddle.parser,
        "modalities": ["multi"],
        "task_types": ["characterization"],
        "output_constraints": False,
    },
    "JAREX": {
        "implementation": qMultiRandomizedStraddle,
        "hyperparameter_defaults": {
            "threshold": {"val": None, "dtype": list[float], "optional": True},
            "beta": {"val": None, "dtype": list[float], "optional": True},
            "batch_size": {"val": 1, "dtype": int, "optional": True},
            "sync_objective_beta": {"val": False, "dtype": bool, "optional": True},
            "weights": {"val": None, "dtype": list[float], "optional": True},
            "k_decay": {"val": 2.0, "dtype": float, "optional": True},
            "tau": {"val": 0.5, "dtype": float, "optional": True},
            "kernel_reduction": {"val": "softmin", "dtype": str, "optional": True},
            "generator": {"val": None, "dtype": torch.Generator, "optional": True},
        },
        "hyperparameter_parser": qMultiRandomizedStraddle.parser,
        "modalities": ["multi"],
        "task_types": ["characterization"],
        "output_constraints": False,
    },
    # Universal functions
    "RS": {
        "implementation": RandomSampling,
        "hyperparameter_defaults": {
            "generator": {"val": None, "dtype": torch.Generator, "optional": True},
        },
        "hyperparameter_parser": RandomSampling.parser,
        "modalities": ["single", "multi"],
        "task_types": ["optimization", "characterization"],
        "output_constraints": False,
    },
    "SF": {
        "implementation": qSpaceFill,
        "hyperparameter_defaults": {},
        "hyperparameter_parser": qSpaceFill.parser,
        "modalities": ["single", "multi"],
        "task_types": ["optimization", "characterization"],
        "output_constraints": False,
    },
}

# Backward compatibility alias
_builtin_acq_func_dict["MRANDSTR"] = _builtin_acq_func_dict["JAREX"]

# Create the singleton registry instance
_registry = acq_utils.AcquisitionRegistry(_builtin_acq_func_dict)


def reset_registry():
    """
    Reset the acquisition function registry to its default state.

    This clears all custom/external registered functions and restores only
    the built-in acquisition functions. Useful for testing or when you need
    to start fresh.
    """
    _registry.reset(_builtin_acq_func_dict)


# Export the registration function with proper signature
acquisition_function_register = _registry.register_acquisition_function

# Export the registry properties for backward compatibility
aq_class_dict = _registry.aq_class_dict
aq_hp_defaults = _registry.aq_hp_defaults
external_aqs = _registry.external_aqs
# Note that one nested layer is added for future characterization functions
# For now, to keep backward compatibility, we flatten the structure
valid_aqs = _registry.valid_opt_aqs
aq_defaults = _registry.aq_defaults
universal_aqs = _registry.universal_aqs
unconstrainable_aqs = _registry.unconstrainable_aqs

# Export the registry for direct access if needed
registry = _registry
