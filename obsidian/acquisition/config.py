"""Method pointers and config for acquisition functions"""

from typing import Callable, Dict, Type

# avoid possible naming space collision
import obsidian.acquisition.utils as acq_utils

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
from .custom import qMean, qSpaceFill

_builtin_acq_func_dict = {
    # Single-objective optimization
    "EI": {
        "implementation": qLogExpectedImprovement,
        "hyperparameter_defaults": {
            "inflate": {"val": 0, "dtype": float, "optional": True}
        },
        "hyperparameter_parser": acq_utils._ei_hyperparameter_parser,
        "modalities": ["single"],
    },
    "NEI": {
        "implementation": qLogNoisyExpectedImprovement,
        "hyperparameter_defaults": {},
        "hyperparameter_parser": acq_utils._noisy_parser,
        "modalities": ["single"],
    },
    "PI": {
        "implementation": qProbabilityOfImprovement,
        "hyperparameter_defaults": {"inflate": {"val": 0, "dtype": float, "optional": True}},
        "hyperparameter_parser": acq_utils._ei_hyperparameter_parser,
        "modalities": ["single"],
    },
    "UCB": {
        "implementation": qUpperConfidenceBound,
        "hyperparameter_defaults": {"beta": {"val": 1, "dtype": float, "optional": True}},
        "hyperparameter_parser": acq_utils._ucb_hyperparameter_parser,
        "modalities": ["single"],
    },
    "SR": {
        "implementation": qSimpleRegret,
        "hyperparameter_defaults": {},
        "hyperparameter_parser": None,
        "hyperparameter_parser": None,
        "modalities": ["single"],
    },
    "NIPV": {
        "implementation": qNegIntegratedPosteriorVariance,
        "hyperparameter_defaults": {},
        "hyperparameter_parser": acq_utils._nipv_hyperparameter_parser,
        "modalities": ["single"],
    },
    # Multi-objective optimization
    "EHVI": {
        "implementation": qLogExpectedHypervolumeImprovement,
        "hyperparameter_defaults": {"ref_point": {"val": None, "dtype": list, "optional": True}},
        "hyperparameter_parser": acq_utils._ehvi_hyperparameter_parser,
        "modalities": ["multi"],
    },
    "NEHVI": {
        "implementation": qLogNoisyExpectedHypervolumeImprovement,
        "hyperparameter_defaults": {"ref_point": {"val": None, "dtype": list, "optional": True}},
        "hyperparameter_parser": acq_utils._nehvi_hyperparameter_parser,
        "modalities": ["multi"],
    },
    "NParEGO": {
        "implementation": qLogNParEGO,
        "hyperparameter_defaults": {
            "scalarization_weights": {"val": None, "dtype": list, "optional": True}
        },
        "hyperparameter_parser": acq_utils._nparego_hyperparameter_parser,
        "modalities": ["multi"],
    },
    # Universal optimization functions
    "RS": {
        "implementation": None,
        "hyperparameter_defaults": {},
        "hyperparameter_parser": None,
        "modalities": ["single", "multi"],
    },
    "Mean": {
        "implementation": qMean,
        "hyperparameter_defaults": {},
        "hyperparameter_parser": None,
        "modalities": ["single", "multi"],
    },
    "SF": {
        "implementation": qSpaceFill,
        "hyperparameter_defaults": {},
        "hyperparameter_parser": acq_utils._sf_hyperparameter_parser,
        "modalities": ["single", "multi"],
    },
}

# Create the singleton registry instance
_registry = acq_utils.AcquisitionRegistry(_builtin_acq_func_dict)


# Export the registration function with the same interface
def acquisition_function_register(
    name: str,
    implementation: Type,
    hp_defaults: Dict | None = None,
    is_single_target=False,
    is_multi_target=False,
    set_as_default=False,
    overloading=False,
    internal_parser=False,
    external_parser: Callable | None = None,
):
    """
    Register a new acquisition function (backward compatibility wrapper).

    This function maintains the original API while using the new registry internally.
    """
    _registry.register_acquisition_function(
        name=name,
        implementation=implementation,
        hp_defaults=hp_defaults,
        is_single_target=is_single_target,
        is_multi_target=is_multi_target,
        set_as_default=set_as_default,
        overloading=overloading,
        internal_hyperparameter_parser=internal_parser,
        external_hyperparameter_parser=external_parser,
    )


# Export the registry properties for backward compatibility
aq_class_dict = _registry.aq_class_dict
aq_hp_defaults = _registry.aq_hp_defaults
external_aqs = _registry.external_aqs
# Note that one nested layer is added for future characterization functions
# For now, to keep backward compatibility, we flatten the structure
valid_aqs = _registry.valid_opt_aqs
aq_defaults = _registry.aq_defaults["optimization"]
universal_aqs = _registry.universal_aqs

# Export the registry for direct access if needed
registry = _registry
