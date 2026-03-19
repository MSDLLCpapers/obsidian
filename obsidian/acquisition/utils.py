import inspect
import warnings
from dataclasses import dataclass, field
from typing import Any, Callable, Type, TypedDict

import torch
from botorch.acquisition.objective import MCAcquisitionObjective, PosteriorTransform
from botorch.models.model import Model, ModelList
from botorch.models.model_list_gp_regression import ModelListGP
from botorch.sampling import SobolQMCNormalSampler
from botorch.sampling.list_sampler import ListSampler
from botorch.utils.multi_objective.box_decompositions.non_dominated import (
    NondominatedPartitioning,
)
from botorch.utils.sampling import draw_sobol_samples

from ..config import TORCH_DTYPE
from ..exceptions import UnsupportedError
from ..parameters import Target
from ..surrogates import EnsembleModel
from ..utils import TaskType


class ParserContext(TypedDict):
    """Type hints for parser context"""

    f_t: torch.Tensor
    X_baseline: torch.Tensor
    m_batch: int
    n_dim: int
    target: list[Target]
    generator: torch.Generator | None
    objective: MCAcquisitionObjective | None


def default_hyperparameter_parser(
    aq_kwargs: dict[str, Any], hps: dict[str, Any], context: ParserContext | None = None
) -> dict[str, Any]:
    """Default dummy hyperparameter parser"""
    return aq_kwargs


@dataclass
class AcquisitionConfig:
    """Configuration for an acquisition function"""

    name: str
    implementation: type
    hyperparameter_defaults: dict[str, dict[str, Any]] = field(default_factory=dict)
    hyperparameter_parser: Callable[[dict[str, Any], dict[str, Any], Any], dict[str, Any]] = field(
        default=default_hyperparameter_parser
    )
    modalities: list[str] = field(default_factory=list)
    task_types: list[str] = field(default_factory=list)
    is_external: bool = False

    def get_default_hyperparameters(self) -> dict[str, Any]:
        """Get default hyperparameter values"""
        return {key: config["val"] for key, config in self.hyperparameter_defaults.items() if "val" in config}

    def merge_with_defaults(self, hps: dict[str, Any]) -> dict[str, Any]:
        """Merge provided hyperparameters with defaults. This method does not perform
        validation or parsing, so users should ensure that the provided hyperparameters
        are valid and all required hyperparameters are included."""
        defaults = self.get_default_hyperparameters()
        return {**defaults, **hps}

    def instantiate(self, **kwargs) -> Any:
        """Instantiate the acquisition function with given parameters"""
        return self.implementation(**kwargs)

    def parse_hyperparameters(
        self, aq_kwargs: dict[str, Any], hps: dict[str, Any], context: ParserContext
    ) -> dict[str, Any]:
        """Apply hyperparameter parser for this acquisition function"""
        return self.hyperparameter_parser(aq_kwargs, hps, context)


class AcquisitionRegistry:
    """Singleton registry for acquisition function configurations"""

    _instance = None

    def __new__(cls, BUILTIN_CONFIGS: dict[str, dict[str, Any]]):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._initialize(BUILTIN_CONFIGS)
        return cls._instance

    def _initialize(self, BUILTIN_CONFIGS: dict[str, dict[str, Any]]):
        """Initialize the registry with default acquisition functions"""
        self.configs: dict[str, AcquisitionConfig] = {}
        self.valid_aqs: dict[str, dict[str, set[str]]] = {
            "optimization": {
                "single": set(),
                "multi": set(),
            },
            "characterization": {
                "single": set(),
                "multi": set(),
            },
        }
        self.external_aqs: set[str] = set()
        self.aq_defaults: dict[str, dict[str, str]] = {
            "optimization": {"single": "NEI", "multi": "NEHVI"},
            "characterization": {"single": "RANDSTR", "multi": "MRANDSTR"},
        }
        self.universal_opt_aqs: set[str] = set()
        self.universal_aqs: set[str] = set()
        # Register built-in functions
        self._register_builtin_functions(BUILTIN_CONFIGS)

    def _register_individual_function(
        self,
        name,
        implementation,
        hyperparameter_defaults,
        modalities,
        task_types,
        hyperparameter_parser=default_hyperparameter_parser,
        is_external=False,
    ):
        config = AcquisitionConfig(
            name=name,
            implementation=implementation,
            hyperparameter_defaults=hyperparameter_defaults,
            hyperparameter_parser=hyperparameter_parser,
            modalities=modalities,
            task_types=task_types,
            is_external=is_external,
        )
        self.configs[name] = config

        # Add to appropriate lists
        for task in config.task_types:
            for modality in config.modalities:
                # if task == "characterization" and modality == "multi":
                #     continue
                self.valid_aqs[task][modality].add(name)

    def _register_builtin_functions(self, builtin_configs: dict[str, dict[str, Any]]):
        """Register all built-in acquisition functions from config dictionary"""
        for name, config_data in builtin_configs.items():
            self._register_individual_function(name, **config_data)

        single_opt_aqs = self.valid_aqs["optimization"]["single"]
        multi_opt_aqs = self.valid_aqs["optimization"]["multi"]
        single_charact_aqs = self.valid_aqs["characterization"]["single"]
        multi_charact_aqs = self.valid_aqs["characterization"]["multi"]
        self.universal_opt_aqs = single_opt_aqs & multi_opt_aqs
        self.universal_charact_aqs = single_charact_aqs & multi_charact_aqs
        self.universal_aqs = self.universal_aqs & self.universal_charact_aqs

    @staticmethod
    def _filter_botorch_arguments(
        aq_kwargs: dict[str, Any], hps: dict[str, Any]
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        if "posterior_transform" in hps:
            warnings.warn(
                "`posterior_transform` is not explicitly handled within the package. Users are solely responsible for"
                " using it with the acquisition function."
            )
            aq_kwargs["posterior_transform"] = hps.pop("posterior_transform")
        if "objective" in hps:
            warnings.warn("Consider directly passing the objective to the `suggest` function.")
            hps_objective = hps.pop("objective")
            if not aq_kwargs["objective"]:
                if hps_objective:
                    warnings.warn(f"Ignoring provided objective {hps_objective} in favor of {aq_kwargs['objective']}.")
                aq_kwargs["objective"] = hps_objective
        for i in ["sampler", "X_pending", "constraints"]:
            if i in hps:
                warnings.warn(f"{i.capitalize()} is handled internally by the package, ignoring provided {i} {hps[i]}.")
                hps.pop(i)

        return aq_kwargs, hps

    def validate_hyperparameters(
        self,
        task_type: TaskType,
        o_dim: int,
        aq_name: str,
        hps: dict[str, Any],
        aq_kwargs: dict[str, Any],
    ) -> tuple[dict, dict]:
        """
        Validates acquisition function and prepares base arguments.

        Args:
            o_dim: Output dimensionality
            acquisition: Acquisition function name (str) or {name: hyperparameters} dict
            aq_kwargs: Base keyword arguments for the acquisition function

        Returns:
            tuple of (aq_name, aq_hps)
        """
        # both dictionaries are modified in-place, return for clarity
        aq_kwargs, hps = self._filter_botorch_arguments(aq_kwargs, hps)

        # Determine optimization type
        optim_type = "single" if o_dim == 1 else "multi"
        """
        if task_type.value == "characterization" and optim_type == "multi":
            raise UnsupportedError("Acquisition functions for characterization tasks does not support multi-objective.")
        """

        # Validate acquisition name
        if aq_name not in self.valid_aqs[task_type.value][optim_type]:
            raise UnsupportedError(
                f"Acquisition function must be selected from: {self.valid_aqs[task_type.value][optim_type]}."
                f" {aq_name} for {task_type.value} and {optim_type}-objective is not supported."
            )

        # Get config and validate hyperparameters
        config = self.get_config(aq_name)

        # Check for unknown hyperparameters
        valid_keys = set(config.hyperparameter_defaults.keys())
        provided_keys = set(hps.keys())
        unknown_keys = provided_keys - valid_keys
        if unknown_keys:
            raise ValueError(f"Unknown hyperparameters {unknown_keys} for {aq_name}. Valid options are: {valid_keys}")

        # Build complete hyperparameters with defaults
        aq_hps = {}
        for key, defaults in config.hyperparameter_defaults.items():
            if hps.get(key) is None:
                if not defaults.get("optional", True):
                    raise ValueError(f"Must specify hyperparameter value {key} for {aq_name}")
                # Special handling for weights
                if key in ["weights", "scalarization_weights"]:
                    aq_hps[key] = [1] * o_dim
                else:
                    aq_hps[key] = defaults.get("val")
            else:
                aq_hps[key] = hps[key]

        return aq_kwargs, aq_hps

    def register_acquisition_function(
        self,
        name: str,
        implementation: Type | None,
        hp_defaults: dict | None = None,
        is_optimization: bool = False,
        is_characterization: bool = False,
        is_single_target: bool = False,
        is_multi_target: bool = False,
        set_as_default: bool = False,
        overloading: bool = False,
        internal_hyperparameter_parser: bool = False,
        external_hyperparameter_parser: Callable | None = None,
    ):
        """
        Register a new acquisition function.

        Parameters:
        - name (str): Name of the acquisition function.
        - implementation (callable or None): The function implementation. If None, no implementation stored.
        - hp_defaults (dict): Optional hyperparameter defaults. If None and implementation provided, attempt to infer.
        - task_type (TaskType): Task type this acquisition is intended for (optimization/characterization).
        - is_single_target (bool): Whether this acquisition function is for single-target optimization.
        - is_multi_target (bool): Whether this acquisition function is for multi-target optimization.
        - set_as_default (bool): Whether to set this acquisition function as default for its modality(s).
        - overloading (bool): Whether to allow overloading an existing acquisition function with the same name.
        - internal_hyperparameter_parser (bool): Whether to use an internal hyperparameter parser from an existing config.
        - external_hyperparameter_parser (Callable | None): A function to parse arguments for the function.
        """
        # Determine task_types
        task_types: list[str] = []
        if is_optimization:
            task_types.append("optimization")
        if is_characterization:
            task_types.append("characterization")
        if not task_types:
            raise ValueError(
                f"Cannot register function '{name}' without specifying task type (optimization or characterization)."
            )

        # Determine modalities
        modalities = []
        if is_single_target and is_multi_target:
            modalities = ["single", "multi"]
        elif is_single_target:
            modalities = ["single"]
        elif is_multi_target:
            modalities = ["multi"]
        else:
            raise ValueError(
                f"Cannot register function '{name}' without specifying target modality (single-target or multi-target)."
            )

        # For characterization tasks, only single-target is allowed
        # Comment this block out for experimenting multi-target
        """
        tmp_list = task_types + modalities
        if TaskType.CHARACTERIZATION in task_types and "multi" in modalities and len(tmp_list) < 4:
            # the logic is this
            # opt + char / single + multi -> good
            # opt + char / single -> good
            # opt + char / single -> good
            # opt / single + multi -> good
            # opt / single -> good
            # opt / multi -> good
            # char / single -> good
            # char / multi -> no
            # opt + char / multi -> no
            # char / single + multi -> no
            raise ValueError("Characterization acquisitions must be single-target only (cannot register multi-target).")
        """

        # Check existence and overloading
        exists = name in self.configs
        if exists and not overloading:
            raise ValueError(
                f"Acquisition function '{name}' already registered. To overload the existing function, set"
                " 'overloading=True'."
            )
        if exists and overloading:
            warnings.warn(f"Overloading existing acquisition function '{name}'.")

        # internal_hyperparameter_parser: only allowed if the existing config has an internal parser
        hyperparameter_parser = None
        if internal_hyperparameter_parser:
            if not exists:
                raise ValueError(
                    "internal_hyperparameter_parser=True requested but acquisition function does not already exist to "
                    "take the internal parser from."
                )
            existing = self.configs.get(name)
            if not existing or not existing.hyperparameter_parser:
                raise ValueError(
                    "internal_hyperparameter_parser=True requested but no internal parser is available on the existing"
                    " config."
                )
            hyperparameter_parser = existing.hyperparameter_parser

        # external parser cannot be combined with internal parser flag
        if external_hyperparameter_parser is not None:
            if internal_hyperparameter_parser:
                raise ValueError(
                    "Cannot supply both external_hyperparameter_parser and internal_hyperparameter_parser=True."
                )
            hyperparameter_parser = external_hyperparameter_parser

        # Extract hp_defaults if not provided and implementation is provided
        if hp_defaults is None and implementation is not None:
            hp_defaults = _extract_hp_defaults(implementation)
        elif hp_defaults is None:
            hp_defaults = {}

        self._register_individual_function(
            name,
            implementation,
            hp_defaults,
            modalities,
            task_types,
            hyperparameter_parser=hyperparameter_parser,
            is_external=True,
        )
        # Set as default if requested
        if set_as_default:
            for task in task_types:
                if task == "optimization":
                    for mod in modalities:
                        self.aq_defaults[task][mod] = name
                elif task == "characterization":
                    self.aq_defaults[task]["single"] = name

    def parse_hyperparameters(
        self,
        name: str,
        aq_kwargs: dict[str, Any],
        hps: dict[str, Any],
        context: ParserContext,
    ) -> dict[str, Any]:
        """Parse hyperparameters for a specific acquisition function"""
        return self.get_config(name).parse_hyperparameters(aq_kwargs, hps, context)

    def get_config(self, name: str) -> AcquisitionConfig:
        """Get configuration for an acquisition function"""
        if name not in self.configs:
            raise ValueError(f"Unknown acquisition function '{name}'")
        return self.configs[name]

    def instantiate_acquisition(self, name: str, **kwargs) -> Any:
        """Instantiate an acquisition function with parameters"""
        config = self.get_config(name)
        return config.instantiate(**kwargs)

    def get_valid_hyperparameters(self, name: str) -> set[str]:
        """Get valid hyperparameter names for an acquisition function"""
        config = self.get_config(name)
        return set(config.hyperparameter_defaults.keys())

    def get_default_hyperparameters(self, name: str) -> dict[str, Any]:
        """Get default hyperparameters for an acquisition function"""
        config = self.get_config(name)
        return config.hyperparameter_defaults

    def get_hyperparameter_parser(self, name: str) -> Callable[[dict, dict, ParserContext], dict[str, Any]]:
        config = self.get_config(name)
        return config.hyperparameter_parser

    def process_acquisition_dict(self, acquisition: dict[str, dict[str, Any]]) -> tuple[str, dict[str, Any]]:
        """
        Process acquisition dictionary format {name: hyperparameters}
        Returns tuple of (name, validated_hyperparameters)
        """
        if len(acquisition) != 1:
            raise ValueError("Acquisition dict must have exactly one key")

        aq_name, hps = next(iter(acquisition.items()))
        self.validate_acquisition(aq_name, hps)

        # merge with defaults
        config = self.get_config(aq_name)
        merged_hps = config.merge_with_defaults(hps)

        return aq_name, merged_hps

    # Properties for backward compatibility
    @property
    def valid_opt_aqs(self) -> dict[str, list[str]]:
        """Backward compatibility for valid_aqs"""
        return self.valid_aqs["optimization"]

    @property
    def valid_charact_aqs(self) -> dict[str, list[str]]:
        """Backward compatibility for valid_aqs"""
        return self.valid_aqs["characterization"]

    @property
    def aq_class_dict(self) -> dict[str, type | None]:
        """Get dictionary of acquisition function implementations"""
        return {name: config.implementation for name, config in self.configs.items()}

    @property
    def aq_hp_defaults(self) -> dict[str, dict[str, Any]]:
        """Get dictionary of acquisition function hyperparameter defaults"""
        return {name: config.hyperparameter_defaults for name, config in self.configs.items()}


def _extract_hp_defaults(cls):
    """
    Extract constructor parameters and their default values from a class.

    Returns a dict mapping parameter names to default values or None if no default.
    Skips 'self' and 'model' parameters.
    """
    sig = inspect.signature(cls.__init__)
    hp_defaults = {}
    for name, param in sig.parameters.items():
        # assume all custom acquisition functions use `model` as the variable name
        # for the Gaussian process model
        if name in ("self", "model"):
            continue
        if param.default is inspect.Parameter.empty:
            hp_defaults[name] = {"val": None, "optional": False}
        else:
            hp_defaults[name] = {"val": param.default, "optional": True}
    return hp_defaults


# Hyperparameter parsing helpers
def _objective_transform(context: ParserContext) -> torch.Tensor:
    f_t = context["f_t"]
    X_baseline = context["X_baseline"]
    objective = context.get("objective")
    return f_t if not objective else objective(f_t.unsqueeze(0), X_baseline).squeeze(0)


def _ei_hyperparameter_parser(aq_kwargs: dict[str, Any], hps: dict[str, Any], context: ParserContext) -> dict[str, Any]:
    """Parser for EI and PI acquisition functions"""
    o = _objective_transform(context)
    o_max = o.max(dim=0).values * (1 + hps.get("inflate", 0))
    aq_kwargs["best_f"] = o_max
    return aq_kwargs


def _ucb_hyperparameter_parser(
    aq_kwargs: dict[str, Any], hps: dict[str, Any], context: ParserContext
) -> dict[str, Any]:
    """Parser for UCB acquisition function"""
    aq_kwargs["beta"] = hps.get("beta", 1.0)
    return aq_kwargs


def _noisy_parser(aq_kwargs: dict[str, Any], hps: dict[str, Any], context: ParserContext) -> dict[str, Any]:
    """Parser for noisy acquisition functions"""
    aq_kwargs["X_baseline"] = context["X_baseline"]
    if any(isinstance(m, EnsembleModel) for m in aq_kwargs["model"].models):
        aq_kwargs["cache_root"] = False
    return aq_kwargs


def _nipv_hyperparameter_parser(
    aq_kwargs: dict[str, Any], hps: dict[str, Any], context: ParserContext
) -> dict[str, Any]:
    """Parser for NIPV acquisition function"""
    n_dim = context["n_dim"]
    m_batch = context["m_batch"]
    objective = context.get("objective")

    X_bounds = torch.tensor([[0.0, 1.0]] * n_dim, dtype=TORCH_DTYPE).T
    qmc_samples = draw_sobol_samples(bounds=X_bounds, n=128, q=m_batch)
    aq_kwargs["mc_points"] = qmc_samples.squeeze(-2)
    aq_kwargs["sampler"] = None
    if objective:
        raise UnsupportedError("NIPV does not support objectives")
    return aq_kwargs


# Utility functions for composite parsers
def _ref_point(aq_kwargs: dict[str, Any], hps: dict[str, Any], context: ParserContext) -> None:
    """Reference point for hypervolume-based acquisition functions"""
    o = _objective_transform(context)
    ref_point = hps.get("ref_point")
    if ref_point is None:
        max_val = o.max(dim=0).values
        min_val = o.min(dim=0).values
        ref_point = min_val - 0.1 * (max_val - min_val)
    else:
        ref_point = torch.tensor(ref_point)
    aq_kwargs["ref_point"] = ref_point


def _prune_baseline(aq_kwargs: dict[str, Any], hps: dict[str, Any], context: ParserContext) -> None:
    """Parser for functions that need baseline pruning"""
    aq_kwargs["prune_baseline"] = True


def _space_partitioning(aq_kwargs: dict[str, Any], hps: dict[str, Any], context: ParserContext) -> None:
    """Additional parser specific to EHVI"""
    o = _objective_transform(context)
    aq_kwargs["partitioning"] = NondominatedPartitioning(aq_kwargs["ref_point"], Y=o)


def _scalarization_weights_parser(aq_kwargs: dict[str, Any], hps: dict[str, Any], context: ParserContext) -> None:
    """Parser for NParEGO scalarization weights"""
    w = hps.get("scalarization_weights")
    if isinstance(w, list):
        w = torch.tensor(w)
        w = w / torch.sum(torch.abs(w))
    aq_kwargs["scalarization_weights"] = w


# Composite parsers for acquisition functions that need multiple parsers
def _nehvi_hyperparameter_parser(
    aq_kwargs: dict[str, Any], hps: dict[str, Any], context: ParserContext
) -> dict[str, Any]:
    """Composite parser for NEHVI"""
    _noisy_parser(aq_kwargs, hps, context)
    _ref_point(aq_kwargs, hps, context)
    _prune_baseline(aq_kwargs, hps, context)
    return aq_kwargs


def _ehvi_hyperparameter_parser(
    aq_kwargs: dict[str, Any], hps: dict[str, Any], context: ParserContext
) -> dict[str, Any]:
    """Composite parser for EHVI"""
    _ref_point(aq_kwargs, hps, context)
    _space_partitioning(aq_kwargs, hps, context)
    return aq_kwargs


def _nparego_hyperparameter_parser(
    aq_kwargs: dict[str, Any], hps: dict[str, Any], context: ParserContext
) -> dict[str, Any]:
    """Composite parser for NParEGO"""
    _noisy_parser(aq_kwargs, hps, context)
    _prune_baseline(aq_kwargs, hps, context)
    _scalarization_weights_parser(aq_kwargs, hps, context)
    return aq_kwargs


# we explicitly choose the samplers in optimizer
# this is just a placeholder in case people call the function outside of optimizer
default_sampler = SobolQMCNormalSampler(sample_shape=torch.Size([512]))
