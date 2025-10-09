import inspect
import warnings
from dataclasses import dataclass, field
from typing import Any, Callable, Type, TypedDict

import torch
from botorch.acquisition.objective import MCAcquisitionObjective
from botorch.models.model import Model
from botorch.utils.multi_objective.box_decompositions.non_dominated import (
    NondominatedPartitioning,
)
from botorch.utils.sampling import draw_sobol_samples

from obsidian.config import TORCH_DTYPE
from obsidian.exceptions import UnsupportedError
from obsidian.surrogates import EnsembleModel


class ParserContext(TypedDict):
    """Type hints for parser context"""

    f_t: torch.Tensor
    X_baseline: torch.Tensor
    objective: Callable | None
    model: Model
    m_batch: int
    n_dim: int
    # TODO: rename this variable
    # the current name is mainly for compatibility with the existing code
    o: torch.Tensor


# Hyperparameter parsing helpers
def _ei_hyperparameter_parser(
    aq_kwargs: dict[str, Any], hps: dict[str, Any], context: ParserContext
) -> None:
    """Parser for EI and PI acquisition functions"""
    o = context["o"]
    o_max = o.max(dim=0).values * (1 + hps.get("inflate", 0))
    aq_kwargs["best_f"] = o_max


def _ucb_hyperparameter_parser(
    aq_kwargs: dict[str, Any], hps: dict[str, Any], context: ParserContext
) -> None:
    """Parser for UCB acquisition function"""
    aq_kwargs["beta"] = hps.get("beta", 1.0)


def _noisy_parser(
    aq_kwargs: dict[str, Any], hps: dict[str, Any], context: ParserContext
) -> None:
    """Parser for noisy acquisition functions"""
    aq_kwargs["X_baseline"] = context["X_baseline"]
    if any(isinstance(m, EnsembleModel) for m in context["model"].models):
        aq_kwargs["cache_root"] = False


def _ref_point(
    aq_kwargs: dict[str, Any], hps: dict[str, Any], context: ParserContext
) -> None:
    """Reference point for hypervolume-based acquisition functions"""
    o = context["o"]
    ref_point = hps.get("ref_point")
    if ref_point is None:
        max_val = o.max(dim=0).values
        min_val = o.min(dim=0).values
        ref_point = min_val - 0.1 * (max_val - min_val)
    else:
        ref_point = torch.tensor(ref_point)
    aq_kwargs["ref_point"] = ref_point


def _nipv_hyperparameter_parser(
    aq_kwargs: dict[str, Any], hps: dict[str, Any], context: ParserContext
) -> None:
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


def _sf_hyperparameter_parser(
    aq_kwargs: dict[str, Any], hps: dict[str, Any], context: ParserContext
) -> None:
    """Parser for Space Filling acquisition function"""
    aq_kwargs["X_baseline"] = context["X_baseline"]


# Utility functions for composite parsers
def _prune_baseline(
    aq_kwargs: dict[str, Any], hps: dict[str, Any], context: ParserContext
) -> None:
    """Parser for functions that need baseline pruning"""
    aq_kwargs["prune_baseline"] = True


def _space_partitioning(
    aq_kwargs: dict[str, Any], hps: dict[str, Any], context: ParserContext
) -> None:
    """Additional parser specific to EHVI"""
    o = context["o"]
    aq_kwargs["partitioning"] = NondominatedPartitioning(aq_kwargs["ref_point"], Y=o)


def _scalarization_weights_parser(
    aq_kwargs: dict[str, Any], hps: dict[str, Any], context: ParserContext
) -> None:
    """Parser for NParEGO scalarization weights"""
    w = hps.get("scalarization_weights")
    if isinstance(w, list):
        w = torch.tensor(w)
        w = w / torch.sum(torch.abs(w))
    aq_kwargs["scalarization_weights"] = w


# Composite parsers for acquisition functions that need multiple parsers
def _nehvi_hyperparameter_parser(
    aq_kwargs: dict[str, Any], hps: dict[str, Any], context: ParserContext
) -> None:
    """Composite parser for NEHVI"""
    _noisy_parser(aq_kwargs, hps, context)
    _ref_point(aq_kwargs, hps, context)
    _prune_baseline(aq_kwargs, hps, context)


def _ehvi_hyperparameter_parser(
    aq_kwargs: dict[str, Any], hps: dict[str, Any], context: ParserContext
) -> None:
    """Composite parser for EHVI"""
    _ref_point(aq_kwargs, hps, context)
    _space_partitioning(aq_kwargs, hps, context)


def _nparego_hyperparameter_parser(
    aq_kwargs: dict[str, Any], hps: dict[str, Any], context: ParserContext
) -> None:
    """Composite parser for NParEGO"""
    _noisy_parser(aq_kwargs, hps, context)
    _prune_baseline(aq_kwargs, hps, context)
    _scalarization_weights_parser(aq_kwargs, hps, context)


@dataclass
class AcquisitionConfig:
    """Configuration for an acquisition function"""

    name: str
    implementation: type
    hyperparameter_defaults: dict[str, dict[str, Any]] = field(default_factory=dict)
    hyperparameter_parser: Callable | None = None
    modalities: list[str] = field(default_factory=list)
    is_external: bool = False

    @staticmethod
    def default_hyperparameter_parser(
        aq_kwargs: dict[str, Any],
        hps: dict[str, Any],
        context: ParserContext | None = None,
    ) -> None:
        """Default dummy hyperparameter parser"""
        pass

    def get_default_hyperparameters(self) -> dict[str, Any]:
        """Get default hyperparameter values"""
        return {
            key: config["val"]
            for key, config in self.hyperparameter_defaults.items()
            if "val" in config
        }

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
        self, aq_kwargs: dict[str, Any], hps: dict[str, Any], context: dict[str, Any]
    ) -> None:
        """Apply hyperparameter parser for this acquisition function"""
        if self.hyperparameter_parser is not None:
            self.hyperparameter_parser(aq_kwargs, hps, context)


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
        self.valid_aqs: dict[str, dict[str, list[str]]] = {
            "optimization": {
                "single": [],
                "multi": [],
            }
        }
        self.valid_char_aqs: set[str] = set()
        self.external_aqs: set[str] = set()
        self.aq_defaults: dict[str, dict[str, str]] = {
            "optimization": {"single": "NEI", "multi": "NEHVI"},
        }
        self.universal_opt_aqs: list[str] = ["RS", "Mean", "SF"]

        # Register built-in functions
        self._register_builtin_functions(BUILTIN_CONFIGS)

    def _register_builtin_functions(self, builtin_configs: dict[str, dict[str, Any]]):
        """Register all built-in acquisition functions from config dictionary"""
        for name, config_data in builtin_configs.items():
            config = AcquisitionConfig(
                name=name,
                implementation=config_data["implementation"],
                hyperparameter_defaults=config_data["hyperparameter_defaults"],
                hyperparameter_parser=config_data.get("hyperparameter_parser"),
                modalities=config_data["modalities"],
                is_external=False,
            )
            self.configs[name] = config

            # Add to appropriate lists
            for modality in config.modalities:
                self.valid_opt_aqs[modality].append(name)

    @staticmethod
    def _filter_botorch_arguments(aq_kwargs: dict[str, Any], hps: dict[str, Any]):
        if "posterior_transform" in hps:
            warnings.warn(
                "`posterior_transform` is not explicitly handled within the package. Users are solely responsible for using it with the acquisition function."
            )
            aq_kwargs["posterior_transform"] = hps.pop("posterior_transform")
        if "objective" in hps:
            warnings.warn(
                "Consider directly passing the objective to the `suggest` function."
            )
            hps_objective = hps.pop("objective")
            if not aq_kwargs["objective"]:
                if hps_objective:
                    warnings.warn(
                        f"Ignoring provided objective {hps_objective} in favor of {aq_kwargs['objective']}."
                    )
                aq_kwargs["objective"] = hps_objective
        for i in ["sampler", "X_pending", "constraints"]:
            if i in hps:
                warnings.warn(
                    f"{i.capitalize()} is handled internally by the package, ignoring provided {i} {hps[i]}."
                )
                hps.pop(i)

    def validate_hyperparameters(
        self, o_dim: int, acquisition: str | dict, aq_kwargs: dict[str, Any]
    ) -> tuple[str, dict]:
        """
        Validates acquisition function and prepares base arguments.

        Args:
            o_dim: Output dimensionality
            acquisition: Acquisition function name (str) or {name: hyperparameters} dict
            aq_kwargs: Base keyword arguments for the acquisition function

        Returns:
            tuple of (aq_name, aq_hps)
        """
        # Parse acquisition input
        if isinstance(acquisition, str):
            aq_name = acquisition
            hps = {}
        else:
            if len(acquisition) != 1:
                raise ValueError(
                    "One dictionary of hyperparameters must be provided for each acquisition function"
                )
            aq_name, hps = next(iter(acquisition.items()))
            if not isinstance(hps, dict):
                raise TypeError("Hyperparameters must be provided as a dictionary")
        
        self._filter_botorch_arguments(hps, aq_kwargs)

        # Determine optimization type
        optim_type = "single" if o_dim == 1 else "multi"

        # Validate acquisition name
        if aq_name not in self.valid_opt_aqs[optim_type]:
            raise UnsupportedError(
                f"Acquisition function must be selected from: {self.valid_opt_aqs[optim_type]}"
            )

        # Get config and validate hyperparameters
        config = self.get_config(aq_name)

        # Check for unknown hyperparameters
        valid_keys = set(config.hyperparameter_defaults.keys())
        provided_keys = set(hps.keys())
        unknown_keys = provided_keys - valid_keys
        if unknown_keys:
            raise ValueError(
                f"Unknown hyperparameters {unknown_keys} for {aq_name}. "
                f"Valid options are: {valid_keys}"
            )

        # Build complete hyperparameters with defaults
        aq_hps = {}
        for key, defaults in config.hyperparameter_defaults.items():
            if hps.get(key) is None:
                if not defaults.get("optional", True):
                    raise ValueError(
                        f"Must specify hyperparameter value {key} for {aq_name}"
                    )
                # Special handling for weights
                if key in ["weights", "scalarization_weights"]:
                    aq_hps[key] = [1] * o_dim
                else:
                    aq_hps[key] = defaults.get("val")
            else:
                aq_hps[key] = hps[key]

        return aq_name, aq_hps

    def register_acquisition_function(
        self,
        name: str,
        implementation: Type,
        hp_defaults: dict | None = None,
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
        - implementation (callable or None): The function implementation.
        - hp_defaults (dict): Optional hyperparameter defaults.
            If not specified, a dummy placeholder dict will be constructed by automatically
            inferring from the acquisition function object, which can be dangerous.
            It is highly recommended to provide explicit hyperparameter defaults.
        - is_single_target (bool): Whether this acquisition function is for single-target optimization.
        - is_multi_target (bool): Whether this acquisition function is for multi-target optimization.
        - overloading (bool): Whether to allow overloading an existing acquisition function with the same name.
            If False and the acquisition function already exists, a ValueError is raised.
            If True, the existing acquisition function will be replaced.
        - set_as_default (bool): Whether to set this acquisition function as default for its modality(s).
        - internal_hyperparameter_parser (bool): Whether to use an internal hyperparameter parser for the
            acquisition function, assuming the parser already exists for this function.
            For example, existing function is overloaded, but no new parser needed.
        - external_hyperparameter_parser (Callable | None, optional): A function to parse arguments for the
            acquisition function. The parser function should take `aq_kwargs` and `hps` dictionaries
            as inputs and modify `aq_kwargs` in-place.
        """

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
                f"Cannot register function '{name}' without specifying target modality "
                f"(single-target or multi-target)."
            )

        # Check for conflicts with default acquisitions
        default_aqs = [name for mod in modalities if self.aq_defaults.get(mod) == name]

        # Check if function already exists
        if name in self.configs:
            if not overloading:
                raise ValueError(
                    f"Acquisition function '{name}' already registered. "
                    f"To overload the existing function, set 'overloading=True'."
                )
            else:
                warnings.warn(f"Overloading existing acquisition function '{name}'.")
                if set_as_default and name in default_aqs:
                    warnings.warn(
                        f"Overloading the default acquisition function {name}."
                    )

                # Remove from existing modality lists
                for modality_list in self.valid_aqs.values():
                    if name in modality_list:
                        modality_list.remove(name)

        # Extract hyperparameter defaults if not provided
        if hp_defaults is None and implementation is not None:
            hp_defaults = _extract_hp_defaults(implementation)
        elif hp_defaults is None:
            hp_defaults = {}

        # Determine hyperparameter parser
        hyperparameter_parser = None
        if external_hyperparameter_parser is not None:
            if internal_hyperparameter_parser:
                raise ValueError(
                    "An external hyperparameter parser is supplied but user also requires "
                    "using an internal parser."
                )
            hyperparameter_parser = external_hyperparameter_parser
        elif internal_hyperparameter_parser:
            # Try to use existing parser if available
            existing_config = self.configs.get(name)
            if existing_config and existing_config.hyperparameter_parser:
                hyperparameter_parser = existing_config.hyperparameter_parser

        # Create new configuration
        config = AcquisitionConfig(
            name=name,
            implementation=implementation,
            hyperparameter_defaults=hp_defaults,
            hyperparameter_parser=hyperparameter_parser,
            modalities=modalities,
            is_external=True,
        )

        # Register the configuration
        self.configs[name] = config

        # Add to valid acquisition lists
        for modality in modalities:
            if modality in self.valid_opt_aqs:
                self.valid_opt_aqs[modality].append(name)

        # Set as default if requested
        if set_as_default:
            for modality in modalities:
                self.aq_defaults["optimization"][modality] = name

    def get_hyperparameter_parser(self, name: str) -> Callable | None:
        """Get the hyperparameter parser for an acquisition function"""
        config = self.configs.get(name)
        return config.hyperparameter_parser if config else None

    def parse_hyperparameters(
        self,
        name: str,
        aq_kwargs: dict[str, Any],
        hps: dict[str, Any],
        context: dict[str, Any],
    ) -> None:
        """Parse hyperparameters for a specific acquisition function"""
        config = self.configs.get(name)
        if config:
            config.parse_hyperparameters(aq_kwargs, hps, context)

    def validate_acquisition(self, name: str, hps: dict[str, Any]) -> None:
        """Validate acquisition function and its hyperparameters"""
        if name not in self.configs:
            raise ValueError(
                f"Unknown acquisition function '{name}'. "
                f"Available options: {list(self.configs.keys())}"
            )

        if not isinstance(hps, dict):
            raise TypeError("Hyperparameters must be provided as a dictionary")

        config = self.configs[name]
        config.validate_hyperparameters(hps)

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

    def process_acquisition_dict(
        self, acquisition: dict[str, dict[str, Any]]
    ) -> tuple[str, dict[str, Any]]:
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
    def universal_aqs(self) -> list[str]:
        """Backward compatibility for universal_aqs"""
        return self.universal_opt_aqs

    @property
    def aq_class_dict(self) -> dict[str, type | None]:
        """Get dictionary of acquisition function implementations"""
        return {name: config.implementation for name, config in self.configs.items()}

    @property
    def aq_hp_defaults(self) -> dict[str, dict[str, Any]]:
        """Get dictionary of acquisition function hyperparameter defaults"""
        return {
            name: config.hyperparameter_defaults
            for name, config in self.configs.items()
        }


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
