"""PyTests for obsidian.acquisition registry and configuration"""

import pandas as pd
import pytest
from botorch.acquisition.monte_carlo import MCAcquisitionFunction
from botorch.utils import t_batch_mode_transform

from obsidian.acquisition import acquisition_function_register, registry
from obsidian.acquisition.config import _builtin_acq_func_dict, _registry, reset_registry
from obsidian.experiment import ExpDesigner, Simulator
from obsidian.experiment.benchmark import shifted_parab
from obsidian.optimizer import BayesianOptimizer
from obsidian.parameters import Target
from obsidian.tests.param_configs import X_sp_cont_small


@pytest.fixture(autouse=True)
def reset_registry_after_test():
    """Reset registry after each test to avoid state pollution between tests"""
    yield
    reset_registry()


@pytest.mark.fast
def test_registry_singleton():
    """Test that the registry is a singleton"""
    from obsidian.acquisition.config import registry as reg1
    from obsidian.acquisition import registry as reg2

    assert reg1 is reg2
    assert reg1 is _registry


@pytest.mark.fast
def test_registry_has_builtin_functions():
    """Test that all built-in acquisition functions are registered"""
    builtin_aqs = list(_builtin_acq_func_dict.keys())

    for aq_name in builtin_aqs:
        assert aq_name in registry.configs, f"Built-in acquisition function {aq_name} not found in registry"
        config = registry.get_config(aq_name)
        assert config.name == aq_name
        assert config.implementation is not None
        assert config.hyperparameter_parser is not None


@pytest.mark.fast
def test_registry_valid_aqs():
    """Test that valid acquisition function sets are correctly populated"""
    # Single-objective optimization
    assert "NEI" in registry.valid_aqs["optimization"]["single"]
    assert "EI" in registry.valid_aqs["optimization"]["single"]
    assert "UCB" in registry.valid_aqs["optimization"]["single"]

    # Multi-objective optimization
    assert "NEHVI" in registry.valid_aqs["optimization"]["multi"]
    assert "EHVI" in registry.valid_aqs["optimization"]["multi"]
    assert "NParEGO" in registry.valid_aqs["optimization"]["multi"]

    # Universal functions
    assert "RS" in registry.valid_aqs["optimization"]["single"]
    assert "RS" in registry.valid_aqs["optimization"]["multi"]


@pytest.mark.fast
def test_registry_defaults():
    """Test that default acquisition functions are set correctly"""
    assert registry.aq_defaults["optimization"]["single"] == "NEI"
    assert registry.aq_defaults["optimization"]["multi"] == "NEHVI"
    # Characterization acquisition function defaults
    assert registry.aq_defaults["characterization"]["single"] == "RANDSTR"
    assert registry.aq_defaults["characterization"]["multi"] == "JAREX"


@pytest.mark.fast
def test_registry_characterization_functions():
    """Test that all characterization acquisition functions are registered"""
    char_aqs = ["STR", "RANDSTR", "MSTR", "JAREX", "MRANDSTR"]
    for aq_name in char_aqs:
        assert aq_name in registry.configs, f"{aq_name} not found in registry"
        config = registry.get_config(aq_name)
        assert "characterization" in config.task_types


@pytest.mark.fast
def test_registry_characterization_valid_aqs():
    """Test that characterization acquisition functions appear in valid_aqs"""
    # Single-objective characterization
    assert "STR" in registry.valid_aqs["characterization"]["single"]
    assert "RANDSTR" in registry.valid_aqs["characterization"]["single"]

    # Multi-objective characterization
    assert "MSTR" in registry.valid_aqs["characterization"]["multi"]
    assert "JAREX" in registry.valid_aqs["characterization"]["multi"]
    assert "MRANDSTR" in registry.valid_aqs["characterization"]["multi"]

    # Universal functions also available for characterization
    assert "RS" in registry.valid_aqs["characterization"]["single"]
    assert "SF" in registry.valid_aqs["characterization"]["multi"]

    # Optimization-only functions should NOT be in characterization
    assert "EI" not in registry.valid_aqs["characterization"]["single"]
    assert "NEHVI" not in registry.valid_aqs["characterization"]["multi"]


def _expected_sets_from_builtin_dict(builtin: dict) -> dict:
    """Standalone reference parser: derives the expected category sets from
    ``_builtin_acq_func_dict`` without going through the registry. Used to
    cross-check the registry's bookkeeping."""
    valid = {
        "optimization": {"single": set(), "multi": set()},
        "characterization": {"single": set(), "multi": set()},
    }
    unconstrainable = set()
    for name, cfg in builtin.items():
        for task in cfg["task_types"]:
            for modality in cfg["modalities"]:
                valid[task][modality].add(name)
        if not cfg.get("output_constraints", True):
            unconstrainable.add(name)
    universal_opt = valid["optimization"]["single"] & valid["optimization"]["multi"]
    universal_charact = valid["characterization"]["single"] & valid["characterization"]["multi"]
    universal = universal_opt & universal_charact
    return {
        "valid_aqs": valid,
        "universal_opt_aqs": universal_opt,
        "universal_charact_aqs": universal_charact,
        "universal_aqs": universal,
        "unconstrainable_aqs": unconstrainable,
    }


@pytest.mark.fast
def test_registry_sets_match_standalone_parser():
    """Cross-check the registry's category sets against a standalone parser.

    The registry computes ``valid_aqs``, ``universal_*_aqs``, and
    ``unconstrainable_aqs`` via incremental updates as functions are registered.
    This test derives the same sets directly from ``_builtin_acq_func_dict`` and
    asserts they match — guarding against regressions like a self-referential
    intersection that silently produces an empty set.
    """
    expected = _expected_sets_from_builtin_dict(_builtin_acq_func_dict)
    expected_aqs = expected["valid_aqs"]

    for task in ["optimization", "characterization"]:
        for modality in ["single", "multi"]:
            assert registry.valid_aqs[task][modality] == expected_aqs[task][modality]

    for set_name in ["universal_opt_aqs", "universal_charact_aqs", "universal_aqs", "unconstrainable_aqs"]:
        assert getattr(registry, set_name) == expected[set_name]


@pytest.mark.fast
def test_registry_mrandstr_jarex_alias():
    """Test that MRANDSTR is an alias for JAREX (same implementation and config)"""
    from obsidian.acquisition.characterization import qMultiRandomizedStraddle

    jarex_config = registry.get_config("JAREX")
    mrandstr_config = registry.get_config("MRANDSTR")

    assert jarex_config.implementation is mrandstr_config.implementation
    assert jarex_config.implementation is qMultiRandomizedStraddle
    assert jarex_config.hyperparameter_parser is mrandstr_config.hyperparameter_parser


@pytest.mark.fast
@pytest.mark.parametrize(
    "aq_name,optional_hp",
    [
        ("STR", "threshold"),
        ("RANDSTR", "threshold"),
        ("MSTR", "threshold"),
        ("JAREX", "threshold"),
    ],
)
def test_registry_characterization_optional_hyperparameters(aq_name, optional_hp):
    """Test that characterization functions have threshold as optional hyperparameter"""
    defaults = registry.get_default_hyperparameters(aq_name)
    assert optional_hp in defaults
    # Threshold is now optional - can be set on Target object instead
    assert defaults[optional_hp]["optional"] is True


@pytest.mark.fast
@pytest.mark.parametrize("aq_name", ["RANDSTR", "JAREX"])
def test_registry_characterization_generator_hyperparameter(aq_name):
    """Test that randomized characterization functions expose generator as optional hp"""
    import torch

    defaults = registry.get_default_hyperparameters(aq_name)
    assert "generator" in defaults
    assert defaults["generator"]["optional"] is True
    assert defaults["generator"]["val"] is None
    assert defaults["generator"]["dtype"] is torch.Generator


@pytest.mark.fast
def test_get_config():
    """Test retrieving acquisition function configuration"""
    config = registry.get_config("NEI")
    assert config.name == "NEI"
    assert config.implementation is not None
    assert config.hyperparameter_defaults == {}
    assert config.modalities == ["single"]
    assert config.task_types == ["optimization"]


@pytest.mark.fast
def test_get_config_unknown():
    """Test that getting unknown acquisition function raises error"""
    with pytest.raises(ValueError, match="Unknown acquisition function"):
        registry.get_config("UNKNOWN_AQ")


@pytest.mark.fast
def test_get_default_hyperparameters():
    """Test getting default hyperparameters"""
    # EI has inflate parameter
    ei_defaults = registry.get_default_hyperparameters("EI")
    assert "inflate" in ei_defaults
    assert ei_defaults["inflate"]["val"] == 0

    # NEI has no hyperparameters
    nei_defaults = registry.get_default_hyperparameters("NEI")
    assert nei_defaults == {}


@pytest.mark.fast
def test_merge_with_defaults():
    """Test merging provided hyperparameters with defaults"""
    config = registry.get_config("EI")

    # With custom value
    merged = config.merge_with_defaults({"inflate": 0.1})
    assert merged == {"inflate": 0.1}

    # Without custom value, should use default
    merged = config.merge_with_defaults({})
    assert merged == {"inflate": 0}


# Test registering an external function
class CustomAcquisition(MCAcquisitionFunction):
    """A simple custom acquisition function that favors exploration"""

    def __init__(
        self,
        model,
        sampler=None,
        objective=None,
        posterior_transform=None,
        X_pending=None,
        exploration_weight=1.0,
        constraints=None,
        **kwargs,
    ):
        super().__init__(model, sampler, objective, posterior_transform, X_pending)
        self.exploration_weight = exploration_weight
        # Ignore constraints and other kwargs for this simple implementation

    @t_batch_mode_transform()
    def forward(self, X):
        """Returns mean + exploration_weight * std"""
        # X shape: [batch, q, d]
        posterior = self.model.posterior(X)
        # Get samples: shape [n_samples, batch, q, output_dim]
        samples = self.get_posterior_samples(posterior)

        # Apply objective
        obj = self.objective(samples, X)
        # obj shape: [n_samples, batch, q] or [n_samples, batch, q, m] for multi-output

        # Mean over samples
        mean = obj.mean(dim=0)  # [batch, q] or [batch, q, m]

        # Std over samples for exploration bonus
        std = obj.std(dim=0)  # [batch, q] or [batch, q, m]

        # Combine mean and std
        exploration_bonus = mean + self.exploration_weight * std  # [batch, q] or [batch, q, m]

        # Sum over q dimension (and m dimension if multi-output)
        return (
            exploration_bonus.sum(dim=-1).sum(dim=-1) if exploration_bonus.dim() > 2 else exploration_bonus.sum(dim=-1)
        )


def custom_parser(aq_kwargs, hps, context):
    """Parser for custom acquisition function"""
    aq_kwargs["exploration_weight"] = hps.get("exploration_weight", 1.0)
    return aq_kwargs


@pytest.mark.fast
def test_register_external_function():
    """Test registering an external acquisition function"""
    # Register the custom acquisition function
    acquisition_function_register(
        name="CustomExplore",
        implementation=CustomAcquisition,
        hp_defaults={"exploration_weight": {"val": 1.0, "dtype": float, "optional": True}},
        is_optimization=True,
        is_single_target=True,
        is_multi_target=False,
        parser=custom_parser,
    )

    # Verify it's registered
    assert "CustomExplore" in registry.configs
    config = registry.get_config("CustomExplore")
    assert config.name == "CustomExplore"
    assert config.implementation == CustomAcquisition
    assert config.is_external is True
    assert "exploration_weight" in config.hyperparameter_defaults
    assert config.hyperparameter_parser == custom_parser

    # Verify it's in valid_aqs
    assert "CustomExplore" in registry.valid_aqs["optimization"]["single"]
    assert "CustomExplore" not in registry.valid_aqs["optimization"]["multi"]


@pytest.mark.fast
def test_register_external_function_overloading():
    """Test that overloading an existing function requires overloading=True"""
    # First registration should succeed
    acquisition_function_register(
        name="TempCustom1",
        implementation=CustomAcquisition,
        is_optimization=True,
        is_single_target=True,
    )

    # Second registration without overloading should fail
    with pytest.raises(ValueError, match="already registered"):
        acquisition_function_register(
            name="TempCustom1",
            implementation=CustomAcquisition,
            is_optimization=True,
            is_single_target=True,
        )

    # With overloading=True should succeed
    acquisition_function_register(
        name="TempCustom1",
        implementation=CustomAcquisition,
        is_optimization=True,
        is_single_target=True,
        overloading=True,
    )


@pytest.mark.fast
def test_register_external_function_reuse_parser():
    """Test reusing parser from existing function when overloading"""
    # Register initial function with custom parser
    acquisition_function_register(
        name="TempCustom2",
        implementation=CustomAcquisition,
        is_optimization=True,
        is_single_target=True,
        parser=custom_parser,
    )

    # Overload with reuse_parser=True
    class CustomAcquisition2(CustomAcquisition):
        pass

    acquisition_function_register(
        name="TempCustom2",
        implementation=CustomAcquisition2,
        is_optimization=True,
        is_single_target=True,
        overloading=True,
        reuse_parser=True,
    )

    # Verify parser was reused
    config = registry.get_config("TempCustom2")
    assert config.hyperparameter_parser == custom_parser


@pytest.mark.fast
def test_reset_registry():
    """Test resetting the registry to default state"""
    # Register a custom function
    acquisition_function_register(
        name="TempResetTest",
        implementation=CustomAcquisition,
        is_optimization=True,
        is_single_target=True,
        parser=custom_parser,
    )

    # Verify it's registered
    assert "TempResetTest" in registry.configs

    # Also overload a built-in function
    original_ei = registry.get_config("EI").implementation
    acquisition_function_register(
        name="EI",
        implementation=CustomAcquisition,
        is_optimization=True,
        is_single_target=True,
        overloading=True,
    )
    assert registry.get_config("EI").implementation != original_ei

    # Reset the registry
    reset_registry()

    # Verify custom function is gone
    with pytest.raises(ValueError, match="Unknown acquisition function"):
        registry.get_config("TempResetTest")

    # Verify built-in function is restored
    assert registry.get_config("EI").implementation == original_ei

    # Verify all built-in functions are still present
    for aq_name in _builtin_acq_func_dict.keys():
        assert aq_name in registry.configs


@pytest.mark.fast
def test_overload_internal_function():
    """Test overloading a built-in internal acquisition function"""
    # Get the original config
    original_config = registry.get_config("EI")
    original_impl = original_config.implementation

    # Overload with custom implementation
    class CustomEI(CustomAcquisition):
        """Custom implementation overloading built-in EI"""

        pass

    acquisition_function_register(
        name="EI",
        implementation=CustomEI,
        hp_defaults={"exploration_weight": {"val": 2.0, "dtype": float, "optional": True}},
        is_optimization=True,
        is_single_target=True,
        parser=custom_parser,
        overloading=True,
    )

    # Verify it was overloaded
    config = registry.get_config("EI")
    assert config.implementation == CustomEI
    assert config.implementation != original_impl
    assert config.hyperparameter_parser == custom_parser

    # Restore original (important for other tests)
    acquisition_function_register(
        name="EI",
        implementation=original_impl,
        hp_defaults={"inflate": {"val": 0, "dtype": float, "optional": True}},
        is_optimization=True,
        is_single_target=True,
        overloading=True,
    )


@pytest.mark.fast
def test_overload_internal_function_reuse_parser():
    """Test overloading a built-in function while reusing its parser"""
    # Get the original config
    original_config = registry.get_config("UCB")
    original_impl = original_config.implementation
    original_parser = original_config.hyperparameter_parser

    # Overload with custom implementation but reuse parser
    class CustomUCB(CustomAcquisition):
        """Custom implementation overloading built-in UCB"""

        pass

    acquisition_function_register(
        name="UCB",
        implementation=CustomUCB,
        is_optimization=True,
        is_single_target=True,
        overloading=True,
        reuse_parser=True,
    )

    # Verify it was overloaded
    config = registry.get_config("UCB")
    assert config.implementation == CustomUCB
    assert config.implementation != original_impl
    # Parser should be reused from original
    assert config.hyperparameter_parser == original_parser

    # Restore original (important for other tests)
    acquisition_function_register(
        name="UCB",
        implementation=original_impl,
        hp_defaults={"beta": {"val": 1, "dtype": float, "optional": True}},
        is_optimization=True,
        is_single_target=True,
        overloading=True,
    )


@pytest.mark.fast
def test_register_external_function_validation():
    """Test validation when registering external function"""
    # Missing task type
    with pytest.raises(ValueError, match="without specifying task type"):
        acquisition_function_register(
            name="BadAq1",
            implementation=CustomAcquisition,
            is_single_target=True,
        )

    # Missing modality
    with pytest.raises(ValueError, match="without specifying target modality"):
        acquisition_function_register(
            name="BadAq2",
            implementation=CustomAcquisition,
            is_optimization=True,
        )

    # Cannot supply both parser and reuse_parser - need to register first
    acquisition_function_register(
        name="BadAq3",
        implementation=CustomAcquisition,
        is_optimization=True,
        is_single_target=True,
        parser=custom_parser,
    )

    with pytest.raises(ValueError, match="Cannot supply both"):
        acquisition_function_register(
            name="BadAq3",
            implementation=CustomAcquisition,
            is_optimization=True,
            is_single_target=True,
            parser=custom_parser,
            reuse_parser=True,
            overloading=True,
        )


# Test that registered external function can be used in suggest
@pytest.fixture()
def setup_optimizer():
    """Setup optimizer with initial data for testing suggest with custom acquisition"""
    X_space = X_sp_cont_small
    designer = ExpDesigner(X_space, seed=1)
    X0 = designer.initialize(m_initial=6, method="LHS")
    simulator = Simulator(X_space, shifted_parab, eps=0.05, rng=1)
    y0 = simulator.simulate(X0)
    Z0 = pd.concat([X0, y0], axis=1)

    optimizer = BayesianOptimizer(X_space, surrogate="GP", seed=0, verbose=0)
    target = Target(name="Response", f_transform="Standard", aim="max")
    optimizer.fit(Z0, target=target)

    return optimizer


@pytest.mark.fast
def test_external_function_in_suggest(setup_optimizer):
    """Test that a registered external acquisition function can be used in suggest"""
    optimizer = setup_optimizer

    # Register custom acquisition if not already registered
    if "CustomExplore" not in registry.configs:
        acquisition_function_register(
            name="CustomExplore",
            implementation=CustomAcquisition,
            hp_defaults={"exploration_weight": {"val": 1.0, "dtype": float, "optional": True}},
            is_optimization=True,
            is_single_target=True,
            parser=custom_parser,
        )

    # Test suggest with custom acquisition function
    X_suggest, eval_suggest = optimizer.suggest(
        m_batch=2,
        acquisition=["CustomExplore"],
        optim_sequential=False,
        optim_samples=32,
        optim_restarts=2,
    )

    # Verify suggestions are valid
    assert len(X_suggest) == 2
    assert not X_suggest.isna().any().any()
    assert len(eval_suggest) == 2


@pytest.mark.fast
def test_external_function_with_hyperparameters(setup_optimizer):
    """Test using custom acquisition function with custom hyperparameters"""
    optimizer = setup_optimizer

    # Ensure CustomExplore is registered
    if "CustomExplore" not in registry.configs:
        acquisition_function_register(
            name="CustomExplore",
            implementation=CustomAcquisition,
            hp_defaults={"exploration_weight": {"val": 1.0, "dtype": float, "optional": True}},
            is_optimization=True,
            is_single_target=True,
            parser=custom_parser,
        )

    # Test with custom hyperparameters
    X_suggest, eval_suggest = optimizer.suggest(
        m_batch=2,
        acquisition=[{"CustomExplore": {"exploration_weight": 2.0}}],
        optim_sequential=False,
        optim_samples=32,
        optim_restarts=2,
    )

    # Verify suggestions are valid
    assert len(X_suggest) == 2
    assert not X_suggest.isna().any().any()


@pytest.mark.fast
def test_builtin_and_external_acquisition_comparison(setup_optimizer):
    """Test that both built-in and external acquisition functions work"""
    optimizer = setup_optimizer

    # Ensure CustomExplore is registered
    if "CustomExplore" not in registry.configs:
        acquisition_function_register(
            name="CustomExplore",
            implementation=CustomAcquisition,
            hp_defaults={"exploration_weight": {"val": 1.0, "dtype": float, "optional": True}},
            is_optimization=True,
            is_single_target=True,
            parser=custom_parser,
        )

    # Test built-in
    X_builtin, eval_builtin = optimizer.suggest(
        m_batch=2,
        acquisition=["NEI"],
        optim_sequential=False,
        optim_samples=32,
        optim_restarts=2,
    )

    # Test external
    X_custom, eval_custom = optimizer.suggest(
        m_batch=2,
        acquisition=["CustomExplore"],
        optim_sequential=False,
        optim_samples=32,
        optim_restarts=2,
    )

    # Both should produce valid suggestions
    assert len(X_builtin) == 2
    assert len(X_custom) == 2
    assert not X_builtin.isna().any().any()
    assert not X_custom.isna().any().any()


# Additional tests for utils.py - coverage for uncovered lines


class TestAcquisitionConfig:
    """Test AcquisitionConfig dataclass"""

    def test_get_default_hyperparameters(self):
        """Test getting default hyperparameters"""
        from obsidian.acquisition.utils import AcquisitionConfig

        config = AcquisitionConfig(
            name="test",
            implementation=None,
            hyperparameter_defaults={
                "beta": {"val": 2.0, "optional": True},
                "tau": {"val": 1e-3, "optional": True},
                "no_val": {"optional": False},  # No "val" key
            },
        )
        defaults = config.get_default_hyperparameters()
        assert defaults == {"beta": 2.0, "tau": 1e-3}
        assert "no_val" not in defaults

    def test_merge_with_defaults(self):
        """Test merging hyperparameters with defaults"""
        from obsidian.acquisition.utils import AcquisitionConfig

        config = AcquisitionConfig(
            name="test",
            implementation=None,
            hyperparameter_defaults={
                "beta": {"val": 2.0, "optional": True},
                "tau": {"val": 1e-3, "optional": True},
            },
        )
        merged = config.merge_with_defaults({"beta": 3.0})
        assert merged == {"beta": 3.0, "tau": 1e-3}


@pytest.mark.fast
class TestFilterBotorchArguments:
    """Test _filter_botorch_arguments method"""

    def test_posterior_transform_warning(self):
        """Test that posterior_transform warning is issued"""
        aq_kwargs = {"objective": None}
        hps = {"posterior_transform": lambda x: x}

        with pytest.warns(UserWarning, match="posterior_transform"):
            result_kwargs, result_hps = registry._filter_botorch_arguments(aq_kwargs, hps)

        assert "posterior_transform" in result_kwargs
        assert "posterior_transform" not in result_hps

    def test_objective_warning(self):
        """Test objective handling with warning"""
        aq_kwargs = {"objective": None}
        hps = {"objective": "test_objective"}

        with pytest.warns(UserWarning, match="Consider directly passing"):
            result_kwargs, result_hps = registry._filter_botorch_arguments(aq_kwargs, hps)

        assert "objective" not in result_hps

    def test_ignored_params_warnings(self):
        """Test that sampler, X_pending, constraints are ignored"""
        aq_kwargs = {}
        hps = {"sampler": "test", "X_pending": "test", "constraints": "test"}

        with pytest.warns(UserWarning) as record:
            result_kwargs, result_hps = registry._filter_botorch_arguments(aq_kwargs, hps)

        # Should have warnings for all three
        assert len(record) >= 3
        assert "sampler" not in result_hps
        assert "X_pending" not in result_hps
        assert "constraints" not in result_hps


@pytest.mark.fast
class TestHyperparameterParsers:
    """Test hyperparameter parser functions"""

    def test_nipv_parser(self):
        """Test NIPV hyperparameter parser"""
        from obsidian.acquisition.utils import _nipv_hyperparameter_parser, ParserContext
        import torch

        context = ParserContext(
            f_t=torch.randn(10, 1),
            X_baseline=torch.randn(10, 2),
            m_batch=2,
            n_dim=2,
            target=[Target("y", aim="max")],
            objective=None,
        )
        aq_kwargs = {"objective": None}
        hps = {"seed": 42, "n_mc_points": 64}

        result = _nipv_hyperparameter_parser(aq_kwargs, hps, context)

        assert "mc_points" in result
        assert result["sampler"] is None
        assert "objective" not in result

    def test_scalarization_weights_parser_list(self):
        """Test scalarization weights parser with list"""
        from obsidian.acquisition.utils import _scalarization_weights_parser, ParserContext
        import torch

        context = ParserContext(
            f_t=torch.randn(10, 2),
            X_baseline=torch.randn(10, 2),
            m_batch=1,
            n_dim=2,
            target=[Target("y1", aim="max"), Target("y2", aim="max")],
            objective=None,
        )
        aq_kwargs = {}
        hps = {"scalarization_weights": [1.0, 2.0]}

        _scalarization_weights_parser(aq_kwargs, hps, context)

        assert "scalarization_weights" in aq_kwargs
        assert isinstance(aq_kwargs["scalarization_weights"], torch.Tensor)

    def test_space_partitioning_parser(self):
        """Test space partitioning parser for EHVI"""
        from obsidian.acquisition.utils import _space_partitioning, ParserContext
        import torch

        context = ParserContext(
            f_t=torch.randn(10, 2),
            X_baseline=torch.randn(10, 2),
            m_batch=1,
            n_dim=2,
            target=[Target("y1", aim="max"), Target("y2", aim="max")],
            objective=None,
        )
        aq_kwargs = {"ref_point": torch.tensor([0.0, 0.0])}
        hps = {}

        _space_partitioning(aq_kwargs, hps, context)

        assert "partitioning" in aq_kwargs

    def test_ref_point_from_targets(self):
        """Test extracting ref_point from targets"""
        from obsidian.acquisition.utils import _ref_point, ParserContext
        import torch

        target1 = Target("y1", aim="max")
        target2 = Target("y2", aim="min")

        context = ParserContext(
            f_t=torch.tensor([[1.0, 2.0], [3.0, 1.0]]),
            X_baseline=torch.randn(2, 2),
            m_batch=1,
            n_dim=2,
            target=[target1, target2],
            objective=None,
        )

        aq_kwargs = {}
        hps = {}

        _ref_point(aq_kwargs, hps, context)

        assert "ref_point" in aq_kwargs
        assert isinstance(aq_kwargs["ref_point"], torch.Tensor)


@pytest.mark.fast
class TestValidateHyperparameters:
    """Test validate_hyperparameters method - critical validation paths"""

    def test_unsupported_acquisition_function_error(self):
        """Test error for unsupported acquisition function"""
        from obsidian.utils import TaskType
        from obsidian.exceptions import UnsupportedError

        aq_kwargs = {}
        with pytest.raises(UnsupportedError, match="Acquisition function must be selected from"):
            registry.validate_hyperparameters(
                task_type=TaskType.CHARACTERIZATION,
                o_dim=1,
                aq_name="EI",  # EI is optimization-only
                hps={},
                aq_kwargs=aq_kwargs,
            )

    def test_unknown_hyperparameters_error(self):
        """Test error for unknown hyperparameters"""
        from obsidian.utils import TaskType

        aq_kwargs = {}
        with pytest.raises(ValueError, match="Unknown hyperparameters"):
            registry.validate_hyperparameters(
                task_type=TaskType.OPTIMIZATION,
                o_dim=1,
                aq_name="NEI",
                hps={"invalid_param": 123, "another_bad_param": 456},
                aq_kwargs=aq_kwargs,
            )


@pytest.mark.fast
class TestRegisterExternalModalityBranches:
    """Test register_external with different modality combinations"""

    def test_register_both_single_and_multi(self):
        """Test registering for both single and multi modalities"""
        acquisition_function_register(
            name="TestBothModalities",
            implementation=CustomAcquisition,
            is_optimization=True,
            is_single_target=True,
            is_multi_target=True,
        )

        assert "TestBothModalities" in registry.valid_aqs["optimization"]["single"]
        assert "TestBothModalities" in registry.valid_aqs["optimization"]["multi"]

    def test_register_multi_only(self):
        """Test registering for multi-objective only"""
        acquisition_function_register(
            name="TestMultiOnly",
            implementation=CustomAcquisition,
            is_optimization=True,
            is_single_target=False,
            is_multi_target=True,
        )

        assert "TestMultiOnly" not in registry.valid_aqs["optimization"]["single"]
        assert "TestMultiOnly" in registry.valid_aqs["optimization"]["multi"]

    def test_register_characterization_function(self):
        """Test registering characterization function"""
        acquisition_function_register(
            name="TestCharacterization",
            implementation=CustomAcquisition,
            is_optimization=False,
            is_characterization=True,
            is_single_target=True,
            is_multi_target=False,
        )

        assert "TestCharacterization" in registry.valid_aqs["characterization"]["single"]

    def test_reuse_parser_without_existing_error(self):
        """Test reuse_parser error when function doesn't exist"""
        with pytest.raises(ValueError, match="reuse_parser=True requested but"):
            acquisition_function_register(
                name="NonExistentFunction",
                implementation=CustomAcquisition,
                is_optimization=True,
                is_single_target=True,
                reuse_parser=True,
            )

    def test_register_with_no_hp_defaults_and_no_implementation(self):
        """Test registering with neither hp_defaults nor implementation"""
        acquisition_function_register(
            name="TestNoDefaults",
            implementation=None,
            hp_defaults=None,
            is_optimization=True,
            is_single_target=True,
        )

        config = registry.get_config("TestNoDefaults")
        assert config.hyperparameter_defaults == {}

    def test_set_as_default_optimization(self):
        """Test set_as_default for optimization"""
        original_default = registry.aq_defaults["optimization"]["single"]

        acquisition_function_register(
            name="TestDefaultOpt",
            implementation=CustomAcquisition,
            is_optimization=True,
            is_single_target=True,
            set_as_default=True,
        )

        assert registry.aq_defaults["optimization"]["single"] == "TestDefaultOpt"
        registry.aq_defaults["optimization"]["single"] = original_default

    def test_set_as_default_characterization(self):
        """Test set_as_default for characterization"""
        original_default = registry.aq_defaults["characterization"]["single"]

        acquisition_function_register(
            name="TestDefaultChar",
            implementation=CustomAcquisition,
            is_characterization=True,
            is_single_target=True,
            set_as_default=True,
        )

        assert registry.aq_defaults["characterization"]["single"] == "TestDefaultChar"
        registry.aq_defaults["characterization"]["single"] = original_default


@pytest.mark.fast
class TestRegistryHelperMethods:
    """Test registry helper methods"""

    def test_get_valid_hyperparameters(self):
        """Test get_valid_hyperparameters method"""
        valid_hps = registry.get_valid_hyperparameters("EI")
        assert "inflate" in valid_hps
        assert isinstance(valid_hps, set)

    def test_get_hyperparameter_parser(self):
        """Test get_hyperparameter_parser method"""
        parser = registry.get_hyperparameter_parser("UCB")
        assert callable(parser)

    def test_valid_charact_aqs_property(self):
        """Test valid_charact_aqs property"""
        char_aqs = registry.valid_charact_aqs
        assert "single" in char_aqs
        assert "multi" in char_aqs
        assert "RANDSTR" in char_aqs["single"]
        assert "JAREX" in char_aqs["multi"]


@pytest.mark.fast
class TestExtractHpDefaults:
    """Test _extract_hp_defaults helper function"""

    def test_extract_hp_defaults_with_defaults(self):
        """Test extracting hyperparameter defaults from a class"""
        from obsidian.acquisition.utils import _extract_hp_defaults

        class DummyAcquisition:
            def __init__(self, model, beta=2.0, tau=1e-3):
                pass

        defaults = _extract_hp_defaults(DummyAcquisition)

        assert "self" not in defaults
        assert "model" not in defaults
        assert "beta" in defaults
        assert defaults["beta"]["val"] == 2.0
        assert defaults["beta"]["optional"] is True
        assert "tau" in defaults
        assert defaults["tau"]["val"] == 1e-3
        assert defaults["tau"]["optional"] is True

    def test_extract_hp_defaults_no_default(self):
        """Test extracting required parameters without defaults"""
        from obsidian.acquisition.utils import _extract_hp_defaults

        class DummyAcquisition:
            def __init__(self, model, required_param):
                pass

        defaults = _extract_hp_defaults(DummyAcquisition)

        assert "required_param" in defaults
        assert defaults["required_param"]["val"] is None
        assert defaults["required_param"]["optional"] is False


@pytest.mark.fast
class TestDefaultHyperparameterParser:
    """Test default_hyperparameter_parser function"""

    def test_default_parser_returns_aq_kwargs(self):
        """Test that default parser just returns aq_kwargs unchanged"""
        from obsidian.acquisition.utils import default_hyperparameter_parser

        aq_kwargs = {"model": "test", "beta": 2.0}
        hps = {"some_hp": "value"}
        context = None

        result = default_hyperparameter_parser(aq_kwargs, hps, context)

        assert result is aq_kwargs
        assert result == {"model": "test", "beta": 2.0}
