
"""PyTests for AdvExpDesigner – inheritance from ExpDesigner and Campaign integration."""

import json

import numpy as np
import pandas as pd
import pytest

from obsidian.experiment import ExpDesigner, AdvExpDesigner
from obsidian.experiment.advanced_design import (
    _serialize_continuous_params,
    _deserialize_continuous_params,
    _serialize_cond_subparam,
    _deserialize_cond_subparam,
)
from obsidian.parameters import (
    Param_Continuous,
    Param_Categorical,
    ParamSpace,
    Target,
)
from obsidian.campaign import Campaign

import matplotlib
matplotlib.use('Agg')

# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------

CONTINUOUS_PARAMS_UNIFORM = {
    'Temperature': (60, 90, 10),          # (low, high, step) format
    'pH': [6.0, 6.5, 7.0, 7.5, 8.0],     # custom levels, equal weight
}

CONTINUOUS_PARAMS_BIASED = {
    'Temperature': {
        'levels': [60, 70, 80, 90],
        'biases': [0.05, 0.10, 0.35, 0.50],   # strongly biased to upper range
    },
    'pH': {
        'levels': [6.0, 6.5, 7.0, 7.5, 8.0],
        'biases': [0.05, 0.10, 0.20, 0.30, 0.35],
    },
}

CONDITIONAL_SUBPARAMETERS = {
    'Buffer': {
        'Citrate':   {'freq': 0.4, 'Conductivity': ([5, 10, 15], [0.2, 0.5, 0.3])},
        'Phosphate': {'freq': 0.6, 'Conductivity': ([8, 12, 20], [0.3, 0.4, 0.3])},
    }
}

# ParamSpace mirroring the keys used in CONTINUOUS_PARAMS_BIASED
X_SPACE = ParamSpace([
    Param_Continuous('Temperature', 60, 90),
    Param_Continuous('pH', 6.0, 8.0),
])

TARGET = Target(name='Yield', f_transform='Standard', aim='max')


# ---------------------------------------------------------------------------
# 1. Inheritance
# ---------------------------------------------------------------------------

@pytest.mark.fast
def test_is_subclass_of_exp_designer():
    """AdvExpDesigner must be a subclass of ExpDesigner."""
    assert issubclass(AdvExpDesigner, ExpDesigner)


@pytest.mark.fast
def test_isinstance_exp_designer():
    """An AdvExpDesigner instance must satisfy isinstance(..., ExpDesigner)."""
    designer = AdvExpDesigner(
        continuous_params=CONTINUOUS_PARAMS_BIASED,
        X_space=X_SPACE,
        seed=0,
    )
    assert isinstance(designer, ExpDesigner)


# ---------------------------------------------------------------------------
# 2. Backward-compatible construction (no X_space)
# ---------------------------------------------------------------------------

@pytest.mark.fast
def test_standalone_construction_no_x_space():
    """AdvExpDesigner should construct without X_space for standalone use."""
    designer = AdvExpDesigner(
        continuous_params=CONTINUOUS_PARAMS_UNIFORM,
        seed=0,
    )
    assert designer.X_space is None
    assert designer.seed == 0


@pytest.mark.fast
def test_standalone_generate_design():
    """generate_design() works without X_space."""
    designer = AdvExpDesigner(continuous_params=CONTINUOUS_PARAMS_UNIFORM, seed=42)
    design = designer.generate_design(seed=42, n_samples=16)
    assert isinstance(design, pd.DataFrame)
    assert set(CONTINUOUS_PARAMS_UNIFORM.keys()).issubset(design.columns)
    assert len(design) == 16


@pytest.mark.fast
def test_standalone_with_conditional_subparameters():
    """Standalone mode handles categorical conditional subparameters."""
    designer = AdvExpDesigner(
        continuous_params=CONTINUOUS_PARAMS_UNIFORM,
        conditional_subparameters=CONDITIONAL_SUBPARAMETERS,
        seed=7,
    )
    design = designer.generate_design(seed=7, n_samples=20)
    assert 'Buffer' in design.columns
    assert 'Conductivity' in design.columns
    assert set(design['Buffer'].unique()).issubset({'Citrate', 'Phosphate'})


# ---------------------------------------------------------------------------
# 3. initialize() – standalone (requires m_initial)
# ---------------------------------------------------------------------------

@pytest.mark.fast
def test_initialize_standalone_requires_m_initial():
    """Without X_space, initialize() must raise if m_initial is not given."""
    designer = AdvExpDesigner(continuous_params=CONTINUOUS_PARAMS_BIASED)
    with pytest.raises(ValueError, match='m_initial'):
        designer.initialize()


@pytest.mark.fast
def test_initialize_standalone_with_m_initial():
    """Without X_space but with m_initial, initialize() should succeed."""
    designer = AdvExpDesigner(
        continuous_params=CONTINUOUS_PARAMS_BIASED,
        seed=1,
    )
    design = designer.initialize(m_initial=12)
    assert isinstance(design, pd.DataFrame)
    assert len(design) == 12


# ---------------------------------------------------------------------------
# 4. initialize() – with X_space (Campaign-integrated mode)
# ---------------------------------------------------------------------------

@pytest.mark.fast
def test_initialize_with_x_space_default_count():
    """With X_space, initialize() defaults to 2 * n_dim experiments."""
    designer = AdvExpDesigner(
        continuous_params=CONTINUOUS_PARAMS_BIASED,
        X_space=X_SPACE,
        seed=0,
    )
    design = designer.initialize()
    assert len(design) == X_SPACE.n_dim * 2


@pytest.mark.fast
def test_initialize_with_x_space_explicit_count():
    """With X_space and explicit m_initial, correct number of rows returned."""
    designer = AdvExpDesigner(
        continuous_params=CONTINUOUS_PARAMS_BIASED,
        X_space=X_SPACE,
        seed=0,
    )
    design = designer.initialize(m_initial=20)
    assert len(design) == 20
    assert set(CONTINUOUS_PARAMS_BIASED.keys()).issubset(design.columns)


@pytest.mark.fast
def test_initialize_warns_on_missing_columns():
    """initialize() should warn when X_space has params not in the design."""
    extra_space = ParamSpace([
        Param_Continuous('Temperature', 60, 90),
        Param_Continuous('pH', 6.0, 8.0),
        Param_Continuous('Pressure', 1, 5),   # not in continuous_params
    ])
    designer = AdvExpDesigner(
        continuous_params=CONTINUOUS_PARAMS_BIASED,
        X_space=extra_space,
        seed=0,
    )
    with pytest.warns(UserWarning, match='missing columns'):
        designer.initialize(m_initial=10)


# ---------------------------------------------------------------------------
# 5. Bias is actually enforced
# ---------------------------------------------------------------------------

@pytest.mark.fast
def test_upper_range_bias_enforced():
    """Biased Temperature params should produce more samples near 90 than 60."""
    designer = AdvExpDesigner(
        continuous_params=CONTINUOUS_PARAMS_BIASED,
        X_space=X_SPACE,
        seed=42,
    )
    design = designer.initialize(m_initial=200)
    counts = design['Temperature'].value_counts()
    # With biases [0.05, 0.10, 0.35, 0.50], level 90 should dominate over 60
    assert counts.get(90, 0) > counts.get(60, 0), (
        f"Expected more samples at 90 than 60; got counts:\n{counts}"
    )


@pytest.mark.fast
def test_uniform_design_no_dominant_level():
    """Uniform (unbiased) design should distribute Temperature roughly evenly."""
    designer = AdvExpDesigner(
        continuous_params=CONTINUOUS_PARAMS_UNIFORM,
        X_space=ParamSpace([
            Param_Continuous('Temperature', 60, 90),
            Param_Continuous('pH', 6.0, 8.0),
        ]),
        seed=0,
    )
    design = designer.initialize(m_initial=100)
    counts = design['Temperature'].value_counts()
    # No level should be more than 4x the rarest level
    assert counts.max() < 4 * counts.min(), (
        f"Unexpectedly skewed uniform distribution:\n{counts}"
    )


# ---------------------------------------------------------------------------
# 6. Campaign integration
# ---------------------------------------------------------------------------

@pytest.mark.fast
def test_campaign_accepts_adv_exp_designer():
    """Campaign should accept an AdvExpDesigner as the designer= argument."""
    designer = AdvExpDesigner(
        continuous_params=CONTINUOUS_PARAMS_BIASED,
        X_space=X_SPACE,
        seed=0,
    )
    campaign = Campaign(X_SPACE, TARGET, designer=designer)
    assert isinstance(campaign.designer, AdvExpDesigner)
    assert isinstance(campaign.designer, ExpDesigner)


@pytest.mark.fast
def test_campaign_initialize_uses_adv_designer():
    """campaign.initialize() should delegate to AdvExpDesigner and return a DataFrame."""
    designer = AdvExpDesigner(
        continuous_params=CONTINUOUS_PARAMS_BIASED,
        X_space=X_SPACE,
        seed=0,
    )
    campaign = Campaign(X_SPACE, TARGET, designer=designer)
    X0 = campaign.initialize(m_initial=16)
    assert isinstance(X0, pd.DataFrame)
    assert len(X0) == 16
    assert set(CONTINUOUS_PARAMS_BIASED.keys()).issubset(X0.columns)


@pytest.mark.fast
def test_campaign_initialize_bias_preserved():
    """campaign.initialize() via AdvExpDesigner should preserve upper-range bias."""
    designer = AdvExpDesigner(
        continuous_params=CONTINUOUS_PARAMS_BIASED,
        X_space=X_SPACE,
        seed=42,
    )
    campaign = Campaign(X_SPACE, TARGET, designer=designer)
    X0 = campaign.initialize(m_initial=200)
    counts = X0['Temperature'].value_counts()
    assert counts.get(90, 0) > counts.get(60, 0), (
        f"Bias not preserved through campaign.initialize():\n{counts}"
    )


@pytest.mark.fast
def test_campaign_add_data_after_adv_initialize():
    """Data returned by AdvExpDesigner.initialize() can be added to Campaign after appending a response."""
    designer = AdvExpDesigner(
        continuous_params=CONTINUOUS_PARAMS_BIASED,
        X_space=X_SPACE,
        seed=0,
    )
    campaign = Campaign(X_SPACE, TARGET, designer=designer)
    X0 = campaign.initialize(m_initial=10)

    # Simulate a response column
    X0['Yield'] = np.random.uniform(50, 100, size=len(X0))
    campaign.add_data(X0)
    assert campaign.m_exp == 10


# ---------------------------------------------------------------------------
# 7. initialize() method='Optimized' smoke test
# ---------------------------------------------------------------------------

@pytest.mark.slow
def test_initialize_optimized_method():
    """method='Optimized' calls optimize_design() and returns a valid DataFrame."""
    designer = AdvExpDesigner(
        continuous_params=CONTINUOUS_PARAMS_BIASED,
        X_space=X_SPACE,
        seed=0,
    )
    design = designer.initialize(m_initial=12, method='Optimized')
    assert isinstance(design, pd.DataFrame)
    assert len(design) == 12


# ---------------------------------------------------------------------------
# 8. Serialization helpers (continuous_params / conditional_subparameters)
# ---------------------------------------------------------------------------

# Test fixtures for serialization
CONTINUOUS_PARAMS_ALL_FORMATS = {
    "linear_step": (20, 80, 10),
    "geometric": (1, 16, "geometric"),
    "logarithmic": (10, 1000, "logarithmic"),
    "custom_list": [0.5, 1.0, 2.0, 5.0, 10.0],
    "long_tuple": (0, 5, 10, 25, 50, 100, 200, 300),
    "biased_dict": {
        "levels": [1.0, 2.0, 3.0, 5.0, 10.0],
        "biases": [0.1, 0.2, 0.4, 0.2, 0.1],
    },
    "biased_dict_no_biases": {"levels": [10.0, 20.0, 30.0]},
}

CONDITIONAL_SUBPARAMETERS_MULTI = {
    "buffer_type": {
        "citrate": {"freq": 0.5, "pH": ([3.5, 4.0, 5.0], [1 / 3] * 3)},
        "tris":    {"freq": 0.5, "pH": ([7.5, 8.0, 9.0], [1 / 3] * 3)},
    },
    "sugar_type": {
        "no_sugar": {"freq": 0.5, "[Sugar] (%)": ([0], [1.0])},
        "sucrose":  {"freq": 0.5, "[Sugar] (%)": ([2, 4, 6, 8, 10], [1 / 5] * 5)},
    },
}


@pytest.mark.fast
def test_continuous_params_serialization_roundtrip():
    """All continuous_params spec formats must survive serialize/deserialize.

    3-tuple specs (low, high, step) survive as 3-tuples.
    All other formats (list, long tuple, dict) normalize to dict with 'levels' key.
    """
    payload = _serialize_continuous_params(CONTINUOUS_PARAMS_ALL_FORMATS)

    # Verify JSON-safe
    json.dumps(payload)

    # Verify payloads are self-describing dicts
    assert payload["linear_step"] == {"low": 20.0, "high": 80.0, "step": 10.0}
    assert payload["geometric"] == {"low": 1.0, "high": 16.0, "step": "geometric"}
    assert "levels" in payload["custom_list"]
    assert "levels" in payload["biased_dict"]

    # Deserialize and check save + load
    restored = _deserialize_continuous_params(payload)

    # 3-tuples survive as 3-tuples
    assert restored["linear_step"] == (20.0, 80.0, 10.0)
    assert restored["geometric"] == (1.0, 16.0, "geometric")
    assert restored["logarithmic"] == (10.0, 1000.0, "logarithmic")

    # Everything else normalizes to dict form
    assert restored["custom_list"] == {"levels": [0.5, 1.0, 2.0, 5.0, 10.0]}
    assert restored["biased_dict"]["biases"] == [0.1, 0.2, 0.4, 0.2, 0.1]
    assert restored["biased_dict_no_biases"] == {"levels": [10.0, 20.0, 30.0]}


@pytest.mark.fast
@pytest.mark.parametrize("cond_subparam", [
    CONDITIONAL_SUBPARAMETERS,  # single parent param
    CONDITIONAL_SUBPARAMETERS_MULTI,  # multiple parent params
])
def test_conditional_subparameters_serialization_roundtrip(cond_subparam):
    """Conditional subparameters nested structure survives serialize/deserialize.

    (values, weights) tuples at leaves must come back as tuples.
    """
    payload = _serialize_cond_subparam(cond_subparam)

    # Verify JSON-safe
    json.dumps(payload)

    # Deserialize
    restored = _deserialize_cond_subparam(payload)

    # Check structure preserved
    for parent_param, levels in cond_subparam.items():
        assert parent_param in restored
        for level_name, level_info in levels.items():
            assert level_name in restored[parent_param]
            assert restored[parent_param][level_name]["freq"] == pytest.approx(level_info["freq"])

            # Check (values, weights) tuples
            for subparam_name in level_info.keys():
                if subparam_name != "freq":
                    original_values, original_weights = level_info[subparam_name]
                    restored_values, restored_weights = restored[parent_param][level_name][subparam_name]
                    assert list(restored_values) == list(original_values)
                    assert list(restored_weights) == pytest.approx(list(original_weights))


# ---------------------------------------------------------------------------
# 9. AdvExpDesigner state persistence
# ---------------------------------------------------------------------------

@pytest.mark.fast
def test_adv_exp_designer_state_comprehensive_roundtrip():
    """AdvExpDesigner.save_state() / .load_state() must preserve all config.

    Tests:
    - JSON-safe payload
    - All continuous_params formats survive
    - Conditional subparameters with multiple categoricals survive
    - n_category_trials, corr_threshold preserved
    - design_df preserved if present
    - Reloaded designer produces identical designs at same seed
    """
    existing_design = pd.DataFrame({
        "Temperature": [60, 70, 80],
        "pH": [6.5, 7.0, 7.5],
    })

    designer = AdvExpDesigner(
        continuous_params=CONTINUOUS_PARAMS_BIASED,
        conditional_subparameters=CONDITIONAL_SUBPARAMETERS_MULTI,
        design_df=existing_design,
        X_space=X_SPACE,
        seed=7,
        n_category_trials=250,
        corr_threshold=0.005,
    )

    # Save and verify JSON-safe
    state = designer.save_state()
    json.dumps(state)
    assert state["name"] == "AdvExpDesigner"
    assert state["seed"] == 7
    assert state["n_category_trials"] == 250
    assert state["corr_threshold"] == 0.005

    # Load and verify all attributes preserved
    designer2 = AdvExpDesigner.load_state(state)
    assert designer2.seed == 7
    assert designer2.n_category_trials == 250
    assert designer2.corr_threshold == 0.005
    assert "Temperature" in designer2.continuous_params
    assert "buffer_type" in designer2.conditional_subparameters
    assert "sugar_type" in designer2.conditional_subparameters

    # Verify design_df preserved
    pd.testing.assert_frame_equal(
        designer2.design.reset_index(drop=True),
        existing_design.reset_index(drop=True),
        check_dtype=False,
    )

    # Behavioral check: reloaded designer produces identical design at same seed
    design1 = designer.generate_design(seed=42, n_samples=20, optimize_categories=False)
    design2 = designer2.generate_design(seed=42, n_samples=20, optimize_categories=False)
    pd.testing.assert_frame_equal(design1, design2)


@pytest.mark.fast
def test_adv_exp_designer_standalone_load():
    """AdvExpDesigner.load_state() can reconstruct X_space from payload."""
    designer = AdvExpDesigner(
        continuous_params={"A": (0, 1, 0.25)},
        X_space=ParamSpace([Param_Continuous("A", 0, 1)]),
        seed=5,
    )
    state = designer.save_state()

    # Standalone load without X_space override
    designer2 = AdvExpDesigner.load_state(state)
    assert designer2.X_space is not None
    assert list(designer2.X_space.X_names) == ["A"]
    assert designer2.seed == 5


if __name__ == '__main__':
    pytest.main([__file__, '-m', 'fast', '-v'])
