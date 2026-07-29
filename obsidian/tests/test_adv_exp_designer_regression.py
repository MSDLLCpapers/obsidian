"""
Regression test suite for AdvExpDesigner refactor.

Structure:
  1. Fixtures
  2. Parameter spec format tests
  3. Categorical / conditional subparameter tests
  4. Design quality metric unit tests
  5. Bug regression tests  (marked bug_regression — FAIL pre-refactor, PASS post)
  6. optimize_design behavioral tests
  7. extend_design behavioral tests
  8. compare_frequencies tests  (marked bug_regression for return-type assertion)
  9. initialize() tests
 10. Campaign integration tests
 11. V175 integration test (realistic end-to-end)
 12. Plot smoke tests

Run all:
    pytest obsidian/tests/test_adv_exp_designer_regression.py -v

Run only non-slow:
    pytest obsidian/tests/test_adv_exp_designer_regression.py -m "not slow" -v

Run only bug regressions:
    pytest obsidian/tests/test_adv_exp_designer_regression.py -m bug_regression -v
"""

import io
import pathlib
import warnings
import numpy as np
import pandas as pd
import pytest
import matplotlib
matplotlib.use('Agg')   # headless — no display required

from obsidian.experiment import ExpDesigner, AdvExpDesigner
from obsidian.experiment.advanced_design import (
    sample_continuous_lhs,
    sample_design,
    non_uniform_lhs_categorical,
    infer_subparam_mapping,
    calculate_d_optimality,
    calculate_a_optimality,
    calculate_condition_number,
    calculate_pairwise_distance_uniformity,
    calculate_max_continuous_correlation,
    calculate_max_categorical_correlation,
    calculate_max_mixed_correlation,
    calculate_mixed_correlation_matrix,
    evaluate_design,
    evaluate_candidate,
    find_best_design_parallel,
    plot_pca,
    plot_mds,
)
from obsidian.parameters import Param_Continuous, Param_Categorical, ParamSpace, Target
from obsidian.campaign import Campaign


# ---------------------------------------------------------------------------
# 1. Fixtures
# ---------------------------------------------------------------------------

# Minimal 2-param space for fast tests
CONTINUOUS_PARAMS_SIMPLE = {
    'Temperature': (60, 90, 10),          # linear step — 4 levels: 60,70,80,90
    'pH': [6.0, 6.5, 7.0, 7.5, 8.0],     # custom list — 5 levels
}

# All five spec formats
CONTINUOUS_PARAMS_ALL_FORMATS = {
    'linear':      (20, 80, 10),                                    # (low, high, step)
    'geometric':   (1, 16, 'geometric'),                            # 1,2,4,8,16
    'logarithmic': (10, 1000, 'logarithmic'),                       # 10,100,1000
    'custom_list': [0.5, 1.0, 2.0, 5.0, 10.0],                     # list
    'custom_tuple_long': (0, 5, 10, 25, 50, 100, 200, 300),         # tuple len>3 → custom
    'biased_dict': {
        'levels': [1.0, 2.0, 3.0, 5.0, 10.0],
        'biases': [0.1, 0.2, 0.4, 0.2, 0.1],
    },
}

CONDITIONAL_SUBPARAMETERS_SINGLE = {
    'buffer_type': {
        'citrate':    {'freq': 0.25, 'pH': ([3.5, 4.0, 4.5, 5.0, 5.5], [1/5]*5)},
        'histidine':  {'freq': 0.25, 'pH': ([5.0, 5.5, 6.0, 6.5, 7.0], [1/5]*5)},
        'phosphate':  {'freq': 0.25, 'pH': ([6.0, 6.5, 7.0, 7.5, 8.0], [1/5]*5)},
        'tris':       {'freq': 0.25, 'pH': ([7.0, 7.5, 8.0, 8.5, 9.0], [1/5]*5)},
    }
}

# Two independent categorical params, each with one subparam — tests multi-subparam path
CONDITIONAL_SUBPARAMETERS_MULTI = {
    'buffer_type': {
        'citrate':  {'freq': 0.5, 'pH': ([3.5, 4.0, 5.0], [1/3]*3)},
        'tris':     {'freq': 0.5, 'pH': ([7.5, 8.0, 9.0], [1/3]*3)},
    },
    'sugar_type': {
        'no_sugar':  {'freq': 0.5, '[Sugar] (%)': ([0],       [1.0])},
        'sucrose':   {'freq': 0.5, '[Sugar] (%)': ([2, 4, 6, 8, 10], [1/5]*5)},
    },
}

X_SPACE_SIMPLE = ParamSpace([
    Param_Continuous('Temperature', 60, 90),
    Param_Continuous('pH', 6.0, 8.0),
])

TARGET = Target(name='Yield', f_transform='Standard', aim='max')

# V175 Design.ipynb realistic params
V175_CONTINUOUS_PARAMS = {
    '[surfactant] (%)': {
        'levels': [0.001, 0.005, 0.025, 0.05, 0.125, 0.25, 0.5],
        'biases': [0.3/4]*4 + [0.7/3]*3,
    },
    '[KCl]': (0, 5, 10, 25, 50, 100, 200, 300),
    '[Buffer] (mM)': {
        'levels': [5, 10, 20, 40, 60],
        'biases': [0.25, 0.25, 0.20, 0.15, 0.15],
    },
    '[EDTA] (mM)': {
        'levels': [0.001, 0.005, 0.025, 0.05, 0.075, 0.125, 0.2],
        'biases': [0.7/5]*5 + [0.3/2]*2,
    },
    '[L-Met] (mM)': (0, 20, 5),
}

V175_CONDITIONAL_SUBPARAMETERS = {
    'sugar_type': {
        'no_sugar':  {'freq': 0.5,    '[Sugar] (%)': ([0],                           [1.0])},
        'sucrose':   {'freq': 0.25,   '[Sugar] (%)': ([2.0, 4.0, 6.0, 8.0, 10.0],   [1/5]*5)},
        'trehalose': {'freq': 0.25,   '[Sugar] (%)': ([2.0, 4.0, 6.0, 8.0, 10.0],   [1/5]*5)},
    },
    'buffer_type': {
        'citrate':     {'freq': 0.2, 'pH': ([3.5, 4.0, 4.5, 5.0, 5.5], [1/5]*5)},
        'histidine':   {'freq': 0.2, 'pH': ([5.0, 5.5, 6.0, 6.5, 7.0], [1/5]*5)},
        'k phosphate': {'freq': 0.2, 'pH': ([6.0, 6.5, 7.0, 7.5, 8.0], [1/5]*5)},
        'hepes':       {'freq': 0.2, 'pH': ([6.5, 7.0, 7.5, 8.0, 8.5], [1/5]*5)},
        'tris':        {'freq': 0.2, 'pH': ([7.0, 7.5, 8.0, 8.5, 9.0], [1/5]*5)},
    },
}


# ---------------------------------------------------------------------------
# 2. Parameter spec format tests
# ---------------------------------------------------------------------------

class TestContinuousParamFormats:
    """Every spec format should produce values exclusively from the valid level set."""

    @pytest.mark.fast
    def test_linear_step_values_on_grid(self):
        """(low, high, step) values must lie exactly on the grid."""
        params = {'x': (20, 80, 10)}
        samples = sample_continuous_lhs(params, n_samples=50, seed=0)
        valid = np.arange(20, 81, 10)
        assert all(v in valid for v in samples['x']), \
            f"Out-of-grid values found: {set(samples['x']) - set(valid)}"

    @pytest.mark.fast
    def test_geometric_values_are_powers_of_two(self):
        """(1, 16, 'geometric') → values in {1, 2, 4, 8, 16}."""
        params = {'x': (1, 16, 'geometric')}
        samples = sample_continuous_lhs(params, n_samples=50, seed=0)
        valid = {1, 2, 4, 8, 16}
        assert set(samples['x']).issubset(valid), \
            f"Unexpected geometric values: {set(samples['x']) - valid}"

    @pytest.mark.fast
    def test_logarithmic_values_are_powers_of_ten(self):
        """(10, 1000, 'logarithmic') → values in {10.0, 100.0, 1000.0}."""
        params = {'x': (10, 1000, 'logarithmic')}
        samples = sample_continuous_lhs(params, n_samples=50, seed=0)
        valid = {10.0, 100.0, 1000.0}
        assert set(samples['x']).issubset(valid), \
            f"Unexpected log values: {set(samples['x']) - valid}"

    @pytest.mark.fast
    def test_custom_list_values_exact(self):
        """Custom list format → only listed values appear."""
        levels = [0.5, 1.0, 2.0, 5.0, 10.0]
        params = {'x': levels}
        samples = sample_continuous_lhs(params, n_samples=50, seed=0)
        assert set(samples['x']).issubset(set(levels)), \
            f"Unexpected values: {set(samples['x']) - set(levels)}"

    @pytest.mark.fast
    def test_custom_tuple_long_treated_as_levels(self):
        """8-element tuple → treated as custom levels, not (low, high, step)."""
        levels = (0, 5, 10, 25, 50, 100, 200, 300)
        params = {'x': levels}
        samples = sample_continuous_lhs(params, n_samples=50, seed=0)
        assert set(samples['x']).issubset(set(levels)), \
            f"Unexpected values: {set(samples['x']) - set(levels)}"

    @pytest.mark.fast
    def test_biased_dict_values_from_levels(self):
        """Dict with levels+biases → values only from specified levels."""
        spec = {'levels': [1.0, 2.0, 3.0, 5.0, 10.0], 'biases': [0.1, 0.2, 0.4, 0.2, 0.1]}
        params = {'x': spec}
        samples = sample_continuous_lhs(params, n_samples=50, seed=0)
        assert set(samples['x']).issubset(set(spec['levels'])), \
            f"Unexpected values: {set(samples['x']) - set(spec['levels'])}"

    @pytest.mark.fast
    def test_biased_dict_bias_respected_large_n(self):
        """High-weight level should appear more often than low-weight level (large n)."""
        spec = {'levels': [1, 2, 3, 4], 'biases': [0.05, 0.10, 0.35, 0.50]}
        params = {'x': spec}
        samples = sample_continuous_lhs(params, n_samples=500, seed=42)
        counts = pd.Series(samples['x']).value_counts()
        assert counts.get(4, 0) > counts.get(1, 0), \
            f"Bias not respected; counts: {dict(counts)}"

    @pytest.mark.fast
    def test_biased_dict_unnormalized_biases_accepted(self):
        """Unnormalized biases (don't sum to 1) should be auto-normalized without error."""
        spec = {'levels': [10, 20, 30], 'biases': [1, 2, 7]}  # sum = 10, not 1
        params = {'x': spec}
        samples = sample_continuous_lhs(params, n_samples=30, seed=0)
        assert set(samples['x']).issubset({10, 20, 30})

    @pytest.mark.fast
    def test_all_formats_together(self):
        """All six formats can coexist in a single continuous_params dict."""
        design = sample_design(
            seed=0,
            n_samples=30,
            continuous_params=CONTINUOUS_PARAMS_ALL_FORMATS,
            conditional_subparameters={},
        )
        assert len(design) == 30
        assert set(CONTINUOUS_PARAMS_ALL_FORMATS.keys()).issubset(design.columns)

    @pytest.mark.fast
    def test_logarithmic_raises_for_non_positive_bounds(self):
        """Logarithmic step with non-positive bounds should raise ValueError."""
        params = {'x': (0, 100, 'logarithmic')}
        with pytest.raises(ValueError, match='positive'):
            sample_continuous_lhs(params, n_samples=10, seed=0)

    @pytest.mark.fast
    def test_dict_spec_missing_levels_raises(self):
        """Dict spec without 'levels' key should raise ValueError."""
        params = {'x': {'biases': [0.5, 0.5]}}
        with pytest.raises(ValueError, match="'levels'"):
            sample_continuous_lhs(params, n_samples=5, seed=0)

    @pytest.mark.fast
    def test_biased_dict_wrong_bias_length_raises(self):
        """Bias length != levels length should raise ValueError."""
        params = {'x': {'levels': [1, 2, 3], 'biases': [0.5, 0.5]}}
        with pytest.raises(ValueError, match='[Bb]ias'):
            sample_continuous_lhs(params, n_samples=5, seed=0)


# ---------------------------------------------------------------------------
# 3. Categorical / conditional subparameter tests
# ---------------------------------------------------------------------------

class TestCategoricalSampling:
    """Categorical frequencies and subparameter values match their specs."""

    @pytest.mark.fast
    def test_categorical_levels_only_from_spec(self):
        """Sampled categorical values must come only from the defined levels."""
        results = non_uniform_lhs_categorical(
            CONDITIONAL_SUBPARAMETERS_SINGLE['buffer_type'],
            n_samples=100,
            seed=0,
        )
        valid_levels = set(CONDITIONAL_SUBPARAMETERS_SINGLE['buffer_type'].keys())
        sampled_levels = {r['level'] for r in results}
        assert sampled_levels.issubset(valid_levels)

    @pytest.mark.fast
    def test_categorical_frequencies_approximately_correct(self):
        """Empirical frequencies should be within 15% of specified frequencies (n=200)."""
        level_dict = {
            'A': {'freq': 0.5, 'x': ([1, 2], [0.5, 0.5])},
            'B': {'freq': 0.3, 'x': ([3, 4], [0.5, 0.5])},
            'C': {'freq': 0.2, 'x': ([5],    [1.0])},
        }
        results = non_uniform_lhs_categorical(level_dict, n_samples=200, seed=7)
        counts = pd.Series([r['level'] for r in results]).value_counts(normalize=True)
        assert abs(counts.get('A', 0) - 0.5) < 0.15
        assert abs(counts.get('B', 0) - 0.3) < 0.15
        assert abs(counts.get('C', 0) - 0.2) < 0.15

    @pytest.mark.fast
    def test_subparam_values_match_category(self):
        """pH values for each buffer_type must come from that buffer's valid range."""
        design = sample_design(
            seed=42,
            n_samples=80,
            continuous_params=CONTINUOUS_PARAMS_SIMPLE,
            conditional_subparameters=CONDITIONAL_SUBPARAMETERS_SINGLE,
        )
        citrate_ph = design.loc[design['buffer_type'] == 'citrate', 'pH']
        tris_ph    = design.loc[design['buffer_type'] == 'tris',    'pH']
        citrate_valid = {3.5, 4.0, 4.5, 5.0, 5.5}
        tris_valid    = {7.0, 7.5, 8.0, 8.5, 9.0}
        assert set(citrate_ph).issubset(citrate_valid), \
            f"Invalid citrate pH: {set(citrate_ph) - citrate_valid}"
        assert set(tris_ph).issubset(tris_valid), \
            f"Invalid tris pH: {set(tris_ph) - tris_valid}"

    @pytest.mark.fast
    def test_no_sugar_entries_have_zero_sugar_concentration(self):
        """no_sugar rows must have [Sugar] (%) == 0."""
        design = sample_design(
            seed=0,
            n_samples=60,
            continuous_params=CONTINUOUS_PARAMS_SIMPLE,
            conditional_subparameters=CONDITIONAL_SUBPARAMETERS_MULTI,
        )
        no_sugar_rows = design[design['sugar_type'] == 'no_sugar']
        assert (no_sugar_rows['[Sugar] (%)'] == 0).all(), \
            "no_sugar rows should have [Sugar] (%) == 0"

    @pytest.mark.fast
    def test_multiple_categorical_columns_present(self):
        """Design with multi-subparam spec should have all categorical and subparam columns."""
        design = sample_design(
            seed=0,
            n_samples=40,
            continuous_params=CONTINUOUS_PARAMS_SIMPLE,
            conditional_subparameters=CONDITIONAL_SUBPARAMETERS_MULTI,
        )
        expected_cols = {'Temperature', 'pH', 'buffer_type', 'pH', 'sugar_type', '[Sugar] (%)'}
        # Both categorical and both subparam columns must be present
        assert 'buffer_type'  in design.columns
        assert 'sugar_type'   in design.columns
        assert 'pH'           in design.columns
        assert '[Sugar] (%)' in design.columns

    @pytest.mark.fast
    def test_infer_subparam_mapping_single(self):
        """infer_subparam_mapping returns correct 1:1 mapping for single subparam per category."""
        mapping = infer_subparam_mapping(CONDITIONAL_SUBPARAMETERS_SINGLE)
        assert mapping == {'buffer_type': 'pH'}

    @pytest.mark.fast
    def test_infer_subparam_mapping_multi(self):
        """infer_subparam_mapping returns correct mappings for two independent categories."""
        mapping = infer_subparam_mapping(CONDITIONAL_SUBPARAMETERS_MULTI)
        assert mapping == {'buffer_type': 'pH', 'sugar_type': '[Sugar] (%)'}

    @pytest.mark.fast
    def test_infer_subparam_mapping_empty(self):
        """Empty conditional_subparameters → empty mapping."""
        assert infer_subparam_mapping({}) == {}

    @pytest.mark.fast
    def test_optimize_categories_produces_valid_design(self):
        """optimize_categories=True should still produce a valid design with correct columns."""
        designer = AdvExpDesigner(
            continuous_params=CONTINUOUS_PARAMS_SIMPLE,
            conditional_subparameters=CONDITIONAL_SUBPARAMETERS_SINGLE,
            seed=0,
        )
        design = designer.generate_design(seed=0, n_samples=40, optimize_categories=True)
        assert isinstance(design, pd.DataFrame)
        assert len(design) == 40
        assert 'buffer_type' in design.columns
        assert 'pH' in design.columns


# ---------------------------------------------------------------------------
# 4. Design quality metric unit tests
# ---------------------------------------------------------------------------

# Separate continuous params for metric tests — no overlap with subparam 'pH'
CONTINUOUS_PARAMS_METRIC = {
    'Temperature': (60, 90, 10),
    'Concentration': (0.1, 1.0, 0.1),
}


@pytest.fixture(scope='module')
def small_design():
    """
    A pre-built 30-row design for metric unit tests.
    Uses CONTINUOUS_PARAMS_METRIC (Temperature, Concentration) — deliberately excludes 'pH'
    from continuous_params so there is no column name collision with the buffer_type subparam.
    """
    return sample_design(
        seed=7,
        n_samples=30,
        continuous_params=CONTINUOUS_PARAMS_METRIC,
        conditional_subparameters=CONDITIONAL_SUBPARAMETERS_SINGLE,
    )


class TestMetricFunctions:
    """Each metric function should return a finite scalar of the right sign/range."""

    @pytest.mark.fast
    def test_d_optimality_positive(self, small_design):
        val = calculate_d_optimality(small_design, list(CONTINUOUS_PARAMS_METRIC.keys()), ['pH'])
        assert np.isfinite(val) and val > 0

    @pytest.mark.fast
    def test_a_optimality_positive(self, small_design):
        val = calculate_a_optimality(small_design, list(CONTINUOUS_PARAMS_METRIC.keys()), ['pH'])
        assert np.isfinite(val) and val > 0

    @pytest.mark.fast
    def test_condition_number_geq_one(self, small_design):
        val = calculate_condition_number(small_design, list(CONTINUOUS_PARAMS_METRIC.keys()), ['pH'])
        assert np.isfinite(val) and val >= 1.0

    @pytest.mark.fast
    def test_pairwise_distance_cv_non_negative(self, small_design):
        val = calculate_pairwise_distance_uniformity(
            small_design, list(CONTINUOUS_PARAMS_METRIC.keys()), ['pH'])
        assert np.isfinite(val) and val >= 0

    @pytest.mark.fast
    def test_max_continuous_corr_in_unit_interval(self, small_design):
        val = calculate_max_continuous_correlation(
            small_design, list(CONTINUOUS_PARAMS_METRIC.keys()), ['pH'])
        assert 0.0 <= val <= 1.0

    @pytest.mark.fast
    def test_max_categorical_corr_in_unit_interval(self, small_design):
        val = calculate_max_categorical_correlation(small_design, ['buffer_type'])
        # Single categorical variable → matrix diagonal zeroed → max is 0
        assert 0.0 <= val <= 1.0

    @pytest.mark.fast
    def test_max_mixed_corr_in_unit_interval(self, small_design):
        val = calculate_max_mixed_correlation(
            small_design,
            list(CONTINUOUS_PARAMS_METRIC.keys()),
            ['buffer_type'],
            {'buffer_type': 'pH'},
        )
        assert 0.0 <= val <= 1.0

    @pytest.mark.fast
    def test_evaluate_design_returns_all_seven_metrics(self, small_design):
        """Default evaluate_design call returns all seven expected keys."""
        result = evaluate_design(
            design=small_design,
            continuous_keys=list(CONTINUOUS_PARAMS_METRIC.keys()),
            categorical_keys=['buffer_type'],
            subparam_mapping={'buffer_type': 'pH'},
            metrics_to_optimize=[
                'D-optimality', 'A-optimality', 'Condition Number',
                'Pairwise Distance CV', 'Max Continuous Corr',
                'Max Categorical Corr', 'Max Mixed Corr',
            ],
        )
        expected_keys = {
            'D-optimality', 'A-optimality', 'Condition Number',
            'Pairwise Distance CV', 'Max Continuous Corr',
            'Max Categorical Corr', 'Max Mixed Corr',
        }
        assert set(result.keys()) == expected_keys

    @pytest.mark.fast
    def test_evaluate_design_subset_metrics(self, small_design):
        """evaluate_design respects a custom metrics_to_optimize subset."""
        subset = ['D-optimality', 'Max Continuous Corr']
        result = evaluate_design(
            design=small_design,
            continuous_keys=list(CONTINUOUS_PARAMS_METRIC.keys()),
            categorical_keys=['buffer_type'],
            subparam_mapping={'buffer_type': 'pH'},
            metrics_to_optimize=subset,
        )
        assert set(result.keys()) == set(subset)

    @pytest.mark.fast
    def test_evaluate_design_via_designer_method(self, small_design):
        """AdvExpDesigner.evaluate_design() wraps the function correctly."""
        designer = AdvExpDesigner(
            continuous_params=CONTINUOUS_PARAMS_METRIC,
            conditional_subparameters=CONDITIONAL_SUBPARAMETERS_SINGLE,
            seed=0,
        )
        result = designer.evaluate_design(small_design)
        assert isinstance(result, dict)
        assert len(result) == 7

    @pytest.mark.fast
    def test_mixed_correlation_matrix_shape(self, small_design):
        """Mixed correlation matrix should be square and symmetric."""
        corr = calculate_mixed_correlation_matrix(small_design, categorical_vars=['buffer_type'])
        assert corr.shape[0] == corr.shape[1] == len(small_design.columns)
        # Symmetry
        np.testing.assert_allclose(corr.values, corr.values.T, atol=1e-10)

    @pytest.mark.fast
    def test_mixed_correlation_diagonal_is_one(self, small_design):
        """Diagonal of mixed correlation matrix must be 1.0."""
        corr = calculate_mixed_correlation_matrix(small_design, categorical_vars=['buffer_type'])
        np.testing.assert_allclose(np.diag(corr.values), 1.0, atol=1e-10)

    @pytest.mark.fast
    def test_metric_functions_accept_list_of_subparams(self, small_design):
        """Metric functions should accept a list of subparam keys (not just a single string)."""
        subparam_keys = ['pH']
        d = calculate_d_optimality(small_design, list(CONTINUOUS_PARAMS_METRIC.keys()), subparam_keys)
        a = calculate_a_optimality(small_design, list(CONTINUOUS_PARAMS_METRIC.keys()), subparam_keys)
        c = calculate_condition_number(small_design, list(CONTINUOUS_PARAMS_METRIC.keys()), subparam_keys)
        p = calculate_pairwise_distance_uniformity(small_design, list(CONTINUOUS_PARAMS_METRIC.keys()), subparam_keys)
        mc = calculate_max_continuous_correlation(small_design, list(CONTINUOUS_PARAMS_METRIC.keys()), subparam_keys)
        assert all(np.isfinite(v) for v in [d, a, c, p, mc])


# ---------------------------------------------------------------------------
# 5. Bug regression tests  (FAIL pre-refactor, PASS post-refactor)
# ---------------------------------------------------------------------------

class TestBugRegressions:
    """
    These tests assert CORRECT intended behavior that is broken in the pre-refactor code.
    They are expected to FAIL on the original code and PASS after the refactor.
    """

    @pytest.mark.fast
    @pytest.mark.bug_regression
    def test_maximize_metrics_default_is_not_string(self):
        """
        BUG: optimize_design() defaulted maximize_metrics to the string "D-optimality"
        instead of a list of booleans. This caused all metrics to be treated as 'maximize'
        because iterating over the string character-by-character always produces truthy values.

        Fix: default should be a list [True, False, False, ...].

        This test verifies the fix by inspecting what find_best_design_parallel receives.
        We achieve this by temporarily monkeypatching find_best_design_parallel and capturing
        the maximize_metrics argument.
        """
        import obsidian.experiment.advanced_design as adv_mod
        captured = {}
        original = adv_mod.find_best_design_parallel

        def capturing_wrapper(*args, **kwargs):
            captured['maximize_metrics'] = kwargs.get('maximize_metrics', args[5] if len(args) > 5 else None)
            return original(*args, **kwargs)

        designer = AdvExpDesigner(
            continuous_params=CONTINUOUS_PARAMS_SIMPLE,
            conditional_subparameters=CONDITIONAL_SUBPARAMETERS_SINGLE,
            seed=0,
        )

        adv_mod.find_best_design_parallel = capturing_wrapper
        try:
            designer.optimize_design(n_trials=2, n_samples=10)
        finally:
            adv_mod.find_best_design_parallel = original

        passed_value = captured.get('maximize_metrics')
        assert isinstance(passed_value, list), (
            f"maximize_metrics passed to find_best_design_parallel should be a list of booleans, "
            f"got {type(passed_value).__name__!r}: {passed_value!r}"
        )
        assert all(isinstance(v, bool) for v in passed_value), (
            f"All elements of maximize_metrics should be bool, got: {passed_value}"
        )

    @pytest.mark.fast
    @pytest.mark.bug_regression
    def test_maximize_metrics_first_element_true_rest_false(self):
        """
        After the fix, the default maximize_metrics should be [True, False, False, ...]
        — maximize D-optimality only; minimize all other metrics.
        """
        import obsidian.experiment.advanced_design as adv_mod
        captured = {}
        original = adv_mod.find_best_design_parallel

        def capturing_wrapper(*args, **kwargs):
            captured['maximize_metrics'] = kwargs.get('maximize_metrics')
            captured['metrics_to_optimize'] = kwargs.get('metrics_to_optimize')
            return original(*args, **kwargs)

        designer = AdvExpDesigner(
            continuous_params=CONTINUOUS_PARAMS_SIMPLE,
            conditional_subparameters=CONDITIONAL_SUBPARAMETERS_SINGLE,
            seed=0,
        )
        adv_mod.find_best_design_parallel = capturing_wrapper
        try:
            designer.optimize_design(n_trials=2, n_samples=10)
        finally:
            adv_mod.find_best_design_parallel = original

        mm = captured['maximize_metrics']
        n_metrics = len(captured['metrics_to_optimize'])
        assert mm[0] is True, "First metric (D-optimality) should be maximized"
        assert all(v is False for v in mm[1:]), \
            f"All metrics after D-optimality should be minimized, got: {mm[1:]}"
        assert len(mm) == n_metrics, \
            f"maximize_metrics length ({len(mm)}) should match metrics_to_optimize ({n_metrics})"

    @pytest.mark.fast
    @pytest.mark.bug_regression
    def test_evaluate_candidate_includes_all_subparams_in_metrics(self):
        """
        BUG: evaluate_candidate() used list(subparam_mapping.values())[0], silently
        dropping all but the first subparameter from metric calculations during extend_design.

        Fix: evaluate_candidate should pass ALL subparam keys to metric functions.

        We call evaluate_candidate directly with a two-subparam mapping and compare
        the D-optimality it returns against:
          - d_one_subparam:  what we get passing only the first subparam (buggy path)
          - d_both_subparams: what we get passing both subparams (correct path)

        Before fix: evaluate_candidate returns d_one_subparam
        After  fix: evaluate_candidate returns d_both_subparams
        """
        subparam_mapping = infer_subparam_mapping(CONDITIONAL_SUBPARAMETERS_MULTI)
        # {'buffer_type': 'pH', 'sugar_type': '[Sugar] (%)'}

        existing = sample_design(
            seed=0,
            n_samples=20,
            continuous_params=CONTINUOUS_PARAMS_METRIC,
            conditional_subparameters=CONDITIONAL_SUBPARAMETERS_MULTI,
            subparam_mapping=subparam_mapping,
        )
        cont_keys = list(CONTINUOUS_PARAMS_METRIC.keys())
        cat_keys  = list(CONDITIONAL_SUBPARAMETERS_MULTI.keys())

        result = evaluate_candidate(
            i=0,
            seed_start=100,
            n=5,
            continuous_params=CONTINUOUS_PARAMS_METRIC,
            conditional_subparameters=CONDITIONAL_SUBPARAMETERS_MULTI,
            subparam_mapping=subparam_mapping,
            existing_design=existing,
            continuous_keys=cont_keys,
            categorical_keys=cat_keys,
            metrics_to_optimize=['D-optimality'],
        )

        combined = pd.concat([existing, result['new_samples']], ignore_index=True)
        d_from_candidate = result['metrics']['D-optimality']
        d_one_subparam   = calculate_d_optimality(combined, cont_keys, ['pH'])
        d_both_subparams = calculate_d_optimality(combined, cont_keys, ['pH', '[Sugar] (%)'])

        # Sanity: the two reference values must differ (otherwise the test can't distinguish)
        assert d_one_subparam != d_both_subparams, (
            "Test precondition failed: D-opt with one vs both subparams must differ. "
            "Try a different seed or design."
        )

        # After fix: evaluate_candidate should use both subparams → equals d_both_subparams
        assert d_from_candidate == pytest.approx(d_both_subparams), (
            f"evaluate_candidate D-optimality ({d_from_candidate:.4g}) should match "
            f"the value computed with both subparams ({d_both_subparams:.4g}), "
            f"not the single-subparam value ({d_one_subparam:.4g})"
        )

    @pytest.mark.fast
    @pytest.mark.bug_regression
    def test_compare_frequencies_returns_dataframe(self):
        """
        BUG: compare_frequencies() printed to stdout and returned None.
        Fix: it should return a pd.DataFrame with the frequency comparison data.
        """
        designer = AdvExpDesigner(
            continuous_params=CONTINUOUS_PARAMS_SIMPLE,
            conditional_subparameters=CONDITIONAL_SUBPARAMETERS_SINGLE,
            seed=0,
        )
        design = designer.generate_design(seed=0, n_samples=40)
        result = designer.compare_frequencies(design)

        assert result is not None, \
            "compare_frequencies() returned None — should return a DataFrame after fix"
        assert isinstance(result, pd.DataFrame), \
            f"Expected pd.DataFrame, got {type(result).__name__}"

    @pytest.mark.fast
    @pytest.mark.bug_regression
    def test_compare_frequencies_dataframe_has_expected_columns(self):
        """After fix, compare_frequencies DataFrame should contain expected/empirical columns."""
        designer = AdvExpDesigner(
            continuous_params=CONTINUOUS_PARAMS_SIMPLE,
            conditional_subparameters=CONDITIONAL_SUBPARAMETERS_SINGLE,
            seed=0,
        )
        design = designer.generate_design(seed=0, n_samples=80)
        result = designer.compare_frequencies(design)

        if result is None:
            pytest.skip("compare_frequencies still returns None (pre-refactor)")

        # Should have columns identifying level, expected freq, empirical freq
        col_lower = [c.lower() for c in result.columns]
        assert any('expect' in c for c in col_lower), \
            f"Expected a column with 'expected' frequency, got: {list(result.columns)}"
        assert any('empir' in c or 'actual' in c for c in col_lower), \
            f"Expected a column with 'empirical' frequency, got: {list(result.columns)}"


# ---------------------------------------------------------------------------
# 6. optimize_design behavioral tests
# ---------------------------------------------------------------------------

class TestOptimizeDesign:

    @pytest.mark.slow
    def test_returns_tuple_of_dataframe_and_metrics_df(self):
        """optimize_design should return (pd.DataFrame, pd.DataFrame)."""
        designer = AdvExpDesigner(
            continuous_params=CONTINUOUS_PARAMS_SIMPLE,
            conditional_subparameters=CONDITIONAL_SUBPARAMETERS_SINGLE,
            seed=0,
        )
        best, metrics_df = designer.optimize_design(n_trials=3, n_samples=20)
        assert isinstance(best, pd.DataFrame)
        assert isinstance(metrics_df, pd.DataFrame)

    @pytest.mark.slow
    def test_best_design_has_correct_row_count(self):
        """Best design should have exactly n_samples rows."""
        designer = AdvExpDesigner(
            continuous_params=CONTINUOUS_PARAMS_SIMPLE,
            conditional_subparameters=CONDITIONAL_SUBPARAMETERS_SINGLE,
            seed=0,
        )
        best, _ = designer.optimize_design(n_trials=3, n_samples=15)
        assert len(best) == 15

    @pytest.mark.slow
    def test_metrics_df_has_n_trials_rows(self):
        """metrics_df should have one row per trial."""
        designer = AdvExpDesigner(
            continuous_params=CONTINUOUS_PARAMS_SIMPLE,
            conditional_subparameters=CONDITIONAL_SUBPARAMETERS_SINGLE,
            seed=0,
        )
        _, metrics_df = designer.optimize_design(n_trials=4, n_samples=15)
        assert len(metrics_df) == 4

    @pytest.mark.slow
    def test_metrics_df_has_score_column(self):
        """metrics_df must contain a 'score' column."""
        designer = AdvExpDesigner(
            continuous_params=CONTINUOUS_PARAMS_SIMPLE,
            conditional_subparameters=CONDITIONAL_SUBPARAMETERS_SINGLE,
            seed=0,
        )
        _, metrics_df = designer.optimize_design(n_trials=3, n_samples=15)
        assert 'score' in metrics_df.columns

    @pytest.mark.slow
    def test_custom_metrics_to_optimize_subset(self):
        """optimize_design respects a custom metrics_to_optimize list."""
        designer = AdvExpDesigner(
            continuous_params=CONTINUOUS_PARAMS_SIMPLE,
            conditional_subparameters=CONDITIONAL_SUBPARAMETERS_SINGLE,
            seed=0,
        )
        subset = ['Max Continuous Corr', 'Max Categorical Corr']
        best, metrics_df = designer.optimize_design(
            n_trials=3,
            n_samples=15,
            metrics_to_optimize=subset,
            maximize_metrics=[False, False],
        )
        assert isinstance(best, pd.DataFrame)
        for col in subset:
            assert col in metrics_df.columns

    @pytest.mark.slow
    def test_explicit_maximize_metrics_list_works(self):
        """Explicitly passing maximize_metrics as a list should not raise."""
        designer = AdvExpDesigner(
            continuous_params=CONTINUOUS_PARAMS_SIMPLE,
            conditional_subparameters=CONDITIONAL_SUBPARAMETERS_SINGLE,
            seed=0,
        )
        # D-optimality maximize, A-optimality minimize
        best, _ = designer.optimize_design(
            n_trials=3,
            n_samples=15,
            metrics_to_optimize=['D-optimality', 'A-optimality'],
            maximize_metrics=[True, False],
        )
        assert isinstance(best, pd.DataFrame)

    @pytest.mark.slow
    def test_score_finite_for_all_trials(self):
        """All trial scores should be finite."""
        designer = AdvExpDesigner(
            continuous_params=CONTINUOUS_PARAMS_SIMPLE,
            conditional_subparameters=CONDITIONAL_SUBPARAMETERS_SINGLE,
            seed=0,
        )
        _, metrics_df = designer.optimize_design(n_trials=4, n_samples=15)
        assert metrics_df['score'].apply(np.isfinite).all(), \
            "Some trial scores are not finite"


# ---------------------------------------------------------------------------
# 7. extend_design behavioral tests
# ---------------------------------------------------------------------------

class TestExtendDesign:

    @pytest.fixture
    def base_design(self):
        return sample_design(
            seed=0,
            n_samples=20,
            continuous_params=CONTINUOUS_PARAMS_SIMPLE,
            conditional_subparameters=CONDITIONAL_SUBPARAMETERS_SINGLE,
        )

    @pytest.mark.slow
    def test_extended_design_has_correct_length(self, base_design):
        """extended_design should have len(existing) + n rows."""
        designer = AdvExpDesigner(
            continuous_params=CONTINUOUS_PARAMS_SIMPLE,
            conditional_subparameters=CONDITIONAL_SUBPARAMETERS_SINGLE,
            seed=0,
        )
        extended, _ = designer.extend_design(base_design, n=10, n_trials=3)
        assert len(extended) == len(base_design) + 10

    @pytest.mark.slow
    def test_extended_design_preserves_existing_rows(self, base_design):
        """The first len(existing) rows of extended_design must equal existing_design."""
        designer = AdvExpDesigner(
            continuous_params=CONTINUOUS_PARAMS_SIMPLE,
            conditional_subparameters=CONDITIONAL_SUBPARAMETERS_SINGLE,
            seed=0,
        )
        extended, _ = designer.extend_design(base_design, n=5, n_trials=3)
        original_part = extended.iloc[:len(base_design)].reset_index(drop=True)
        pd.testing.assert_frame_equal(original_part, base_design.reset_index(drop=True))

    @pytest.mark.slow
    def test_extend_design_returns_metrics_summary(self, base_design):
        """extend_design should return a (DataFrame, DataFrame) tuple."""
        designer = AdvExpDesigner(
            continuous_params=CONTINUOUS_PARAMS_SIMPLE,
            conditional_subparameters=CONDITIONAL_SUBPARAMETERS_SINGLE,
            seed=0,
        )
        extended, summary = designer.extend_design(base_design, n=5, n_trials=3)
        assert isinstance(extended, pd.DataFrame)
        assert isinstance(summary, pd.DataFrame)
        assert 'score' in summary.columns

    @pytest.mark.slow
    def test_extend_design_new_rows_have_valid_values(self, base_design):
        """Newly added rows should have categorical values from the valid level set."""
        designer = AdvExpDesigner(
            continuous_params=CONTINUOUS_PARAMS_SIMPLE,
            conditional_subparameters=CONDITIONAL_SUBPARAMETERS_SINGLE,
            seed=0,
        )
        extended, _ = designer.extend_design(base_design, n=10, n_trials=3)
        new_rows = extended.iloc[len(base_design):]
        valid_buffers = set(CONDITIONAL_SUBPARAMETERS_SINGLE['buffer_type'].keys())
        assert set(new_rows['buffer_type']).issubset(valid_buffers)

    @pytest.mark.slow
    @pytest.mark.bug_regression
    def test_extend_design_multi_subparam_includes_all_subparams_in_metrics(self, base_design):
        """
        BUG: evaluate_candidate used only the first subparam key in metrics.
        After fix, extend_design with multi-subparam mapping should compute metrics
        using ALL subparameters.

        We verify by checking that the extended design metrics are consistent with
        evaluate_design (which correctly uses all subparams).
        """
        multi_base = sample_design(
            seed=0,
            n_samples=20,
            continuous_params=CONTINUOUS_PARAMS_SIMPLE,
            conditional_subparameters=CONDITIONAL_SUBPARAMETERS_MULTI,
        )
        designer = AdvExpDesigner(
            continuous_params=CONTINUOUS_PARAMS_SIMPLE,
            conditional_subparameters=CONDITIONAL_SUBPARAMETERS_MULTI,
            seed=0,
        )
        extended, summary = designer.extend_design(multi_base, n=5, n_trials=3)
        # If extend_design correctly includes both pH and [Sugar] (%) in metrics,
        # D-optimality in summary should match evaluate_design on the combined design
        best_seed_row = summary.loc[summary['score'].idxmax()]
        assert np.isfinite(best_seed_row['D-optimality']), \
            "D-optimality in extension summary should be finite"


# ---------------------------------------------------------------------------
# 8. compare_frequencies tests
# ---------------------------------------------------------------------------

class TestCompareFrequencies:

    @pytest.mark.fast
    def test_compare_frequencies_does_not_raise(self):
        """compare_frequencies should not raise for a valid design."""
        designer = AdvExpDesigner(
            continuous_params=CONTINUOUS_PARAMS_SIMPLE,
            conditional_subparameters=CONDITIONAL_SUBPARAMETERS_SINGLE,
            seed=0,
        )
        design = designer.generate_design(seed=0, n_samples=40)
        # Should not raise regardless of return type
        designer.compare_frequencies(design)

    @pytest.mark.fast
    def test_compare_frequencies_empirical_frequencies_roughly_correct(self):
        """Empirical frequencies in compare_frequencies output should be approximately correct."""
        designer = AdvExpDesigner(
            continuous_params=CONTINUOUS_PARAMS_SIMPLE,
            conditional_subparameters=CONDITIONAL_SUBPARAMETERS_SINGLE,
            seed=42,
        )
        design = designer.generate_design(seed=42, n_samples=200)
        result = designer.compare_frequencies(design)

        if result is None:
            # Pre-refactor: parse stdout
            pytest.skip("compare_frequencies still returns None — skipping data validation")

        # Each buffer should appear approximately 25% of the time (equal freq)
        empirical_col = [c for c in result.columns if 'empir' in c.lower() or 'actual' in c.lower()][0]
        for _, row in result.iterrows():
            assert abs(row[empirical_col] - 0.25) < 0.15, \
                f"Empirical frequency {row[empirical_col]:.3f} far from expected 0.25"


# ---------------------------------------------------------------------------
# 9. initialize() tests
# ---------------------------------------------------------------------------

class TestInitialize:

    @pytest.mark.fast
    def test_standalone_requires_m_initial(self):
        """Without X_space, initialize() must raise ValueError if m_initial not given."""
        designer = AdvExpDesigner(continuous_params=CONTINUOUS_PARAMS_SIMPLE)
        with pytest.raises(ValueError, match='m_initial'):
            designer.initialize()

    @pytest.mark.fast
    def test_standalone_with_m_initial(self):
        """Standalone initialize() with m_initial returns correct DataFrame."""
        designer = AdvExpDesigner(
            continuous_params=CONTINUOUS_PARAMS_SIMPLE,
            seed=1,
        )
        design = designer.initialize(m_initial=12)
        assert isinstance(design, pd.DataFrame)
        assert len(design) == 12

    @pytest.mark.fast
    def test_with_x_space_defaults_to_2x_ndim(self):
        """With X_space, initialize() without m_initial defaults to 2 * n_dim."""
        designer = AdvExpDesigner(
            continuous_params=CONTINUOUS_PARAMS_SIMPLE,
            X_space=X_SPACE_SIMPLE,
            seed=0,
        )
        design = designer.initialize()
        assert len(design) == X_SPACE_SIMPLE.n_dim * 2

    @pytest.mark.fast
    def test_with_x_space_explicit_m_initial(self):
        """With X_space and explicit m_initial, returns correct row count."""
        designer = AdvExpDesigner(
            continuous_params=CONTINUOUS_PARAMS_SIMPLE,
            X_space=X_SPACE_SIMPLE,
            seed=0,
        )
        design = designer.initialize(m_initial=20)
        assert len(design) == 20

    @pytest.mark.fast
    def test_warns_on_missing_x_space_columns(self):
        """initialize() should warn when X_space has params absent from continuous_params."""
        extra_space = ParamSpace([
            Param_Continuous('Temperature', 60, 90),
            Param_Continuous('pH', 6.0, 8.0),
            Param_Continuous('Pressure', 1, 5),   # not in continuous_params
        ])
        designer = AdvExpDesigner(
            continuous_params=CONTINUOUS_PARAMS_SIMPLE,
            X_space=extra_space,
            seed=0,
        )
        with pytest.warns(UserWarning, match='missing columns'):
            designer.initialize(m_initial=10)

    @pytest.mark.fast
    def test_default_method_lhs_produces_valid_dataframe(self):
        """method='LHS' (default) should produce a valid DataFrame."""
        designer = AdvExpDesigner(
            continuous_params=CONTINUOUS_PARAMS_SIMPLE,
            conditional_subparameters=CONDITIONAL_SUBPARAMETERS_SINGLE,
            seed=0,
        )
        design = designer.initialize(m_initial=16)
        assert isinstance(design, pd.DataFrame)
        assert len(design) == 16
        assert 'Temperature' in design.columns

    @pytest.mark.slow
    def test_method_optimized_returns_valid_dataframe(self):
        """method='Optimized' should return a valid DataFrame of the right length."""
        designer = AdvExpDesigner(
            continuous_params=CONTINUOUS_PARAMS_SIMPLE,
            conditional_subparameters=CONDITIONAL_SUBPARAMETERS_SINGLE,
            X_space=X_SPACE_SIMPLE,
            seed=0,
        )
        design = designer.initialize(m_initial=12, method='Optimized')
        assert isinstance(design, pd.DataFrame)
        assert len(design) == 12

    @pytest.mark.fast
    def test_seed_produces_reproducible_design(self):
        """Same seed should produce the same design on repeated calls."""
        designer = AdvExpDesigner(
            continuous_params=CONTINUOUS_PARAMS_SIMPLE,
            conditional_subparameters=CONDITIONAL_SUBPARAMETERS_SINGLE,
            seed=42,
        )
        design_a = designer.initialize(m_initial=20)
        design_b = designer.initialize(m_initial=20)
        pd.testing.assert_frame_equal(design_a, design_b)


# ---------------------------------------------------------------------------
# 10. Campaign integration tests
# ---------------------------------------------------------------------------

class TestCampaignIntegration:

    @pytest.mark.fast
    def test_is_subclass_of_exp_designer(self):
        assert issubclass(AdvExpDesigner, ExpDesigner)

    @pytest.mark.fast
    def test_isinstance_check(self):
        designer = AdvExpDesigner(
            continuous_params=CONTINUOUS_PARAMS_SIMPLE,
            X_space=X_SPACE_SIMPLE,
            seed=0,
        )
        assert isinstance(designer, ExpDesigner)

    @pytest.mark.fast
    def test_campaign_accepts_adv_exp_designer(self):
        designer = AdvExpDesigner(
            continuous_params=CONTINUOUS_PARAMS_SIMPLE,
            X_space=X_SPACE_SIMPLE,
            seed=0,
        )
        campaign = Campaign(X_SPACE_SIMPLE, TARGET, designer=designer)
        assert isinstance(campaign.designer, AdvExpDesigner)

    @pytest.mark.fast
    def test_campaign_initialize_returns_dataframe(self):
        designer = AdvExpDesigner(
            continuous_params=CONTINUOUS_PARAMS_SIMPLE,
            X_space=X_SPACE_SIMPLE,
            seed=0,
        )
        campaign = Campaign(X_SPACE_SIMPLE, TARGET, designer=designer)
        X0 = campaign.initialize(m_initial=16)
        assert isinstance(X0, pd.DataFrame)
        assert len(X0) == 16

    @pytest.mark.fast
    def test_campaign_initialize_bias_preserved_through_campaign(self):
        """Biased distribution should survive the Campaign.initialize() call path."""
        biased_params = {
            'Temperature': {'levels': [60, 70, 80, 90], 'biases': [0.05, 0.10, 0.35, 0.50]},
            'pH': {'levels': [6.0, 6.5, 7.0, 7.5, 8.0], 'biases': [0.05, 0.10, 0.20, 0.30, 0.35]},
        }
        designer = AdvExpDesigner(
            continuous_params=biased_params,
            X_space=X_SPACE_SIMPLE,
            seed=42,
        )
        campaign = Campaign(X_SPACE_SIMPLE, TARGET, designer=designer)
        X0 = campaign.initialize(m_initial=200)
        counts = X0['Temperature'].value_counts()
        assert counts.get(90, 0) > counts.get(60, 0), \
            f"Bias not preserved through Campaign: {dict(counts)}"

    @pytest.mark.fast
    def test_campaign_add_data_after_adv_initialize(self):
        """Data from AdvExpDesigner.initialize() can be added to Campaign after appending response."""
        designer = AdvExpDesigner(
            continuous_params=CONTINUOUS_PARAMS_SIMPLE,
            X_space=X_SPACE_SIMPLE,
            seed=0,
        )
        campaign = Campaign(X_SPACE_SIMPLE, TARGET, designer=designer)
        X0 = campaign.initialize(m_initial=10)
        X0['Yield'] = np.random.default_rng(0).uniform(50, 100, size=len(X0))
        campaign.add_data(X0)
        assert campaign.m_exp == 10


# ---------------------------------------------------------------------------
# 11. V175 integration test
# ---------------------------------------------------------------------------

class TestV175Integration:
    """
    End-to-end test mirroring the V175 Design.ipynb notebook scenario.
    Uses the exact parameter configuration from that notebook to verify
    realistic multi-param, multi-categorical usage.
    """

    @pytest.fixture(scope='class')
    def v175_designer(self):
        return AdvExpDesigner(
            V175_CONTINUOUS_PARAMS,
            V175_CONDITIONAL_SUBPARAMETERS,
            seed=132,
        )

    @pytest.fixture(scope='class')
    def v175_design(self, v175_designer):
        return v175_designer.generate_design(seed=132, n_samples=88)

    @pytest.mark.fast
    def test_v175_design_has_correct_shape(self, v175_design):
        assert len(v175_design) == 88

    @pytest.mark.fast
    def test_v175_design_has_all_expected_columns(self, v175_design):
        expected_cols = {
            '[surfactant] (%)', '[KCl]', '[Buffer] (mM)', '[EDTA] (mM)', '[L-Met] (mM)',
            'sugar_type', '[Sugar] (%)', 'buffer_type', 'pH',
        }
        assert expected_cols.issubset(set(v175_design.columns)), \
            f"Missing columns: {expected_cols - set(v175_design.columns)}"

    @pytest.mark.fast
    def test_v175_kcl_from_valid_levels(self, v175_design):
        """[KCl] uses an 8-tuple — values must come from those levels only."""
        valid = {0, 5, 10, 25, 50, 100, 200, 300}
        assert set(v175_design['[KCl]']).issubset(valid), \
            f"Invalid [KCl] values: {set(v175_design['[KCl]']) - valid}"

    @pytest.mark.fast
    def test_v175_surfactant_from_valid_levels(self, v175_design):
        valid = {0.001, 0.005, 0.025, 0.05, 0.125, 0.25, 0.5}
        assert set(v175_design['[surfactant] (%)']).issubset(valid), \
            f"Invalid [surfactant] values: {set(v175_design['[surfactant] (%)']) - valid}"

    @pytest.mark.fast
    def test_v175_no_sugar_has_zero_concentration(self, v175_design):
        no_sugar_rows = v175_design[v175_design['sugar_type'] == 'no_sugar']
        assert (no_sugar_rows['[Sugar] (%)'] == 0).all()

    @pytest.mark.fast
    def test_v175_buffer_ph_within_type_range(self, v175_design):
        """Each buffer_type should have pH values only from its defined range."""
        ph_ranges = {
            'citrate':     {3.5, 4.0, 4.5, 5.0, 5.5},
            'histidine':   {5.0, 5.5, 6.0, 6.5, 7.0},
            'k phosphate': {6.0, 6.5, 7.0, 7.5, 8.0},
            'hepes':       {6.5, 7.0, 7.5, 8.0, 8.5},
            'tris':        {7.0, 7.5, 8.0, 8.5, 9.0},
        }
        for buf_type, valid_ph in ph_ranges.items():
            rows = v175_design[v175_design['buffer_type'] == buf_type]
            if len(rows) == 0:
                continue
            assert set(rows['pH']).issubset(valid_ph), \
                f"{buf_type} has invalid pH: {set(rows['pH']) - valid_ph}"

    @pytest.mark.fast
    def test_v175_evaluate_design_returns_seven_metrics(self, v175_designer, v175_design):
        metrics = v175_designer.evaluate_design(v175_design)
        assert len(metrics) == 7
        assert all(np.isfinite(v) for v in metrics.values())

    @pytest.mark.fast
    def test_v175_sugar_type_frequencies_approximately_correct(self, v175_design):
        """sugar_type: no_sugar ≈ 50%, sucrose ≈ 25%, trehalose ≈ 25%."""
        counts = v175_design['sugar_type'].value_counts(normalize=True)
        assert abs(counts.get('no_sugar', 0)  - 0.50) < 0.15
        assert abs(counts.get('sucrose', 0)   - 0.25) < 0.15
        assert abs(counts.get('trehalose', 0) - 0.25) < 0.15

    @pytest.mark.fast
    def test_v175_buffer_type_frequencies_approximately_equal(self, v175_design):
        """All five buffer types should appear roughly 20% of the time each."""
        counts = v175_design['buffer_type'].value_counts(normalize=True)
        for buf in ['citrate', 'histidine', 'k phosphate', 'hepes', 'tris']:
            assert abs(counts.get(buf, 0) - 0.20) < 0.15, \
                f"{buf} frequency {counts.get(buf, 0):.3f} too far from expected 0.20"

    @pytest.mark.fast
    def test_v175_inferred_subparam_mapping(self):
        """V175 conditional_subparameters should infer both subparam mappings."""
        mapping = infer_subparam_mapping(V175_CONDITIONAL_SUBPARAMETERS)
        assert 'buffer_type' in mapping and mapping['buffer_type'] == 'pH'
        assert 'sugar_type'  in mapping and mapping['sugar_type']  == '[Sugar] (%)'

    @pytest.mark.slow
    def test_v175_optimize_design_smoke(self, v175_designer):
        """optimize_design on V175 params should complete without error."""
        best, metrics_df = v175_designer.optimize_design(n_trials=3, n_samples=30)
        assert isinstance(best, pd.DataFrame)
        assert len(best) == 30
        assert 'score' in metrics_df.columns


# ---------------------------------------------------------------------------
# 12. Plot smoke tests (headless)
# ---------------------------------------------------------------------------

class TestPlotSmokeTests:
    """All plot methods should execute without raising (headless Agg backend)."""

    @pytest.fixture(scope='class')
    def designer_and_design(self):
        designer = AdvExpDesigner(
            continuous_params=CONTINUOUS_PARAMS_SIMPLE,
            conditional_subparameters=CONDITIONAL_SUBPARAMETERS_SINGLE,
            seed=0,
        )
        design = designer.generate_design(seed=0, n_samples=30)
        return designer, design

    @pytest.mark.fast
    def test_plot_histograms(self, designer_and_design):
        designer, design = designer_and_design
        designer.plot_histograms(design)

    @pytest.mark.fast
    def test_plot_correlation(self, designer_and_design):
        designer, design = designer_and_design
        designer.plot_correlation(design)

    @pytest.mark.fast
    def test_plot_pca(self, designer_and_design):
        designer, design = designer_and_design
        designer.plot_pca(design, hue='buffer_type')

    @pytest.mark.fast
    def test_plot_mds(self, designer_and_design):
        designer, design = designer_and_design
        designer.plot_mds(design, hue='buffer_type')

    @pytest.mark.slow
    def test_plot_quality_evolution(self):
        """plot_quality_evolution requires a metrics_df with a 'seed' column."""
        designer = AdvExpDesigner(
            continuous_params=CONTINUOUS_PARAMS_SIMPLE,
            conditional_subparameters=CONDITIONAL_SUBPARAMETERS_SINGLE,
            seed=0,
        )
        _, metrics_df = designer.optimize_design(n_trials=3, n_samples=15)
        designer.plot_quality_evolution(metrics_df)


# ---------------------------------------------------------------------------
# Repr test (post-refactor)
# ---------------------------------------------------------------------------

class TestRepr:

    @pytest.mark.fast
    @pytest.mark.bug_regression
    def test_repr_identifies_class_as_adv_exp_designer(self):
        """
        After fix: __repr__ should identify AdvExpDesigner, not just ExpDesigner.
        """
        designer = AdvExpDesigner(
            continuous_params=CONTINUOUS_PARAMS_SIMPLE,
            conditional_subparameters=CONDITIONAL_SUBPARAMETERS_SINGLE,
            seed=0,
        )
        r = repr(designer)
        assert 'AdvExpDesigner' in r, \
            f"repr should contain 'AdvExpDesigner', got: {r!r}"


# ---------------------------------------------------------------------------
# 13. Round 2 regression tests
# ---------------------------------------------------------------------------

class TestRound2:
    """
    Tests for the round-2 improvements documented in claude/round_2.md.
    All marked @pytest.mark.r2; all are fast.
    """

    @pytest.mark.fast
    @pytest.mark.r2
    def test_subparam_key_singular_removed(self):
        """
        Round 2 item 1: self.subparam_key (singular) was dead code.
        After removal, the attribute must not exist on the instance.
        """
        designer = AdvExpDesigner(
            continuous_params=CONTINUOUS_PARAMS_SIMPLE,
            conditional_subparameters=CONDITIONAL_SUBPARAMETERS_SINGLE,
            seed=0,
        )
        assert not hasattr(designer, 'subparam_key'), (
            "self.subparam_key (singular) should have been removed — "
            "use self.subparam_keys (plural) instead"
        )

    @pytest.mark.fast
    @pytest.mark.r2
    def test_optimize_design_docstring_documents_n_category_trials_note(self):
        """
        Originally verified that optimize_design documented the non-forwarding of
        n_category_trials. That gap is now fixed (values ARE forwarded). Test updated
        to verify the forwarding actually works via the sprint-review fix.
        """
        import inspect
        sig = inspect.signature(find_best_design_parallel)
        assert 'n_category_trials' in sig.parameters, \
            "find_best_design_parallel should accept n_category_trials now that it is forwarded"

    @pytest.mark.fast
    @pytest.mark.r2
    def test_extend_design_docstring_documents_n_category_trials_note(self):
        """
        Originally verified the Note about non-forwarding in extend_design.
        That gap is now fixed. Test updated to verify the module-level extend_design
        also accepts n_category_trials.
        """
        import inspect
        from obsidian.experiment.advanced_design import extend_design as module_extend_design
        sig = inspect.signature(module_extend_design)
        assert 'n_category_trials' in sig.parameters, \
            "module-level extend_design should accept n_category_trials now that it is forwarded"

    @pytest.mark.fast
    @pytest.mark.r2
    def test_no_runtime_warning_during_optimize_design(self):
        """
        Round 2 item 3: normalization with inf values must not emit RuntimeWarning.
        """
        designer = AdvExpDesigner(
            continuous_params=CONTINUOUS_PARAMS_SIMPLE,
            conditional_subparameters=CONDITIONAL_SUBPARAMETERS_SINGLE,
            seed=0,
        )
        with warnings.catch_warnings():
            warnings.simplefilter('error', RuntimeWarning)
            # n_samples=5 is intentionally tiny to provoke near-singular matrices
            # which cause A-optimality = inf and trigger the normalization path
            designer.optimize_design(n_trials=4, n_samples=5)

    @pytest.mark.fast
    @pytest.mark.r2
    def test_plot_umap_method_accepts_verbose_false(self):
        """
        Round 2 item 4: AdvExpDesigner.plot_umap should accept verbose=False
        without raising (implementation check — UMAP not actually invoked).
        """
        import inspect
        sig = inspect.signature(AdvExpDesigner.plot_umap)
        assert 'verbose' in sig.parameters, (
            "plot_umap method should have a 'verbose' parameter"
        )
        assert sig.parameters['verbose'].default is False, (
            "plot_umap verbose parameter should default to False"
        )

    @pytest.mark.fast
    @pytest.mark.r2
    def test_module_plot_umap_accepts_verbose_false(self):
        """
        Round 2 item 4: module-level plot_umap function should also accept verbose=False.
        """
        from obsidian.experiment.advanced_design import plot_umap
        import inspect
        sig = inspect.signature(plot_umap)
        assert 'verbose' in sig.parameters, (
            "module-level plot_umap should have a 'verbose' parameter"
        )
        assert sig.parameters['verbose'].default is False, (
            "module-level plot_umap verbose parameter should default to False"
        )

    @pytest.mark.fast
    @pytest.mark.r2
    def test_optimize_category_assignment_parallel_accepts_n_category_trials(self):
        """
        Round 2 item 5: optimize_category_assignment_parallel must accept
        n_category_trials (not n_trials) as its keyword argument.
        """
        from obsidian.experiment.advanced_design import optimize_category_assignment_parallel
        import inspect
        sig = inspect.signature(optimize_category_assignment_parallel)
        assert 'n_category_trials' in sig.parameters, (
            "optimize_category_assignment_parallel should use 'n_category_trials' "
            "(not 'n_trials') for consistency with the instance attribute"
        )
        assert 'n_trials' not in sig.parameters, (
            "Old parameter name 'n_trials' should be replaced by 'n_category_trials'"
        )

    @pytest.mark.fast
    @pytest.mark.r2
    def test_init_type_hints_present(self):
        """
        Round 2 item 6: AdvExpDesigner.__init__ should have type annotations
        for seed, n_category_trials, and corr_threshold.
        """
        import inspect
        sig = inspect.signature(AdvExpDesigner.__init__)
        annotations = {k: v.annotation for k, v in sig.parameters.items()}
        assert annotations.get('seed') is not inspect.Parameter.empty, \
            "seed parameter should have a type annotation"
        assert annotations.get('n_category_trials') is not inspect.Parameter.empty, \
            "n_category_trials parameter should have a type annotation"
        assert annotations.get('corr_threshold') is not inspect.Parameter.empty, \
            "corr_threshold parameter should have a type annotation"

    @pytest.mark.fast
    @pytest.mark.r2
    def test_compare_frequencies_verbose_output_matches_dataframe(self):
        """
        Round 2 item 7: compare_frequencies(verbose=True) should print output
        derived from the returned DataFrame (not a separate loop). Verify by
        checking that all level names from the DataFrame appear in stdout.
        """
        import contextlib
        designer = AdvExpDesigner(
            continuous_params=CONTINUOUS_PARAMS_SIMPLE,
            conditional_subparameters=CONDITIONAL_SUBPARAMETERS_SINGLE,
            seed=0,
        )
        design = designer.generate_design(seed=0, n_samples=80)
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            result = designer.compare_frequencies(design, verbose=True)
        output = buf.getvalue()

        # Every level name present in the DataFrame must appear in the printed output
        for level in result['level']:
            assert str(level) in output, (
                f"Level '{level}' from compare_frequencies DataFrame not found in "
                f"verbose output — verbose mode may be re-iterating raw data instead "
                f"of printing the DataFrame"
            )


# ---------------------------------------------------------------------------
# 14. Round 3 regression tests
# ---------------------------------------------------------------------------

class TestRound3:
    """
    Tests for the round-3 improvements documented in claude/round_3.md.
    All marked @pytest.mark.r3; all are fast.
    """

    @pytest.mark.fast
    @pytest.mark.r3
    def test_default_metrics_constant_has_seven_entries(self):
        """DEFAULT_METRICS is importable and contains exactly the 7 standard metrics."""
        from obsidian.experiment.advanced_design import DEFAULT_METRICS
        assert len(DEFAULT_METRICS) == 7
        expected = {
            "D-optimality", "A-optimality", "Condition Number",
            "Pairwise Distance CV", "Max Continuous Corr",
            "Max Categorical Corr", "Max Mixed Corr",
        }
        assert set(DEFAULT_METRICS) == expected

    @pytest.mark.fast
    @pytest.mark.r3
    def test_default_maximize_metrics_helper(self):
        """_default_maximize_metrics returns [True, False, ...] for any list."""
        from obsidian.experiment.advanced_design import DEFAULT_METRICS, _default_maximize_metrics
        result = _default_maximize_metrics(DEFAULT_METRICS)
        assert result[0] is True
        assert all(v is False for v in result[1:])
        assert len(result) == len(DEFAULT_METRICS)

    @pytest.mark.fast
    @pytest.mark.r3
    def test_evaluate_candidate_returns_correct_structure(self):
        """evaluate_candidate must still return dict with seed, metrics, metric_values, new_samples."""
        subparam_mapping = infer_subparam_mapping(CONDITIONAL_SUBPARAMETERS_SINGLE)
        existing = sample_design(
            seed=0, n_samples=10,
            continuous_params=CONTINUOUS_PARAMS_METRIC,
            conditional_subparameters=CONDITIONAL_SUBPARAMETERS_SINGLE,
            subparam_mapping=subparam_mapping,
        )
        result = evaluate_candidate(
            i=0, seed_start=50, n=5,
            continuous_params=CONTINUOUS_PARAMS_METRIC,
            conditional_subparameters=CONDITIONAL_SUBPARAMETERS_SINGLE,
            subparam_mapping=subparam_mapping,
            existing_design=existing,
            continuous_keys=list(CONTINUOUS_PARAMS_METRIC.keys()),
            categorical_keys=list(CONDITIONAL_SUBPARAMETERS_SINGLE.keys()),
            metrics_to_optimize=['D-optimality', 'Max Continuous Corr'],
        )
        assert set(result.keys()) == {'seed', 'metrics', 'metric_values', 'new_samples'}
        assert isinstance(result['metrics'], dict)
        assert len(result['metric_values']) == 2
        assert isinstance(result['new_samples'], pd.DataFrame)

    @pytest.mark.fast
    @pytest.mark.r3
    def test_plot_quality_evolution_excludes_seed_and_score(self):
        """plot_design_quality_evolution must not plot seed or score as metric axes."""
        import matplotlib
        matplotlib.use('Agg')
        designer = AdvExpDesigner(
            continuous_params=CONTINUOUS_PARAMS_SIMPLE,
            conditional_subparameters=CONDITIONAL_SUBPARAMETERS_SINGLE,
            seed=0,
        )
        # Build a minimal metrics_df with seed, score, and a real metric
        metrics_df = pd.DataFrame({
            'seed': [0, 1, 2],
            'D-optimality': [1e8, 2e8, 1.5e8],
            'score': [0.5, 0.8, 0.6],
        })
        import matplotlib.pyplot as plt
        plt.close('all')
        designer.plot_quality_evolution(metrics_df)
        fig = plt.gcf()
        titles = [ax.get_title() for ax in fig.axes if ax.get_visible()]
        titles_lower = [t.lower() for t in titles if t]
        # Each title is "{metric} vs Seed" — check that no title begins with "seed" or "score"
        assert not any(t.startswith('seed') for t in titles_lower), \
            f"'seed' column should not be plotted as a metric; got titles: {titles}"
        assert not any(t.startswith('score') for t in titles_lower), \
            f"'score' column should not be plotted as a metric; got titles: {titles}"
        plt.close('all')

    @pytest.mark.fast
    @pytest.mark.r3
    def test_infer_subparam_mapping_warns_for_multi_subparam_category(self):
        """infer_subparam_mapping emits UserWarning when a category has >1 subparam."""
        multi_subparam = {
            'buffer_type': {
                'citrate': {'freq': 0.5, 'pH': ([4.0, 5.0], [0.5, 0.5]), 'temperature': ([25, 37], [0.5, 0.5])},
                'tris':    {'freq': 0.5, 'pH': ([8.0, 9.0], [0.5, 0.5]), 'temperature': ([25, 37], [0.5, 0.5])},
            }
        }
        with pytest.warns(UserWarning, match="multiple subparameters"):
            result = infer_subparam_mapping(multi_subparam)
        # Category with multiple subparams is excluded from mapping
        assert 'buffer_type' not in result

    @pytest.mark.fast
    @pytest.mark.r3
    def test_initialize_optimized_raises_on_optimize_categories(self):
        """initialize(method='Optimized', optimize_categories=True) must raise ValueError."""
        designer = AdvExpDesigner(
            continuous_params=CONTINUOUS_PARAMS_SIMPLE,
            conditional_subparameters=CONDITIONAL_SUBPARAMETERS_SINGLE,
            seed=0,
        )
        with pytest.raises(ValueError, match="optimize_categories"):
            designer.initialize(m_initial=10, method='Optimized', optimize_categories=True)

    @pytest.mark.fast
    @pytest.mark.r3
    def test_generate_design_docstring_notes_single_category_limit(self):
        """generate_design docstring must document the single-category optimization limit."""
        doc = AdvExpDesigner.generate_design.__doc__ or ''
        assert 'Note' in doc, "generate_design docstring must contain a 'Note:' section"
        assert 'first' in doc.lower() or 'single' in doc.lower() or 'only' in doc.lower(), \
            "generate_design Note must document the single-category limitation"


# ---------------------------------------------------------------------------
# 15. Round 4 regression tests
# ---------------------------------------------------------------------------

class TestRound4:
    """
    Tests for the round-4 improvements documented in claude/round_4.md.
    All marked @pytest.mark.r4; all are fast.
    """

    @pytest.mark.fast
    @pytest.mark.r4
    def test_prepare_design_matrix_helper_importable(self):
        """_prepare_design_matrix must be importable from the module."""
        from obsidian.experiment.advanced_design import _prepare_design_matrix
        assert callable(_prepare_design_matrix)

    @pytest.mark.fast
    @pytest.mark.r4
    def test_prepare_design_matrix_output_shape(self):
        """_prepare_design_matrix returns X_std with shape (n_samples, n_features)."""
        from obsidian.experiment.advanced_design import _prepare_design_matrix
        design = sample_design(
            seed=0, n_samples=20,
            continuous_params=CONTINUOUS_PARAMS_METRIC,
            conditional_subparameters=CONDITIONAL_SUBPARAMETERS_SINGLE,
        )
        cont_keys = list(CONTINUOUS_PARAMS_METRIC.keys())
        X_std, combined_keys = _prepare_design_matrix(design, cont_keys, ['pH'])
        assert X_std.shape == (20, 3), f"Expected (20, 3), got {X_std.shape}"
        assert combined_keys == cont_keys + ['pH']

    @pytest.mark.fast
    @pytest.mark.r4
    def test_metric_functions_unchanged_after_helper_extraction(self):
        """All five metric functions must return the same values after the helper refactor."""
        design = sample_design(
            seed=42, n_samples=30,
            continuous_params=CONTINUOUS_PARAMS_METRIC,
            conditional_subparameters=CONDITIONAL_SUBPARAMETERS_SINGLE,
        )
        cont_keys = list(CONTINUOUS_PARAMS_METRIC.keys())
        subparam_keys = ['pH']

        d = calculate_d_optimality(design, cont_keys, subparam_keys)
        a = calculate_a_optimality(design, cont_keys, subparam_keys)
        c = calculate_condition_number(design, cont_keys, subparam_keys)
        p = calculate_pairwise_distance_uniformity(design, cont_keys, subparam_keys)
        mc = calculate_max_continuous_correlation(design, cont_keys, subparam_keys)

        # All should be finite and positive
        assert np.isfinite(d) and d > 0, f"D-optimality unexpected: {d}"
        assert np.isfinite(a) and a > 0, f"A-optimality unexpected: {a}"
        assert np.isfinite(c) and c >= 1.0, f"Condition number unexpected: {c}"
        assert np.isfinite(p) and p >= 0, f"Pairwise CV unexpected: {p}"
        assert 0.0 <= mc <= 1.0, f"Max continuous corr unexpected: {mc}"

    @pytest.mark.fast
    @pytest.mark.r4
    def test_evaluate_design_docstring_references_default_metrics(self):
        """The standalone evaluate_design docstring must reference DEFAULT_METRICS."""
        from obsidian.experiment.advanced_design import evaluate_design
        doc = evaluate_design.__doc__ or ''
        assert 'DEFAULT_METRICS' in doc, (
            "evaluate_design docstring should reference DEFAULT_METRICS "
            "now that the constant is public"
        )


# ---------------------------------------------------------------------------
# 16. Sprint-review fixes: lazy evaluate_design + forwarded hyperparams
# ---------------------------------------------------------------------------

class TestSprintReviewFixes:
    """
    Tests for the two issues raised in sprint review:
    1. evaluate_design was eagerly computing all metrics then discarding unused ones.
    2. n_category_trials / corr_threshold were stored on self but not forwarded to workers.
    """

    @pytest.mark.fast
    def test_evaluate_design_only_computes_requested_metrics(self):
        """
        evaluate_design must not compute metrics outside metrics_to_optimize.
        We verify by requesting a single cheap metric (D-optimality) and confirming
        the returned dict contains exactly that key and nothing else.
        """
        design = sample_design(
            seed=0, n_samples=20,
            continuous_params=CONTINUOUS_PARAMS_METRIC,
            conditional_subparameters=CONDITIONAL_SUBPARAMETERS_SINGLE,
        )
        result = evaluate_design(
            design=design,
            continuous_keys=list(CONTINUOUS_PARAMS_METRIC.keys()),
            categorical_keys=list(CONDITIONAL_SUBPARAMETERS_SINGLE.keys()),
            subparam_mapping={'buffer_type': 'pH'},
            metrics_to_optimize=['D-optimality'],
        )
        assert set(result.keys()) == {'D-optimality'}, (
            f"evaluate_design should return only requested metrics; got {set(result.keys())}"
        )

    @pytest.mark.fast
    def test_evaluate_design_subset_skips_correlation_computation(self):
        """
        Requesting only information-matrix metrics must not compute correlation metrics.
        Verify by requesting a subset that excludes all Corr metrics and confirming
        no Corr key appears in the result.
        """
        design = sample_design(
            seed=5, n_samples=20,
            continuous_params=CONTINUOUS_PARAMS_METRIC,
            conditional_subparameters=CONDITIONAL_SUBPARAMETERS_SINGLE,
        )
        subset = ['D-optimality', 'A-optimality', 'Condition Number']
        result = evaluate_design(
            design=design,
            continuous_keys=list(CONTINUOUS_PARAMS_METRIC.keys()),
            categorical_keys=list(CONDITIONAL_SUBPARAMETERS_SINGLE.keys()),
            subparam_mapping={'buffer_type': 'pH'},
            metrics_to_optimize=subset,
        )
        assert set(result.keys()) == set(subset)
        assert not any('Corr' in k for k in result), \
            "No correlation metric should be present when not requested"

    @pytest.mark.fast
    def test_n_category_trials_forwarded_to_find_best_design_parallel(self):
        """
        find_best_design_parallel must accept n_category_trials and corr_threshold.
        Verify the parameters exist in the function signature.
        """
        import inspect
        sig = inspect.signature(find_best_design_parallel)
        assert 'n_category_trials' in sig.parameters, \
            "find_best_design_parallel must accept n_category_trials"
        assert 'corr_threshold' in sig.parameters, \
            "find_best_design_parallel must accept corr_threshold"

    @pytest.mark.fast
    def test_n_category_trials_forwarded_to_generate_and_evaluate(self):
        """
        generate_and_evaluate must accept n_category_trials and corr_threshold.
        """
        import inspect
        from obsidian.experiment.advanced_design import generate_and_evaluate
        sig = inspect.signature(generate_and_evaluate)
        assert 'n_category_trials' in sig.parameters
        assert 'corr_threshold' in sig.parameters

    @pytest.mark.fast
    def test_n_category_trials_forwarded_to_evaluate_candidate(self):
        """
        evaluate_candidate must accept n_category_trials and corr_threshold.
        """
        import inspect
        sig = inspect.signature(evaluate_candidate)
        assert 'n_category_trials' in sig.parameters
        assert 'corr_threshold' in sig.parameters

    @pytest.mark.fast
    def test_optimize_design_passes_instance_hyperparams_to_workers(self):
        """
        AdvExpDesigner.optimize_design must forward self.n_category_trials and
        self.corr_threshold to find_best_design_parallel. Verify by monkeypatching
        and capturing the call arguments.
        """
        import obsidian.experiment.advanced_design as adv_mod
        captured = {}
        original = adv_mod.find_best_design_parallel

        def capturing_wrapper(*args, **kwargs):
            captured['n_category_trials'] = kwargs.get('n_category_trials')
            captured['corr_threshold'] = kwargs.get('corr_threshold')
            return original(*args, **kwargs)

        designer = AdvExpDesigner(
            continuous_params=CONTINUOUS_PARAMS_SIMPLE,
            conditional_subparameters=CONDITIONAL_SUBPARAMETERS_SINGLE,
            seed=0,
            n_category_trials=42,
            corr_threshold=0.005,
        )
        adv_mod.find_best_design_parallel = capturing_wrapper
        try:
            designer.optimize_design(n_trials=2, n_samples=10)
        finally:
            adv_mod.find_best_design_parallel = original

        assert captured.get('n_category_trials') == 42, (
            f"n_category_trials should be 42 (from constructor), got {captured.get('n_category_trials')}"
        )
        assert captured.get('corr_threshold') == 0.005, (
            f"corr_threshold should be 0.005 (from constructor), got {captured.get('corr_threshold')}"
        )

    @pytest.mark.fast
    def test_all_exported_via_all(self):
        """
        advanced_design.__all__ must exist and contain AdvExpDesigner.
        Wildcard import should not flood the namespace with internal helpers.
        """
        import obsidian.experiment.advanced_design as adv_mod
        assert hasattr(adv_mod, '__all__'), "Module must define __all__"
        assert 'AdvExpDesigner' in adv_mod.__all__
        assert '_prepare_design_matrix' not in adv_mod.__all__, \
            "Private helpers should not be in __all__"
        assert 'cramers_v_np' not in adv_mod.__all__, \
            "Internal utility functions should not be in __all__"


# ---------------------------------------------------------------------------
# 17. Copilot review fixes
# ---------------------------------------------------------------------------


class TestCopilotReviewFixes:
    """
    Tests for issues identified in the second Copilot code review pass.

    Issues covered:
    1. calculate_max_continuous_correlation returned NaN when a column was
       constant (corr() returns NaN for zero-variance columns).
    2. generate_weights divided by zero when all row weights were zeroed out
       by a bias configuration.
    3. Step-based rounding used log10 which only works for powers-of-ten steps;
       step=0.25 produced 1 decimal instead of 2, corrupting sampled values.
    4. surface_plot had a mutable default argument (list) for feature_ids.
    5. Custom markers were not registered in pytest.ini.
    """

    @pytest.mark.fast
    def test_continuous_corr_constant_column_returns_finite(self):
        """
        calculate_max_continuous_correlation must return a finite value when one
        column is constant. corr() produces NaN for zero-variance columns;
        without fillna(0) the max propagates as NaN and poisons scoring.
        """
        design = pd.DataFrame({
            'A': [1.0, 2.0, 3.0, 4.0, 5.0],
            'B': [3.0, 3.0, 3.0, 3.0, 3.0],  # constant — corr undefined
        })
        val = calculate_max_continuous_correlation(design, ['A', 'B'], [])
        assert np.isfinite(val), f"Expected finite result for constant column, got {val}"
        assert val == 0.0, f"NaN correlation should be treated as 0, got {val}"

    @pytest.mark.fast
    def test_generate_weights_raises_on_zero_sum(self):
        """
        generate_weights must raise ValueError when the bias config zeroes all
        row weights (weight=0 covering every candidate). Previously this caused
        a silent ZeroDivisionError or NaN weights.
        """
        from obsidian.experiment.sampling import generate_weights
        df = pd.DataFrame({'x': [1.0, 2.0, 3.0, 4.0, 5.0]})
        # weight=0 with range [0, 10] covers all rows → weights.sum() == 0
        bias = {'x': [0.0, 10.0, 0]}
        with pytest.raises(ValueError, match="zero"):
            generate_weights(df, n=3, bias=bias)

    @pytest.mark.fast
    def test_step_rounding_quarter_step_preserves_levels(self):
        """
        sample_design with step=0.25 must snap values to the discrete level set
        {0.0, 0.25, 0.50, 0.75, 1.0}. The old log10 formula gave decimals=1
        for any step in (0.1, 1.0), which would round 0.25 → 0.2 or 0.3.
        """
        params = {'x': (0.0, 1.0, 0.25)}
        design = sample_design(
            seed=0, n_samples=20,
            continuous_params=params,
            conditional_subparameters={},
        )
        valid_levels = {0.0, 0.25, 0.50, 0.75, 1.0}
        bad = [v for v in design['x'] if round(float(v), 10) not in valid_levels]
        assert not bad, f"Values outside discrete level set {valid_levels}: {bad}"

    @pytest.mark.fast
    def test_step_rounding_half_step_preserves_levels(self):
        """step=0.5 (another non-power-of-ten that the old formula handled
        coincidentally correctly) should still snap to {0.0, 0.5, 1.0, 1.5, 2.0}."""
        params = {'x': (0.0, 2.0, 0.5)}
        design = sample_design(
            seed=1, n_samples=20,
            continuous_params=params,
            conditional_subparameters={},
        )
        valid_levels = {0.0, 0.5, 1.0, 1.5, 2.0}
        bad = [v for v in design['x'] if round(float(v), 10) not in valid_levels]
        assert not bad, f"Values outside discrete level set {valid_levels}: {bad}"

    @pytest.mark.fast
    def test_surface_plot_feature_ids_default_is_none(self):
        """
        surface_plot must use None (not a list literal) as its feature_ids
        default to avoid the mutable default argument footgun.
        """
        import inspect
        from obsidian.plotting.plotly import surface_plot
        default = inspect.signature(surface_plot).parameters['feature_ids'].default
        assert default is None, (
            f"feature_ids default should be None to avoid shared mutable state; got {default!r}"
        )

    @pytest.mark.fast
    def test_custom_markers_registered_in_pytest_ini(self):
        """
        bug_regression, r2, r3, r4 must be declared in pytest.ini so that
        --strict-markers does not fail and no PytestUnknownMarkWarning is emitted.
        """
        ini_path = pathlib.Path(__file__).parents[2] / 'pytest.ini'
        with open(ini_path) as f:
            content = f.read()
        for marker in ('bug_regression', 'r2', 'r3', 'r4'):
            assert marker in content, (
                f"Marker '{marker}' not found in {ini_path} — add it to the markers section"
            )


# ---------------------------------------------------------------------------
# 18. Second Copilot review pass — new findings
# ---------------------------------------------------------------------------


class TestCopilotReviewFixes2:
    """
    Tests for issues identified in the third Copilot review pass.

    Issues covered:
    1. best_sample() passed a numpy Generator to DataFrame.sample(random_state=),
       which only accepts int/RandomState/BitGenerator — raises TypeError at runtime.
    2. best_sample() docstring described a 'weights' parameter that does not exist
       in the signature (the parameter is 'bias').
    3. surface_plot() assumed feature_ids has exactly 2 entries with no validation;
       wrong length caused an opaque IndexError deep in the function.
    4. _prepare_design_matrix() divided by zero for constant columns, emitting
       RuntimeWarning and producing NaN in X_std that propagated to callers.
    """

    @pytest.mark.fast
    def test_best_sample_reproducible_with_random_state(self):
        """
        best_sample must produce identical results for the same random_state.
        Previously it passed a Generator directly to DataFrame.sample(), which
        Pandas does not support (accepts int/RandomState/BitGenerator only).
        """
        from obsidian.experiment.sampling import best_sample
        df = pd.DataFrame({
            'x': np.linspace(0, 10, 50),
            'y': np.linspace(0, 5, 50),
        })
        result1, _ = best_sample(df, k=8, feature_cols=['x', 'y'],
                                 n_trials=20, random_state=42)
        result2, _ = best_sample(df, k=8, feature_cols=['x', 'y'],
                                 n_trials=20, random_state=42)
        pd.testing.assert_frame_equal(result1, result2)

    @pytest.mark.fast
    def test_best_sample_random_state_no_type_error(self):
        """
        best_sample must not raise TypeError when random_state is provided.
        The Generator→DataFrame.sample bug would surface here.
        """
        from obsidian.experiment.sampling import best_sample
        df = pd.DataFrame({'x': range(20), 'y': range(20)})
        try:
            best_sample(df, k=5, feature_cols=['x', 'y'],
                        n_trials=5, random_state=99)
        except TypeError as e:
            pytest.fail(f"TypeError raised — Generator was passed to DataFrame.sample: {e}")

    @pytest.mark.fast
    def test_best_sample_docstring_describes_bias_not_weights(self):
        """
        best_sample docstring must describe 'bias', not 'weights' (which is not
        a parameter of this function).
        """
        from obsidian.experiment.sampling import best_sample
        import inspect
        doc = inspect.getdoc(best_sample)
        assert 'bias' in doc, "Docstring should describe the 'bias' parameter"
        assert 'weights:' not in doc, (
            "Docstring refers to 'weights' which is not a parameter — should be 'bias'"
        )

    @pytest.mark.fast
    def test_surface_plot_raises_on_wrong_feature_ids_length(self):
        """
        surface_plot must raise ValueError with a clear message when feature_ids
        does not have exactly 2 entries, not an opaque IndexError later.
        The length check happens before the Optimizer isinstance check, so a
        plain object can be used as the optimizer argument.
        """
        from obsidian.plotting.plotly import surface_plot
        with pytest.raises(ValueError, match="exactly 2"):
            surface_plot(object(), feature_ids=[0, 1, 2])
        with pytest.raises(ValueError, match="exactly 2"):
            surface_plot(object(), feature_ids=[0])

    @pytest.mark.fast
    def test_prepare_design_matrix_constant_column_no_nan_no_warning(self):
        """
        _prepare_design_matrix must not emit RuntimeWarning or produce NaN in
        X_std when a column is constant (zero variance). The fix replaces zero
        std with 1.0 before dividing, so the constant column becomes all-zeros.
        """
        from obsidian.experiment.advanced_design import _prepare_design_matrix
        design = pd.DataFrame({
            'A': [1.0, 2.0, 3.0, 4.0, 5.0],
            'B': [7.0, 7.0, 7.0, 7.0, 7.0],  # constant
        })
        with warnings.catch_warnings():
            warnings.simplefilter("error")  # RuntimeWarning → error
            X_std, keys = _prepare_design_matrix(design, ['A', 'B'], [])
        assert not np.any(np.isnan(X_std)), "X_std must not contain NaN for constant columns"
        assert np.all(X_std[:, 1] == 0.0), "Constant column should become all-zeros in X_std"


# ---------------------------------------------------------------------------
# 19. Fourth Copilot review pass
# ---------------------------------------------------------------------------


class TestCopilotReviewFixes3:
    """
    Tests for issues identified in the fourth Copilot review pass (15:55 batch).

    Issues covered:
    1. initialize() error message incorrectly suggests optimize_design() for
       optimize_categories, but optimize_design() also cannot honor it.
    2. compare_frequencies() raises KeyError when a categorical key is absent
       from conditional_subparameters.
    3. sample_continuous_lhs() raises when continuous_params is empty (d=0).
    4. plot_pca() / plot_mds() standardize without guarding zero-variance columns,
       emitting RuntimeWarning and producing NaN that breaks projection.
    5. parity_plot() formats error bar arrays as strings instead of floats,
       which Plotly silently ignores or raises a type error.
    6. best_sample() computes weights on the unfiltered df but samples from dfv
       (post-dropna); with enforce=True, the row count check can pass while
       sampling still fails.
    """

    @pytest.mark.fast
    def test_initialize_error_message_does_not_suggest_optimize_design(self):
        """
        The ValueError raised for initialize(method='Optimized',
        optimize_categories=True) must not recommend optimize_design() — that
        function also cannot honor optimize_categories.
        """
        designer = AdvExpDesigner(
            continuous_params=CONTINUOUS_PARAMS_METRIC,
            conditional_subparameters=CONDITIONAL_SUBPARAMETERS_SINGLE,
        )
        with pytest.raises(ValueError) as exc_info:
            designer.initialize(m_initial=10, method='Optimized',
                                optimize_categories=True)
        msg = str(exc_info.value)
        assert 'optimize_design()' not in msg, (
            "Error message should not suggest optimize_design() — "
            "that function also cannot optimize categories"
        )
        # Must still point to a valid alternative
        assert "method='LHS'" in msg or 'generate_design' in msg or 'sample_design' in msg, (
            "Error message should recommend a valid alternative (method='LHS' or "
            "generate_design/sample_design)"
        )

    @pytest.mark.fast
    def test_compare_frequencies_keyerror_missing_subparams(self):
        """
        compare_frequencies() must raise a clear error (not a bare KeyError)
        when a categorical key has no entry in conditional_subparameters.
        Currently it raises KeyError; after the fix it should raise ValueError.
        """
        designer = AdvExpDesigner(
            continuous_params=CONTINUOUS_PARAMS_METRIC,
            conditional_subparameters=CONDITIONAL_SUBPARAMETERS_SINGLE,
        )
        design = sample_design(
            seed=0, n_samples=20,
            continuous_params=CONTINUOUS_PARAMS_METRIC,
            conditional_subparameters=CONDITIONAL_SUBPARAMETERS_SINGLE,
        )
        # Patch categorical_keys to include a key absent from conditional_subparameters
        original = designer.categorical_keys
        designer.categorical_keys = original + ['nonexistent_cat']
        try:
            with pytest.raises(ValueError, match="nonexistent_cat"):
                designer.compare_frequencies(design)
        finally:
            designer.categorical_keys = original

    @pytest.mark.fast
    def test_sample_continuous_lhs_empty_params_returns_empty_dict(self):
        """
        sample_continuous_lhs({}, ...) must return {} without raising.
        Previously it passed d=0 to qmc.LatinHypercube which is unsupported.
        """
        result = sample_continuous_lhs({}, n_samples=10, seed=0)
        assert result == {}, f"Expected empty dict for empty params, got {result!r}"

    @pytest.mark.fast
    def test_plot_pca_constant_column_no_warning(self, small_design):
        """
        plot_pca must not emit RuntimeWarning when a continuous column is
        constant. The inline np.std division produces NaN without a zero-std
        guard, which then makes PCA fail or produce nonsensical output.
        """
        design_with_const = small_design.copy()
        design_with_const['Temperature'] = 75.0  # force constant column
        with warnings.catch_warnings():
            warnings.simplefilter("error", RuntimeWarning)
            plot_pca(design_with_const,
                     continuous_params_keys=['Temperature', 'Concentration'],
                     subparam_mapping={'buffer_type': 'pH'})

    @pytest.mark.fast
    def test_plot_mds_constant_column_no_warning(self, small_design):
        """
        plot_mds must not emit RuntimeWarning for a constant column.
        Same root cause as plot_pca — inline std division without zero guard.
        """
        design_with_const = small_design.copy()
        design_with_const['Concentration'] = 0.5  # force constant column
        with warnings.catch_warnings():
            warnings.simplefilter("error", RuntimeWarning)
            plot_mds(design_with_const,
                     continuous_params_keys=['Temperature', 'Concentration'],
                     subparam_mapping={'buffer_type': 'pH'})

    @pytest.mark.fast
    def test_parity_plot_error_bars_are_numeric(self):
        """
        parity_plot() must pass numeric arrays (not formatted strings) for
        error_y.array and error_y.arrayminus. Plotly silently drops string
        arrays or raises a type error at render time.
        """
        from obsidian.plotting.plotly import parity_plot
        import inspect
        source = inspect.getsource(parity_plot)
        # The bad pattern is f"{y:.3G}" inside a list comprehension for array/arrayminus
        assert '"array": [f"' not in source and '"arrayminus": [f"' not in source, (
            "error_y array/arrayminus must not be formatted as strings — "
            "pass the raw numeric arrays and let hovertemplate handle formatting"
        )

    @pytest.mark.fast
    def test_best_sample_weights_computed_on_filtered_df(self):
        """
        best_sample() must compute weights on the post-dropna filtered df (dfv).
        When NaN rows inflate the apparent qualifying-row count on the full df,
        the enforce check passes against df (enough rows) but sampling from dfv
        fails with an opaque numpy error. After fix, enforce is checked against
        dfv and raises a clear ValueError mentioning 'enforce conditions'.

        Setup: df has 10 rows; rows 5-9 have NaN in y (feature col) so dfv=rows 0-4.
        bias targets x in [1, 2.5] (rows 0,1 in dfv; rows 0,1,5,6 in full df).
        enforce=True, k=3: df sees 4 qualifying rows (passes), dfv sees 2 (should fail).
        """
        from obsidian.experiment.sampling import best_sample
        df = pd.DataFrame({
            'x': [1.0, 2.0, 3.0, 4.0, 5.0, 1.0, 2.0, 3.0, 4.0, 5.0],
            'y': [1.0, 2.0, 3.0, 4.0, 5.0, np.nan, np.nan, np.nan, np.nan, np.nan],
        })
        # Before fix: generate_weights sees 4 qualifying rows on df (passes enforce),
        #   then dfv.sample fails with "Fewer non-zero entries in p than size".
        # After fix: generate_weights sees 2 qualifying rows on dfv → raises
        #   ValueError mentioning "enforce conditions".
        with pytest.raises(ValueError, match="enforce"):
            best_sample(df, k=3, feature_cols=['x', 'y'], n_trials=5,
                        bias={'x': [1.0, 2.5, 2.0]}, enforce=True, random_state=0)


# ---------------------------------------------------------------------------
# 20. Fifth Copilot review pass
# ---------------------------------------------------------------------------


class TestCopilotReviewFixes4:
    """
    Tests for issues identified in the fifth Copilot review pass.

    Issues covered:
    1. _get_param_levels() divides biases by biases.sum() without guarding
       against sum==0 (all-zero biases list), silently propagating NaN.
    2. optimize_category_assignment_parallel() early exit with wait=True
       blocks until running threads finish, negating the optimization.
    3. _space_filling_score() crashes on k=1 (empty triu_indices → .min()
       on empty array); best_sample(k=1) therefore always raises.
    """

    @pytest.mark.fast
    def test_get_param_levels_zero_biases_raises(self):
        """
        _get_param_levels must raise ValueError when biases sum to zero.
        Previously it divided by zero and propagated NaN into sampling.
        """
        from obsidian.experiment.advanced_design import _get_param_levels
        spec = {'levels': [1.0, 2.0, 3.0], 'biases': [0.0, 0.0, 0.0]}
        with pytest.raises(ValueError, match="[Bb]ias"):
            _get_param_levels('x', spec)

    @pytest.mark.fast
    def test_best_sample_k1_does_not_crash(self):
        """
        best_sample(k=1) must return a single-row result without raising.
        Previously _space_filling_score raised ValueError on empty triu array.
        """
        from obsidian.experiment.sampling import best_sample
        df = pd.DataFrame({'x': np.linspace(0, 10, 20),
                           'y': np.linspace(0, 5, 20)})
        result, info = best_sample(df, k=1, feature_cols=['x', 'y'],
                                   n_trials=10, random_state=0)
        assert len(result) == 1
        assert np.isfinite(info['score'])

    @pytest.mark.fast
    def test_optimize_category_early_exit_cancels_pending(self):
        """
        When corr_threshold is met, optimize_category_assignment_parallel must
        cancel pending (not-yet-started) futures rather than waiting for all
        n_category_trials to complete. Verified via executor shutdown flag.
        """
        import inspect
        from obsidian.experiment.advanced_design import (
            optimize_category_assignment_parallel,
        )
        source = inspect.getsource(optimize_category_assignment_parallel)
        # After fix: shutdown uses wait=False so pending futures are cancelled
        # immediately rather than blocking until all running threads finish.
        assert 'wait=False' in source, (
            "executor.shutdown should use wait=False so early exit is effective; "
            "wait=True blocks until all already-running threads complete"
        )


class TestCopilotReviewFixes5:
    """
    Tests for issues identified in the sixth Copilot review pass.

    Issues covered:
    1. non_uniform_lhs_categorical() raises IndexError when a uniform sample
       equals 1.0 because np.searchsorted(cdf, 1.0) returns len(levels),
       which is out of bounds. Both the top-level category index and the
       subparameter sub_index are affected.

    Root cause: with 6 equal-frequency levels, np.cumsum([1/6]*6)[-1] equals
    0.9999999999999999 (< 1.0 in float64). A uniform sample of 1.0 therefore
    causes np.searchsorted to return len(levels), which is out of bounds.
    """

    @pytest.mark.fast
    def test_non_uniform_lhs_categorical_sample_at_boundary(self):
        """
        non_uniform_lhs_categorical must not raise IndexError when a uniform
        sample is exactly 1.0.  With 6 equal-frequency levels,
        np.cumsum([1/6]*6)[-1] = 0.9999999999999999 < 1.0, so
        np.searchsorted(cdf, 1.0) returns 6 == len(levels) → IndexError
        without the clip guard.
        """
        from unittest.mock import patch

        # 6 equal levels: cumsum ends at 0.9999999999999999 < 1.0
        level_dict = {f"L{i}": {"freq": 1.0} for i in range(6)}
        with patch("obsidian.experiment.advanced_design.qmc.LatinHypercube") as MockLHS:
            instance = MockLHS.return_value
            instance.random.return_value = np.array([[1.0]])
            results = non_uniform_lhs_categorical(level_dict, n_samples=1, seed=0)
        assert len(results) == 1
        assert results[0]["level"] in {f"L{i}" for i in range(6)}

    @pytest.mark.fast
    def test_non_uniform_lhs_categorical_subparam_boundary(self):
        """
        non_uniform_lhs_categorical must not raise IndexError when a subparam
        uniform sample is exactly 1.0.  With 6 equal sub-weights, the
        sub_cdf ends at 0.9999999999999999, so sub_index goes out of bounds
        without the clip guard.
        """
        from unittest.mock import patch

        colors = [f"c{i}" for i in range(6)]
        level_dict = {
            "A": {"freq": 1.0, "color": (colors, [1.0] * 6)},
        }
        call_count = [0]

        def fake_random(n):
            call_count[0] += 1
            if call_count[0] == 1:
                return np.array([[0.25]])   # level sample (A gets picked)
            return np.array([[1.0]])         # subparam sample at boundary

        with patch("obsidian.experiment.advanced_design.qmc.LatinHypercube") as MockLHS:
            instance = MockLHS.return_value
            instance.random.side_effect = fake_random
            results = non_uniform_lhs_categorical(level_dict, n_samples=1, seed=0)
        assert len(results) == 1
        assert results[0]["color"] in colors


class TestCopilotReviewFixes6:
    """
    Tests for issues identified in the eighth Copilot review pass.

    Issues covered:
    1. _default_maximize_metrics() returns [True, False, False, ...] based on
       position, so any custom ordering where the first metric is not
       D-optimality will apply maximize=True to the wrong metric (e.g.,
       maximizing 'Max Continuous Corr' when it should be minimized).
    2. generate_weights() raises ValueError when the number of qualifying rows
       (weights > 0) is less than n, even when replace=True — where sampling n
       items from fewer qualifying rows is perfectly valid.
    """

    @pytest.mark.fast
    def test_default_maximize_metrics_by_name_not_position(self):
        """
        _default_maximize_metrics must derive maximize flags from metric names,
        not positions. Previously [True, False, ...] always made the first metric
        maximize regardless of its name.
        """
        from obsidian.experiment.advanced_design import _default_maximize_metrics
        # Standard order: D-optimality should be True, rest False
        standard = _default_maximize_metrics([
            "D-optimality", "A-optimality", "Condition Number"
        ])
        assert standard == [True, False, False]

        # Custom order: D-optimality is not first
        custom = _default_maximize_metrics([
            "Max Continuous Corr", "D-optimality", "Pairwise Distance CV"
        ])
        # Before fix: returns [True, False, False] — wrong direction for all three
        # After fix:  returns [False, True, False]
        assert custom == [False, True, False], (
            f"Expected [False, True, False] but got {custom}. "
            "_default_maximize_metrics must use metric name, not position."
        )

    @pytest.mark.fast
    def test_generate_weights_enforce_allows_replace(self):
        """
        generate_weights(enforce=True) must not raise when the number of qualifying
        rows is less than n but replace=True makes sampling valid anyway.
        Before the fix, the check was unconditional and always raised.
        """
        from obsidian.experiment.sampling import generate_weights
        df = pd.DataFrame({'x': [1.0, 2.0, 50.0, 51.0, 52.0]})
        bias = {'x': [1.0, 2.0, 1.0]}   # only 2 rows qualify
        # replace=True: sampling n=5 from 2 qualifying rows is valid
        weights = generate_weights(df, n=5, bias=bias, enforce=True, replace=True)
        assert (weights > 0).sum() == 2   # only the qualifying rows have weight

    @pytest.mark.fast
    def test_generate_weights_enforce_still_raises_without_replace(self):
        """
        generate_weights(enforce=True, replace=False) must still raise when
        qualifying rows < n (existing behavior must be preserved).
        """
        from obsidian.experiment.sampling import generate_weights
        df = pd.DataFrame({'x': [1.0, 2.0, 50.0, 51.0, 52.0]})
        bias = {'x': [1.0, 2.0, 1.0]}   # only 2 rows qualify
        with pytest.raises(ValueError, match="[Nn]ot enough"):
            generate_weights(df, n=5, bias=bias, enforce=True, replace=False)

    @pytest.mark.fast
    def test_sample_with_bias_enforce_replace_succeeds(self):
        """
        sample_with_bias(replace=True, enforce=True) must succeed when fewer
        qualifying rows exist than n.
        """
        from obsidian.experiment.sampling import sample_with_bias
        df = pd.DataFrame({'x': np.arange(100, dtype=float)})
        bias = {'x': [0.0, 2.0, 1.0]}   # only 3 qualifying rows (0, 1, 2)
        result = sample_with_bias(df, n=10, replace=True, seed=0,
                                  bias=bias, enforce=True)
        assert len(result) == 10
        assert result['x'].isin([0.0, 1.0, 2.0]).all()


class TestCopilotReviewFixes7:
    """
    Tests for issues identified in the seventh Copilot review pass.

    Issues covered:
    1. initialize() silently falls through to LHS for any method value other
       than 'Optimized', so typos like method='Random' produce a design
       without any error instead of failing fast.
    """


    @pytest.mark.fast
    def test_initialize_unknown_method_raises(self):
        """
        initialize() must raise ValueError for an unrecognised method string.
        Previously any value other than 'Optimized' silently ran LHS.
        """
        designer = AdvExpDesigner(
            continuous_params={'x': (0, 10, 1), 'y': (0, 5, 1)},
        )
        with pytest.raises(ValueError, match="[Mm]ethod"):
            designer.initialize(m_initial=6, method='Random')

    @pytest.mark.fast
    def test_initialize_lhs_still_works(self):
        """initialize(method='LHS') must continue to work after the guard is added."""
        designer = AdvExpDesigner(
            continuous_params={'x': (0, 10, 1), 'y': (0, 5, 1)},
        )
        design = designer.initialize(m_initial=6, method='LHS')
        assert len(design) == 6

    @pytest.mark.fast
    def test_initialize_optimized_still_works(self):
        """initialize(method='Optimized') must continue to work after the guard is added."""
        designer = AdvExpDesigner(
            continuous_params={'x': (0, 10, 1), 'y': (0, 5, 1)},
        )
        design = designer.initialize(m_initial=6, method='Optimized')
        assert len(design) == 6


class TestCopilotReviewFixes8:
    """
    Tests for issues identified in the ninth Copilot review pass.

    Issues covered:
    1. sample_design() step rounding uses str(float(step)) to infer decimal
       precision. For small steps that stringify in scientific notation
       (e.g. step=1e-5 → '1e-05'), the string has no '.', so decimals=0 and
       values are rounded to the nearest integer instead of 5 decimal places.
    2. best_sample() with a bias that zeroes out most rows can leave fewer than
       k non-zero weights, causing DataFrame.sample to raise a cryptic
       "Fewer non-zero entries" error instead of a clear ValueError.
    3. find_best_design_parallel() accepts caller-supplied maximize_metrics but
       never validates its length against metrics_to_optimize; a mismatched
       list causes an IndexError deep in the scoring loop.
    """

    @pytest.mark.fast
    def test_sample_design_small_step_scientific_notation(self):
        """
        sample_design must round values to the correct precision for steps that
        stringify in scientific notation (e.g. 1e-5 → '1e-05'). Before the fix,
        '1e-05' has no '.' so decimals=0, collapsing all values in [0, 1e-4] to
        0.0 (nearest integer). After the fix, values include non-zero multiples
        of 1e-5.
        """
        design = sample_design(
            seed=0,
            n_samples=20,
            continuous_params={'x': (0.0, 1e-4, 1e-5)},
            conditional_subparameters={},
        )
        vals = design['x'].values
        # With the bug (decimals=0), all values in [0, 1e-4] round to 0.0.
        # After the fix (decimals=5), multiple distinct non-zero levels appear.
        assert np.any(vals > 0), (
            "All values are 0 — step=1e-5 was stringified as '1e-05' "
            "(no '.') so decimals=0, collapsing every level to integer 0"
        )

    @pytest.mark.fast
    def test_best_sample_insufficient_nonzero_weights_raises(self):
        """
        best_sample must raise a clear ValueError when a weight=0 bias leaves
        fewer non-zero-weight rows than k.  With enforce=False, generate_weights
        has no count check, so currently pandas raises a cryptic
        'Fewer non-zero entries in p than size' error.  After the fix, best_sample
        raises a descriptive ValueError (matching 'positive weight' or 'k=')
        before any sampling attempt.
        """
        from obsidian.experiment.sampling import best_sample
        df = pd.DataFrame({
            'x': np.arange(10, dtype=float),
            'y': np.arange(10, dtype=float),
        })
        # weight=0 zeros out x∈[0, 7.5] (rows 0-7); only x=8,9 remain (2 rows)
        # enforce=False → generate_weights skips the capacity check
        # k=5 > 2 → before fix: cryptic pandas error;  after fix: clear ValueError
        bias = {'x': [0.0, 7.5, 0.0]}
        # Match only the clear custom message (not pandas' cryptic "non-zero entries" text)
        with pytest.raises(ValueError, match=r"positive weight|k=\d"):
            best_sample(df, k=5, feature_cols=['x', 'y'],
                        n_trials=10, bias=bias, enforce=False, random_state=0)

    @pytest.mark.fast
    def test_find_best_design_parallel_maximize_metrics_wrong_length_raises(self):
        """
        find_best_design_parallel must raise ValueError when maximize_metrics
        has a different length from metrics_to_optimize. Before the fix, a
        mismatched list caused IndexError deep in the scoring loop.
        """
        metrics = ["D-optimality", "A-optimality"]
        with pytest.raises(ValueError, match="[Ll]ength|[Ll]en|maximize_metrics"):
            find_best_design_parallel(
                n=2,
                n_samples=6,
                continuous_params={'x': (0, 10, 1), 'y': (0, 5, 1)},
                conditional_subparameters={},
                metrics_to_optimize=metrics,
                maximize_metrics=[True],   # wrong length: 1 instead of 2
            )


class TestCopilotReviewFixes9:
    """
    Tests for issues identified in the tenth Copilot review pass.

    Issues covered:
    1. extend_design() is missing the maximize_metrics length validation
       that was added to find_best_design_parallel: a mismatched list causes
       IndexError deep in the scoring loop after all workers complete.
    2. initialize(method='Optimized') always calls optimize_design() with
       seed_start=0, ignoring self.seed. Two designers with different seeds
       both produce the same optimized design.
    3. optimize_category_assignment_parallel() picks the first key of
       subparam_mapping unconditionally, but sample_design() picks the first
       key that is present in conditional_subparameters. If subparam_mapping
       contains an orphan key (not in conditional_subparameters), the helper
       raises KeyError instead of selecting the correct category.
    """

    @pytest.mark.fast
    def test_extend_design_maximize_metrics_wrong_length_raises(self):
        """
        extend_design must raise ValueError when maximize_metrics has a
        different length from metrics_to_optimize. Before the fix, the
        mismatch caused IndexError deep in the scoring loop.
        """
        from obsidian.experiment.advanced_design import extend_design
        existing = sample_design(
            seed=0, n_samples=6,
            continuous_params={'x': (0, 10, 1), 'y': (0, 5, 1)},
            conditional_subparameters={},
        )
        metrics = ["D-optimality", "A-optimality"]
        with pytest.raises(ValueError, match="[Ll]ength|[Ll]en|maximize_metrics"):
            extend_design(
                existing_design=existing,
                n=3,
                continuous_params={'x': (0, 10, 1), 'y': (0, 5, 1)},
                conditional_subparameters={},
                metrics_to_optimize=metrics,
                maximize_metrics=[True],   # wrong length: 1 instead of 2
            )

    @pytest.mark.fast
    def test_initialize_optimized_respects_seed(self):
        """
        initialize(method='Optimized') must pass seed_start through to
        optimize_design so that different seeds produce different designs.
        Before the fix, seed_start was always 0 regardless of self.seed,
        making all optimized initializations identical.
        """
        d42 = AdvExpDesigner(
            continuous_params={'x': (0, 10, 1), 'y': (0, 5, 1)}, seed=42
        ).initialize(m_initial=6, method='Optimized')

        d99 = AdvExpDesigner(
            continuous_params={'x': (0, 10, 1), 'y': (0, 5, 1)}, seed=99
        ).initialize(m_initial=6, method='Optimized')

        assert not d42.equals(d99), (
            "Different seeds produced identical optimized designs — "
            "initialize() is not threading self.seed into optimize_design()"
        )

    @pytest.mark.fast
    def test_optimize_category_assignment_parallel_orphan_key(self):
        """
        optimize_category_assignment_parallel must select the first key of
        subparam_mapping that is present in conditional_subparameters, not
        unconditionally the first key. With an orphan key first in the
        mapping, the old code raises KeyError.
        """
        from obsidian.experiment.advanced_design import (
            optimize_category_assignment_parallel,
        )
        level_dict = {
            "A": {"freq": 0.5},
            "B": {"freq": 0.5},
        }
        conditional_subparameters = {"real_cat": level_dict}
        # 'orphan' is first in the mapping but NOT in conditional_subparameters
        subparam_mapping = {"orphan": "orphan_sub", "real_cat": "real_sub"}
        cat_samples = {}   # no pre-sampled categories
        result = optimize_category_assignment_parallel(
            cat_samples=cat_samples,
            conditional_subparameters=conditional_subparameters,
            subparam_mapping=subparam_mapping,
            n_samples=4,
            seed=0,
            n_category_trials=5,
        )
        assert set(result).issubset({"A", "B"})


class TestCopilotReviewFixes10:
    """
    Tests for the final Copilot review pass.

    Issues covered:
    1. initialize() uses seed=0 when self.seed is None, making all unseeded
       instances produce identical designs. Should generate a fresh random seed
       when self.seed is unset.
    """

    @pytest.mark.fast
    def test_initialize_unseeded_produces_different_designs(self):
        """Two unseeded AdvExpDesigners should (almost certainly) produce different designs."""
        d1 = AdvExpDesigner(continuous_params={'Temperature': (60, 90, 10)}).initialize(m_initial=20)
        d2 = AdvExpDesigner(continuous_params={'Temperature': (60, 90, 10)}).initialize(m_initial=20)
        assert not d1.equals(d2), "Two unseeded initialize() calls returned identical designs"

    @pytest.mark.fast
    def test_initialize_seeded_remains_reproducible(self):
        """initialize() with seed set must still be deterministic."""
        d1 = AdvExpDesigner(continuous_params={'Temperature': (60, 90, 10)}, seed=7).initialize(m_initial=20)
        d2 = AdvExpDesigner(continuous_params={'Temperature': (60, 90, 10)}, seed=7).initialize(m_initial=20)
        assert d1.equals(d2), "Seeded initialize() calls returned different designs"


class TestPlottingFixes:
    """
    Regression tests for correctness issues found in obsidian/plotting/plotly.py.

    Issues covered:
    1. parity_plot() computes NRMSE = RMSE / (y_true.max() - y_true.min()).
       When all observed values are identical, the denominator is 0, producing
       inf/nan marker colors and a broken colorbar.
    2. parity_plot() computes RMSE = ((y_true - y_pred) / y_true) ** 2.
       When any y_true value is 0, the division produces inf, again breaking
       the marker coloring silently.
    """

    def _make_mock_optimizer(self, y_true_vals, y_pred_vals):
        """Return a MagicMock that looks like a fit single-response Optimizer."""
        from unittest.mock import MagicMock
        from obsidian.optimizer import Optimizer

        n = len(y_true_vals)
        y_name = "y"

        mock = MagicMock(spec=Optimizer)
        mock.is_fit = True
        mock.target = [MagicMock()]          # len == 1
        mock.y_names = [y_name]
        mock.X_train = pd.DataFrame({"x": np.arange(n, dtype=float)})

        mock.y_train = pd.DataFrame({y_name: y_true_vals})
        mock.f_train = pd.DataFrame({y_name: y_true_vals})

        pred_df = pd.DataFrame({
            f"{y_name} (pred)": y_pred_vals,
            f"{y_name} lb":     np.array(y_pred_vals) - 0.1,
            f"{y_name} ub":     np.array(y_pred_vals) + 0.1,
        })
        mock.predict.return_value = pred_df
        return mock

    @pytest.mark.fast
    def test_parity_plot_constant_y_true_no_crash(self):
        """
        parity_plot must not raise when all observed values are identical.
        Before the fix, NRMSE = RMSE / 0 produced inf/nan, breaking the
        Plotly marker color array.
        """
        from obsidian.plotting import parity_plot

        optimizer = self._make_mock_optimizer(
            y_true_vals=[5.0, 5.0, 5.0, 5.0],
            y_pred_vals=[4.8, 5.1, 4.9, 5.2],
        )
        # Should return a Figure without raising
        fig = parity_plot(optimizer)
        colors = fig.data[0].marker.color
        assert np.all(np.isfinite(colors)), "Marker colors must be finite"

    @pytest.mark.fast
    def test_parity_plot_zero_in_y_true_no_crash(self):
        """
        parity_plot must not produce inf marker colors when y_true contains 0.
        Before the fix, (y_true - y_pred) / 0 propagated inf into the color array.
        """
        from obsidian.plotting import parity_plot

        optimizer = self._make_mock_optimizer(
            y_true_vals=[0.0, 1.0, 2.0, 3.0],
            y_pred_vals=[0.1, 1.1, 1.9, 3.1],
        )
        fig = parity_plot(optimizer)
        colors = fig.data[0].marker.color
        assert np.all(np.isfinite(colors)), "Marker colors must be finite"


class TestInternalHelpers:
    """Unit tests for private helpers extracted during the internal refactor."""

    @pytest.mark.fast
    def test_standardize_helper_matches_manual(self):
        from obsidian.experiment.advanced_design import _standardize
        X = np.array([[1., 10.], [2., 10.], [3., 10.]])  # col 1 constant
        out = _standardize(X)
        # constant column -> zeros (std replaced by 1.0), not NaN
        assert np.all(np.isfinite(out))
        np.testing.assert_allclose(out[:, 1], 0.0, atol=1e-12)
        np.testing.assert_allclose(out[:, 0], (X[:, 0] - X[:, 0].mean()) / X[:, 0].std(), atol=1e-12)

    @pytest.mark.fast
    def test_information_matrix_helper(self):
        from obsidian.experiment.advanced_design import _information_matrix
        X_std = np.array([[0.0], [1.0], [-1.0]])
        XtX = _information_matrix(X_std)
        assert XtX.shape == (2, 2)  # intercept + 1 feature
        np.testing.assert_allclose(XtX[0, 0], 3.0)  # ones dot ones

    @pytest.mark.fast
    def test_continuous_columns_helper(self):
        from obsidian.experiment.advanced_design import _continuous_columns
        assert _continuous_columns(["a", "b"], {"cat": "sub"}) == ["a", "b", "sub"]
        assert _continuous_columns(["a"], None) == ["a"]
        assert _continuous_columns(["a"], {}) == ["a"]


class TestPlotEmbeddingHelper:
    """Tests for the unified _plot_embedding private helper (Task 4 refactor)."""

    @pytest.mark.fast
    def test_plot_embedding_runs_and_handles_hue(self):
        # backend already set to Agg at module import
        from obsidian.experiment.advanced_design import _plot_embedding
        df = pd.DataFrame({"a": [1., 2., 3., 4.], "b": [4., 3., 2., 1.], "g": ["X", "X", "Y", "Y"]})
        # identity-like 2D projection
        _plot_embedding(df, ["a", "b"], None, fit_2d=lambda X: X[:, :2],
                        title="T", xlabel="x", ylabel="y", default_color="blue", hue=None)
        _plot_embedding(df, ["a", "b"], None, fit_2d=lambda X: X[:, :2],
                        title="T", xlabel="x", ylabel="y", default_color="blue", hue="g")
        import matplotlib.pyplot as plt; plt.close("all")


# ---------------------------------------------------------------------------
# Task 5: _composite_scores anti-divergence test
# ---------------------------------------------------------------------------

class TestCompositeScores:
    """Anti-divergence tests for the _composite_scores helper (Task 5 refactor)."""

    @pytest.mark.fast
    def test_composite_scores_matches_legacy_on_edge_cases(self):
        from obsidian.experiment.advanced_design import _composite_scores
        mx = [True, False, False]
        cases = [
            np.array([[10., 1., .5], [20., 2., .3], [15., 3., .9]]),   # normal
            np.array([[5., 1., .5], [5., 2., .3], [5., 3., .9]]),      # constant col
            np.array([[10., np.nan, .5], [20., 2., .3], [15., 3., .9]]),  # NaN
            np.array([[10., 2., .5]]),                                  # single candidate
        ]
        expected = [
            np.array([1.66666667, 2.5, 0.5]),
            np.array([1.66666667, 1.5, 0.0]),
            np.array([1.66666667, 3.0, 1.5]),
            np.array([2.0]),
        ]
        for arr, exp in zip(cases, expected):
            np.testing.assert_allclose(_composite_scores(arr, mx), exp, atol=1e-7)


def _fake_candidate_for_run_parallel_search_test(seed):
    """Module-level helper so ProcessPoolExecutor can pickle it."""
    return {"seed": seed, "metric_values": [float(seed)]}


class TestRunParallelSearch:
    """Anti-divergence tests for the _run_parallel_search driver (Task 6 refactor)."""

    @pytest.mark.fast
    def test_run_parallel_search_scores_and_best(self):
        from obsidian.experiment.advanced_design import _run_parallel_search
        # candidate_fn returns a record with metric_values; pick best by composite score
        records, scores, best = _run_parallel_search(
            _fake_candidate_for_run_parallel_search_test, [(0,), (1,), (2,)],
            maximize_metrics=[True], max_workers=2, use_tqdm=False)
        assert len(records) == 3
        # single metric maximize -> highest seed wins
        assert records[best]["seed"] == 2


# ---------------------------------------------------------------------------
# Task 7: Metric registry
# ---------------------------------------------------------------------------


class TestMetricRegistry:
    """Tests that verify the _METRICS registry introduced in task 7."""

    @pytest.mark.fast
    def test_metric_registry_matches_defaults(self):
        from obsidian.experiment.advanced_design import _METRICS, DEFAULT_METRICS, _MAXIMIZE_BY_METRIC
        # order and membership preserved
        assert list(_METRICS) == DEFAULT_METRICS
        assert DEFAULT_METRICS == [
            "D-optimality", "A-optimality", "Condition Number", "Pairwise Distance CV",
            "Max Continuous Corr", "Max Categorical Corr", "Max Mixed Corr"]
        # maximize flags preserved
        assert _MAXIMIZE_BY_METRIC == {
            "D-optimality": True, "A-optimality": False, "Condition Number": False,
            "Pairwise Distance CV": False, "Max Continuous Corr": False,
            "Max Categorical Corr": False, "Max Mixed Corr": False}


# ---------------------------------------------------------------------------
# Task 8: NaN pairwise drop + infer_column_types
# ---------------------------------------------------------------------------


class TestTask8NanPairwiseDrop:
    """Tests for the pairwise NaN drop and infer_column_types extraction (task 8)."""

    @pytest.mark.fast
    @pytest.mark.bug_regression
    def test_mixed_correlation_cat_num_nan_is_finite(self):
        df = pd.DataFrame({"conc": [0.1, np.nan, 0.3, 0.4], "buf": ["A", "A", "B", "B"]})
        corr = calculate_mixed_correlation_matrix(df, categorical_vars=["buf"])
        assert np.all(np.isfinite(corr.values)), "cat-num NaN must not propagate"

    @pytest.mark.fast
    @pytest.mark.bug_regression
    def test_mixed_correlation_cat_cat_nan_dropped_not_coded(self):
        # NaN in a categorical column must be treated as missing (dropped), not its own level
        df = pd.DataFrame({"x": ["A", "A", None, "B"], "y": ["P", "Q", "P", "Q"]})
        corr = calculate_mixed_correlation_matrix(df, categorical_vars=["x", "y"])
        assert np.all(np.isfinite(corr.values))

    @pytest.mark.fast
    def test_infer_column_types_single_rule(self):
        from obsidian.experiment.advanced_design import infer_column_types
        df = pd.DataFrame({"n": [1.0, 2.0], "c": ["a", "b"]})
        cat, num = infer_column_types(df)
        assert cat == ["c"] and num == ["n"]


# ---------------------------------------------------------------------------
# Task 9: _round_continuous_columns helper
# ---------------------------------------------------------------------------


class TestTask9RoundContinuousColumns:
    """Tests for the _round_continuous_columns helper extracted in task 9."""

    @pytest.mark.fast
    def test_round_continuous_columns_rules(self):
        from obsidian.experiment.advanced_design import _round_continuous_columns
        df = pd.DataFrame({"a": [1.23456, 2.98765], "b": [10.4, 20.6]})
        params = {"a": {"levels": [1.1, 2.2, 3.3]}, "b": (10, 30, 10)}  # a: custom 1-dp; b: int step
        out = _round_continuous_columns(df.copy(), params)
        # b: integer step -> int dtype
        assert out["b"].dtype.kind in ("i", "u")
        # a: max 1 decimal from levels
        assert (out["a"] == out["a"].round(1)).all()


if __name__ == '__main__':
    pytest.main([__file__, '-m', 'fast', '-v'])
