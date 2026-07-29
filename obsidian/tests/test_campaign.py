"""PyTests for obsidian.campaign"""

import json
import warnings

import pandas as pd
import pytest
from numpy.testing import assert_array_equal

from obsidian.campaign import Campaign, Explainer, calc_ofat_ranges
from obsidian.constraints import Blank_Constraint, L1_Constraint
from obsidian.exceptions import IncompatibleObjectiveError, UnfitError
from obsidian.experiment import ExpDesigner, Simulator, AdvExpDesigner
from obsidian.experiment.benchmark import shifted_parab, two_leaves
from obsidian.objectives import Identity_Objective
from obsidian.parameters import Param_Continuous, ParamSpace, Target
from obsidian.plotting import plot_interactions, plot_ofat_ranges
from obsidian.tests.param_configs import X_sp_cont_ndims, X_sp_default
from obsidian.tests.utils import DEFAULT_MOO_PATH, DEFAULT_SOO_PATH, equal_state_dicts

target_test = [
    [
        Target(name="Response 1", f_transform="Standard", aim="max"),
        Target(name="Response 2", f_transform="Standard", aim="max"),
    ],
    Target(name="Response", f_transform="Standard", aim="max"),
]


@pytest.mark.parametrize(
    "X_space, sim_fcn, target",
    [(X_sp_cont_ndims[2], two_leaves, target_test[0]), (X_sp_default, shifted_parab, target_test[1])],
)
def test_campaign_basics(X_space, sim_fcn, target, rng_mode):
    # Standard usage
    campaign = Campaign(X_space, target, seed=114514)
    simulator = Simulator(X_space, sim_fcn, eps=0.05, rng=114514)
    X0 = campaign.suggest()
    y0 = simulator.simulate(X0)
    Z0 = pd.concat([X0, y0], axis=1)

    # Set an objective, suggest, clear
    campaign.set_objective(Identity_Objective(mo=len(campaign.target) > 1))
    campaign.suggest()
    campaign.clear_objective()

    # Add, fit, clear, examine
    campaign.add_data(Z0)
    campaign.fit()
    campaign.clear_data()
    campaign.y
    campaign.__repr__()

    # Add with iteration, examine, fit, analyze
    Z0["Iteration"] = 5
    campaign.add_data(Z0)
    campaign.y
    campaign.fit()
    campaign.response_max
    campaign.X_best

    # Serialize, deserialize, re-serialize
    obj_dict = campaign.save_state()
    campaign2 = Campaign.load_state(obj_dict)
    obj_dict2 = campaign2.save_state()
    assert equal_state_dicts(obj_dict, obj_dict2), "Error during serialization"


@pytest.mark.parametrize(
    "X_space, sim_fcn, target",
    [(X_sp_cont_ndims[2], two_leaves, target_test[0]), (X_sp_default, shifted_parab, target_test[1])],
)
def test_campaign_unfitted_save_load(X_space, sim_fcn, target, rng_mode):
    """Test that unfitted campaigns can be saved and loaded, then independently fitted"""
    # Create unfitted campaign
    campaign1 = Campaign(X_space, target, seed=114514)

    # Save unfitted campaign (will trigger warning from optimizer.load_state)
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        obj_dict = campaign1.save_state()

        # Load unfitted campaign
        campaign2 = Campaign.load_state(obj_dict)

        # Expect warning from loading unfitted optimizer
        assert any("unfitted" in str(warning.message).lower() for warning in w)

    # Verify both campaigns are unfitted
    assert not campaign1.optimizer.is_fit
    assert not campaign2.optimizer.is_fit

    # Generate same data for both campaigns
    simulator = Simulator(X_space, sim_fcn, eps=0.05, rng=114514)
    X0 = campaign1.suggest()
    y0 = simulator.simulate(X0)
    Z0 = pd.concat([X0, y0], axis=1)

    # Add iteration column
    Z0["Iteration"] = 1

    # Fit both campaigns independently with same data
    campaign1.add_data(Z0)
    campaign1.fit()

    campaign2.add_data(Z0)
    campaign2.fit()

    # Both should now be fitted
    assert campaign1.optimizer.is_fit
    assert campaign2.optimizer.is_fit

    # Verify they produce the same results
    # Compare response_max
    rm1 = campaign1.response_max
    rm2 = campaign2.response_max
    if isinstance(rm1, pd.Series) and isinstance(rm2, pd.Series):
        assert_array_equal(rm1.to_numpy(), rm2.to_numpy())
    elif not isinstance(rm1, pd.Series) and not isinstance(rm2, pd.Series):
        assert abs(float(rm1) - float(rm2)) < 1e-6
    else:
        raise AssertionError("response_max types don't match between campaigns")

    # Compare X_best (use DataFrame comparison to handle categorical variables)
    pd.testing.assert_frame_equal(campaign1.X_best, campaign2.X_best)

    # Compare predictions at training points
    y_pred1 = campaign1.optimizer.predict(campaign1.optimizer.X_train)
    y_pred2 = campaign2.optimizer.predict(campaign2.optimizer.X_train)
    assert_array_equal(y_pred1.values, y_pred2.values)

    # Verify save/load serialization works after fitting
    obj_dict1 = campaign1.save_state()
    obj_dict2 = campaign2.save_state()
    assert equal_state_dicts(obj_dict1, obj_dict2), "Fitted campaigns should have identical states"


# Load default
with open(DEFAULT_MOO_PATH) as json_file:
    obj_dict = json.load(json_file)
    # Override seeds for determinism with new RNG control
    obj_dict["seed"] = 114514
    obj_dict["optimizer"]["opt_attrs"]["seed"] = 114514
campaign = Campaign.load_state(obj_dict)
X_space = campaign.X_space
target = campaign.target


def test_explain():
    # SOO includes discrete variables
    with open(DEFAULT_SOO_PATH) as json_file:
        obj_dict = json.load(json_file)
        # Override seeds for determinism with new RNG control
        obj_dict["seed"] = 114514
        obj_dict["optimizer"]["opt_attrs"]["seed"] = 114514
    campaign = Campaign.load_state(obj_dict)

    # Standard usage
    exp = Explainer(campaign.optimizer)
    exp.shap_explain(n=50)
    exp.__repr__

    # Test SHAP plots
    exp.shap_summary()
    exp.shap_summary_bar()

    # Test PDP-ICE, with options and  discrete variables
    exp.shap_pdp_ice(ind=3, ice_color_var=None, npoints=10)
    exp.shap_pdp_ice(ind=0, ice_color_var=3, npoints=None)
    exp.shap_pdp_ice(ind=3, hist=True)
    exp.shap_pdp_ice(ind=(0, 3), npoints=5)
    exp.shap_pdp_ice(ind=(3, 0), npoints=None)

    # Test pairwise SHAP analysis, with options
    X_new = campaign.X.iloc[0, :]
    X_ref = campaign.X.loc[1, :]
    df_shap_value_new, fig_bar, fig_line = exp.shap_single_point(X_new)
    df_shap_value_new, fig_bar, fig_line = exp.shap_single_point(X_new, X_ref=X_ref)

    # Test sensitivity analysis, with options
    df_sens = exp.sensitivity()
    df_sens = exp.sensitivity(X_ref=X_ref)


X_ref_test = [None, campaign.X.iloc[campaign.y.idxmax()["Response 1"], :]]


@pytest.mark.parametrize("X_ref", X_ref_test)
def test_analysis(X_ref):
    # OFAT ranges with/out interactions and with/out X_ref
    ofat_ranges, _ = calc_ofat_ranges(campaign.optimizer, threshold=0.5, X_ref=X_ref, calc_interacts=False)
    ofat_ranges, cor = calc_ofat_ranges(campaign.optimizer, threshold=0.5, X_ref=X_ref)
    plot_interactions(campaign.optimizer, cor)
    plot_ofat_ranges(campaign.optimizer, ofat_ranges)

    # OFAT ranges where all results should be NaN
    ofat_ranges, cor = calc_ofat_ranges(campaign.optimizer, threshold=9999, X_ref=X_ref)
    plot_interactions(campaign.optimizer, cor)
    plot_ofat_ranges(campaign.optimizer, ofat_ranges)


@pytest.mark.fast
def test_suggest_kwarg_overrides():
    """Regression test: passing objective= or out_constraints= in optim_kwargs
    must not raise TypeError due to duplicate keyword arguments."""
    from obsidian.constraints import L1_Constraint

    # campaign is the module-level MOO fixture (already fit)
    override_objective = Identity_Objective(mo=True)
    override_constraint = L1_Constraint(target=campaign.target, offset=1.0)

    # Passing objective= explicitly — caller value should win, no TypeError
    result = campaign.suggest(objective=override_objective)
    assert result is not None, "suggest() with explicit objective= should return a result"

    # Passing out_constraints= explicitly as None — should not collide with self.output_constraints
    result = campaign.suggest(out_constraints=None)
    assert result is not None, "suggest() with explicit out_constraints=None should return a result"

    # Passing out_constraints= explicitly as a real constraint
    result = campaign.suggest(out_constraints=override_constraint)
    assert result is not None, "suggest() with explicit out_constraints= should return a result"

    # Passing both together
    result = campaign.suggest(objective=override_objective, out_constraints=override_constraint)
    assert result is not None, "suggest() with both explicit kwargs should return a result"


@pytest.mark.fast
def test_suggest_attribute_passthrough():
    """Test that campaign.objective and campaign.output_constraints are passed through
    and actually used by optimizer.suggest()."""
    from typing import Callable
    from obsidian.constraints import Output_Constraint
    import torch

    # Test case 1: Campaign with objective set, suggest() without override
    # Create a tracked objective that records when it's called
    class TrackedObjective(Identity_Objective):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self.was_called = False

        def forward(self, *args, **kwargs):
            self.was_called = True
            return super().forward(*args, **kwargs)

    campaign_with_obj = Campaign(X_space, target)
    campaign_with_obj.add_data(campaign.data)
    campaign_with_obj.fit()
    tracked_objective = TrackedObjective(mo=True)
    campaign_with_obj.set_objective(tracked_objective)

    result = campaign_with_obj.suggest()
    assert result is not None, "suggest() should return a result"
    assert tracked_objective.was_called, \
        "Campaign objective should be invoked during suggest()"

    # Test case 2: Campaign with output_constraints set, suggest() without override
    # Create a tracked constraint that records when it's called
    class TrackedConstraint(Output_Constraint):
        def __init__(self, target):
            super().__init__(target)
            self.was_called = False

        def forward(self, scale: bool = True) -> Callable:
            self.was_called = True
            def constraint(samples: torch.Tensor) -> torch.Tensor:
                # Return dummy feasible values (negative = feasible)
                return -torch.ones_like(samples[..., 0])
            return constraint

    campaign_with_constraints = Campaign(X_space, target)
    campaign_with_constraints.add_data(campaign.data)
    campaign_with_constraints.fit()
    tracked_constraint = TrackedConstraint(target=target)
    campaign_with_constraints.constrain_outputs(tracked_constraint)

    result = campaign_with_constraints.suggest()
    assert result is not None, \
        "suggest() should return a result"
    assert tracked_constraint.was_called, \
        "Campaign output_constraints should be invoked during suggest()"


# VALIDATION TESTS - Force errors to be raised in object usage


@pytest.mark.fast
def test_campaign_validation():
    # Missing X names
    random_data = pd.DataFrame(data={"A": [1, 2, 3], "B": [4, 5, 6]})
    with pytest.raises(KeyError):
        campaign.add_data(random_data)

    # Missing Y names
    with pytest.raises(KeyError):
        campaign.add_data(campaign.X)

    # Missing data
    with pytest.raises(ValueError):
        campaign2 = Campaign(X_space, target)
        campaign2.fit()

    # Bad objective
    with pytest.raises(IncompatibleObjectiveError):
        campaign.set_objective(Identity_Objective(mo=False))


@pytest.mark.fast
def test_campaign_warns_all_tracking_only_targets(rng_mode):
    tracking_only_target = Target(name="Response", f_transform="Standard", aim="max", tracking_only=True)
    with pytest.warns(UserWarning, match="All targets are tracking-only"):
        Campaign(X_sp_default, tracking_only_target, seed=114514)


@pytest.mark.fast
def test_explainer_validation():
    # Unfit optimizer
    campaign2 = Campaign(X_space, target)
    with pytest.raises(UnfitError):
        exp = Explainer(campaign2.optimizer)

    # Unfit SHAP
    exp = Explainer(campaign.optimizer)
    with pytest.raises(UnfitError):
        exp.shap_summary()

    # Unfit SHAP
    with pytest.raises(UnfitError):
        exp.shap_summary_bar()

    # Unfit SHAP
    with pytest.raises(UnfitError):
        exp.shap_single_point(X_new=campaign.optimizer.X_best_f)

    random_data = pd.DataFrame(data={"A": [1], "B": [4]})
    long_data = pd.DataFrame(data={"Parameter 1": [1, 2], "Parameter 2": [1, 2]})

    # Missing X names
    with pytest.raises(ValueError):
        exp.shap_explain(n=50, X_ref=random_data)

    # X_ref > 1 row
    with pytest.raises(ValueError):
        exp.shap_explain(n=50, X_ref=long_data)

    exp.shap_explain(n=50)

    # Missing X names
    with pytest.raises(ValueError):
        exp.shap_single_point(X_new=random_data)

    # Missing X names
    with pytest.raises(ValueError):
        exp.shap_single_point(X_new=campaign.optimizer.X_best_f, X_ref=random_data)

    # Missing X names
    with pytest.raises(ValueError):
        exp.sensitivity(X_ref=random_data)

    # X_ref > 1 row
    with pytest.raises(ValueError):
        exp.sensitivity(X_ref=long_data)


def test_campaign_suggest_consistency():
    """Test that Campaign suggest gives consistent results with same seed"""
    from obsidian import create_rng_manager
    from obsidian.experiment import ExpDesigner

    X_space = X_sp_default
    target = Target(name="Response", f_transform="Standard", aim="max")
    simulator = Simulator(X_space, shifted_parab, eps=0.05, rng=114514)

    # Create two campaigns with same seed
    campaign1 = Campaign(X_space, target, seed=114514)
    campaign2 = Campaign(X_space, target, seed=114514)

    # Initialize and fit both campaigns with same data
    X0 = campaign1.initialize(m_initial=5, method="LHS")
    y0 = simulator.simulate(X0)
    Z0 = pd.concat([X0, y0], axis=1)

    campaign1.add_data(Z0)
    campaign2.add_data(Z0)

    campaign1.fit()
    campaign2.fit()

    # Suggest with EI (deterministic acquisition function)
    X_suggest1, _ = campaign1.suggest(m_batch=3, acquisition=["EI"])
    X_suggest2, _ = campaign2.suggest(m_batch=3, acquisition=["EI"])

    # Results should be identical
    pd.testing.assert_frame_equal(X_suggest1, X_suggest2)


def test_campaign_manual_seed_override():
    """Test that Campaign.suggest() respects manual_seed parameter"""
    from obsidian.experiment import ExpDesigner

    X_space = X_sp_default
    target = Target(name="Response", f_transform="Standard", aim="max")
    simulator = Simulator(X_space, shifted_parab, eps=0.05, rng=114514)

    # Create campaign
    campaign = Campaign(X_space, target, seed=114514)

    # Initialize and fit
    X0 = campaign.initialize(m_initial=5, method="LHS")
    y0 = simulator.simulate(X0)
    Z0 = pd.concat([X0, y0], axis=1)
    campaign.add_data(Z0)
    campaign.fit()

    # Suggest with default seed
    X_suggest1, _ = campaign.suggest(m_batch=3, acquisition=["EI"])

    # Suggest with manual seed override
    X_suggest2, _ = campaign.suggest(m_batch=3, acquisition=["EI"], manual_seed=999)

    # Results should be different due to manual seed override
    assert not X_suggest1.equals(X_suggest2)


def test_campaign_save_load_rng_state():
    """Test that Campaign save/load preserves RNG state"""
    from obsidian.optimizer import BayesianOptimizer

    X_space = X_sp_default
    target = Target(name="Response", f_transform="Standard", aim="max")
    simulator = Simulator(X_space, shifted_parab, eps=0.05, rng=114514)

    # Create campaign with specific seed
    campaign1 = Campaign(X_space, target, seed=12345)

    # Initialize, add data, and fit
    X0 = campaign1.initialize(m_initial=5, method="LHS")
    y0 = simulator.simulate(X0)
    Z0 = pd.concat([X0, y0], axis=1)
    campaign1.add_data(Z0)
    campaign1.fit()

    # Make a suggestion
    X_suggest1, _ = campaign1.suggest(m_batch=2, acquisition=["EI"])

    # Save state
    state = campaign1.save_state()

    # Load into new campaign
    campaign2 = Campaign.load_state(state)

    # RNG state should be preserved
    assert campaign2.seed == 12345
    assert hasattr(campaign2, "rng")
    assert campaign2.rng.seed == 12345

    # Make another suggestion - should continue from saved RNG state
    X_suggest2, _ = campaign2.suggest(m_batch=2, acquisition=["EI"])

    # Both campaigns should have the same optimizer state
    assert campaign1.optimizer.is_fit == campaign2.optimizer.is_fit

    # Test optimizer save/load specifically
    opt_state = campaign1.optimizer.save_state()
    opt_loaded = BayesianOptimizer.load_state(opt_state)

    # Check RNG state is preserved in optimizer
    if "rng_state" in opt_state:
        assert opt_loaded.seed == campaign1.optimizer.seed
        assert hasattr(opt_loaded, "rng")


def test_shared_rng_flag():
    """Test Campaign._owns_rng flag when sharing RNG with optimizer"""
    from obsidian import create_rng_manager
    from obsidian.optimizer import BayesianOptimizer

    X_space = X_sp_default
    target = Target(name="Response", f_transform="Standard", aim="max")

    # Create shared RNG
    shared_rng = create_rng_manager(42)

    # Create optimizer with shared RNG
    optimizer = BayesianOptimizer(X_space, rng=shared_rng)

    # Creating campaign with shared RNG should set _owns_rng=False
    campaign = Campaign(X_space, target, optimizer=optimizer, rng=shared_rng)
    assert campaign.rng is optimizer.rng
    assert campaign._owns_rng is False


def test_campaign_copy_fit(rng_mode):
    """Test that copy() of a fit campaign preserves data, targets, objective, and fit state."""
    X_space = X_sp_default
    target = Target(name="Response", f_transform="Standard", aim="max")
    simulator = Simulator(X_space, shifted_parab, eps=0.05, rng=114514)

    campaign_orig = Campaign(X_space, target, seed=114514)
    X0 = campaign_orig.initialize(m_initial=5, method="LHS")
    y0 = simulator.simulate(X0)
    Z0 = pd.concat([X0, y0], axis=1)
    campaign_orig.add_data(Z0)
    campaign_orig.fit()

    campaign_copy = campaign_orig.copy()

    # Fit state preserved
    assert campaign_copy.optimizer.is_fit

    # y_names preserved
    assert campaign_copy.y_names == campaign_orig.y_names

    # Data preserved (same shape and values)
    assert campaign_copy.data.shape == campaign_orig.data.shape
    pd.testing.assert_frame_equal(
        campaign_copy.data[campaign_orig.y_names].reset_index(drop=True),
        campaign_orig.data[campaign_orig.y_names].reset_index(drop=True),
    )

    # No output constraints in either
    assert campaign_copy.output_constraints is None

    # Full save/load state equality
    assert equal_state_dicts(campaign_orig.save_state(), campaign_copy.save_state())


def test_campaign_copy_unfit(rng_mode):
    """Test that copy() works on an unfit campaign and preserves the unfit state."""
    X_space = X_sp_default
    target = Target(name="Response", f_transform="Standard", aim="max")

    with warnings.catch_warnings(record=True):
        warnings.simplefilter("always")
        campaign_orig = Campaign(X_space, target, seed=114514)
        campaign_copy = campaign_orig.copy()

    assert not campaign_copy.optimizer.is_fit
    assert campaign_copy.y_names == campaign_orig.y_names
    assert campaign_copy.output_constraints is None


def test_campaign_copy_multiple_output_constraints(rng_mode):
    """Test that copy() preserves all output constraints when multiple are set.

    This is a regression test: load_state() used to call constrain_outputs() once
    per constraint (overwriting each time), so only the last constraint survived.
    """
    X_space = X_sp_default
    target = Target(name="Response", f_transform="Standard", aim="max")
    simulator = Simulator(X_space, shifted_parab, eps=0.05, rng=114514)

    campaign_orig = Campaign(X_space, target, seed=114514)
    X0 = campaign_orig.initialize(m_initial=5, method="LHS")
    y0 = simulator.simulate(X0)
    Z0 = pd.concat([X0, y0], axis=1)
    campaign_orig.add_data(Z0)
    campaign_orig.fit()

    # Set two output constraints
    c1 = Blank_Constraint(campaign_orig.target)
    c2 = L1_Constraint(campaign_orig.target, offset=1)
    campaign_orig.constrain_outputs([c1, c2])

    assert len(campaign_orig.output_constraints) == 2

    campaign_copy = campaign_orig.copy()

    # Both constraints must survive the copy
    assert campaign_copy.output_constraints is not None, "output_constraints should not be None after copy"
    assert (
        len(campaign_copy.output_constraints) == 2
    ), f"Expected 2 output constraints after copy, got {len(campaign_copy.output_constraints)}"

    # Verify constraint types are preserved in order
    assert campaign_copy.output_constraints[0].__class__.__name__ == "Blank_Constraint"
    assert campaign_copy.output_constraints[1].__class__.__name__ == "L1_Constraint"


# ---------------------------------------------------------------------------
# Campaign designer persistence tests
# ---------------------------------------------------------------------------


@pytest.mark.fast
def test_campaign_save_includes_designer():
    """Campaign.save_state() must include a 'designer' entry."""
    X_space = ParamSpace(
        [
            Param_Continuous("Temperature", 60, 90),
            Param_Continuous("pH", 6.0, 8.0),
        ]
    )
    target = Target(name="Yield", f_transform="Standard", aim="max")
    designer = ExpDesigner(X_space, seed=42)
    campaign = Campaign(X_space, target, designer=designer)

    state = campaign.save_state()
    assert "designer" in state
    assert state["designer"]["name"] == "ExpDesigner"
    assert state["designer"]["seed"] == 42


@pytest.mark.fast
def test_campaign_reload_preserves_basic_designer_seed(rng_mode):
    """Custom seed on basic designer must survive campaign save/load."""
    X_space = ParamSpace(
        [
            Param_Continuous("Temperature", 60, 90),
            Param_Continuous("pH", 6.0, 8.0),
        ]
    )
    target = Target(name="Yield", f_transform="Standard", aim="max")
    designer = ExpDesigner(X_space, seed=17)
    campaign = Campaign(X_space, target, designer=designer)

    state = campaign.save_state()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        campaign2 = Campaign.load_state(state)

    assert isinstance(campaign2.designer, ExpDesigner)
    assert not isinstance(campaign2.designer, AdvExpDesigner)
    assert campaign2.designer.seed == 17


@pytest.mark.fast
def test_campaign_reload_preserves_adv_designer_full_config(rng_mode):
    """AdvExpDesigner type and all config must survive Campaign.save_state() / load_state()."""
    X_space = ParamSpace(
        [
            Param_Continuous("Temperature", 60, 90),
            Param_Continuous("pH", 6.0, 8.0),
        ]
    )
    target = Target(name="Yield", f_transform="Standard", aim="max")

    designer = AdvExpDesigner(
        continuous_params={
            "Temperature": {"levels": [60, 70, 80, 90], "biases": [0.05, 0.10, 0.35, 0.50]},
            "pH": [6.0, 6.5, 7.0, 7.5, 8.0],
        },
        conditional_subparameters={
            "Buffer": {
                "citrate": {"freq": 0.5, "Conductivity": ([5, 10, 15], [0.2, 0.5, 0.3])},
                "phosphate": {"freq": 0.5, "Conductivity": ([8, 12, 20], [0.3, 0.4, 0.3])},
            },
        },
        X_space=X_space,
        seed=42,
        n_category_trials=75,
        corr_threshold=0.02,
    )
    campaign = Campaign(X_space, target, designer=designer)

    state = campaign.save_state()
    json.dumps(state)  # whole campaign payload must be JSON-safe

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        campaign2 = Campaign.load_state(state)

    assert isinstance(campaign2.designer, AdvExpDesigner)
    assert campaign2.designer.seed == 42
    assert campaign2.designer.n_category_trials == 75
    assert campaign2.designer.corr_threshold == 0.02
    assert "Temperature" in campaign2.designer.continuous_params
    assert "Buffer" in campaign2.designer.conditional_subparameters


@pytest.mark.fast
def test_campaign_reload_preserves_biased_sampling(rng_mode):
    """Biased sampling statistics must survive campaign save/load.

    After reload, initialize() should produce the biased distribution,
    not a uniform one.
    """
    X_space = ParamSpace(
        [
            Param_Continuous("Temperature", 60, 90),
            Param_Continuous("pH", 6.0, 8.0),
        ]
    )
    target = Target(name="Yield", f_transform="Standard", aim="max")

    designer = AdvExpDesigner(
        continuous_params={
            "Temperature": {"levels": [60, 70, 80, 90], "biases": [0.05, 0.10, 0.35, 0.50]},
            "pH": [6.0, 6.5, 7.0, 7.5, 8.0],
        },
        X_space=X_space,
        seed=42,
    )
    campaign = Campaign(X_space, target, designer=designer)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        campaign2 = Campaign.load_state(campaign.save_state())

    X0 = campaign2.initialize(m_initial=200)
    counts = X0["Temperature"].value_counts()

    # Strong bias toward 90 vs 60: must be preserved post-reload
    assert counts.get(90, 0) > counts.get(60, 0), f"Bias not preserved across campaign save/load; counts:\n{counts}"


@pytest.mark.fast
def test_campaign_reload_backward_compat_no_designer_key(rng_mode):
    """Legacy state dicts without 'designer' key must still load.

    Should produce a default ExpDesigner.
    """
    X_space = ParamSpace(
        [
            Param_Continuous("Temperature", 60, 90),
            Param_Continuous("pH", 6.0, 8.0),
        ]
    )
    target = Target(name="Yield", f_transform="Standard", aim="max")
    designer = ExpDesigner(X_space, seed=1)
    campaign = Campaign(X_space, target, designer=designer)

    state = campaign.save_state()
    # Simulate old save format by removing the designer key
    state.pop("designer")

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        campaign2 = Campaign.load_state(state)

    # Falls back to default ExpDesigner
    assert isinstance(campaign2.designer, ExpDesigner)
    assert not isinstance(campaign2.designer, AdvExpDesigner)


@pytest.mark.fast
def test_campaign_reload_unknown_designer_class_warns(rng_mode):
    """Unknown designer class should warn and fall back to default ExpDesigner."""
    X_space = ParamSpace(
        [
            Param_Continuous("Temperature", 60, 90),
            Param_Continuous("pH", 6.0, 8.0),
        ]
    )
    target = Target(name="Yield", f_transform="Standard", aim="max")
    designer = ExpDesigner(X_space, seed=1)
    campaign = Campaign(X_space, target, designer=designer)

    state = campaign.save_state()
    state["designer"]["name"] = "NonExistentDesigner"

    with pytest.warns(UserWarning, match="Unknown designer class"):
        campaign2 = Campaign.load_state(state)

    assert isinstance(campaign2.designer, ExpDesigner)


@pytest.mark.fast
def test_campaign_copy_preserves_adv_designer(rng_mode):
    """Campaign.copy() uses save_state/load_state, so advanced designer must survive."""
    X_space = ParamSpace(
        [
            Param_Continuous("Temperature", 60, 90),
            Param_Continuous("pH", 6.0, 8.0),
        ]
    )
    target = Target(name="Yield", f_transform="Standard", aim="max")

    designer = AdvExpDesigner(
        continuous_params={
            "Temperature": {"levels": [60, 70, 80, 90], "biases": [0.05, 0.10, 0.35, 0.50]},
        },
        X_space=X_space,
        seed=3,
    )
    campaign = Campaign(X_space, target, designer=designer)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        campaign_copy = campaign.copy()

    assert isinstance(campaign_copy.designer, AdvExpDesigner)
    assert campaign_copy.designer.seed == 3


if __name__ == "__main__":
    pytest.main([__file__, "-m", "not slow"])
