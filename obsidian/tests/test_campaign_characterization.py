"""Tests for campaign characterization evaluation"""

import numpy as np
import pandas as pd
import pytest
import torch

from obsidian.campaign import Campaign, CharacterizationEvaluator
from obsidian.optimizer import BayesianOptimizer
from obsidian.parameters import Param_Categorical, Param_Continuous, ParamSpace, Target

# Common test data
X = pd.DataFrame({"x1": [0.1, 0.3, 0.5, 0.7, 0.9], "x2": [0.2, 0.4, 0.6, 0.8, 0.1]})
X_NEW = pd.DataFrame({"x1": [0.15, 0.85], "x2": [0.25, 0.75]})

Y_SINGLE = pd.DataFrame({"y": [3.0, 4.5, 6.0, 7.5, 4.0]})
Y_MULTI = pd.DataFrame({"y1": [3.0, 4.5, 6.0, 7.5, 4.0], "y2": [3.0, 2.5, 1.5, 1.0, 2.8]})

# Specific data for correctness tests (need predictable x-y relationships)
X_LINEAR = pd.DataFrame({"x": [1.0, 3.0, 5.0, 7.0, 9.0]})
Y_LINEAR = pd.DataFrame({"y": [2.0, 3.0, 5.0, 7.0, 8.0]})
Y_LINEAR_MULTI = pd.DataFrame({"y1": [2.0, 4.0, 5.0, 6.0, 8.0], "y2": [4.0, 3.5, 3.0, 2.5, 2.0]})


@pytest.fixture
def simple_campaign():
    """Create a simple fitted campaign with threshold for testing."""
    X_space = ParamSpace([Param_Continuous("x1", 0, 1), Param_Continuous("x2", 0, 1)])
    target = Target("y", aim="max", threshold=5.0)
    campaign = Campaign(X_space, target, task="characterization", seed=42)

    campaign.add_data(pd.concat([X, Y_SINGLE], axis=1))
    campaign.fit()

    return campaign


@pytest.fixture
def multi_target_campaign():
    """Create a campaign with multiple targets for testing joint classification."""
    X_space = ParamSpace([Param_Continuous("x1", 0, 1), Param_Continuous("x2", 0, 1)])
    targets = [Target("y1", aim="max", threshold=5.0), Target("y2", aim="min", threshold=2.0)]
    campaign = Campaign(X_space, targets, task="characterization", seed=42)

    campaign.add_data(pd.concat([X, Y_MULTI], axis=1))
    campaign.fit()

    return campaign


class TestCharacterizationEvaluator:
    """Test CharacterizationEvaluator class."""

    def test_classification_correctness_maximize(self):
        """Test that points are correctly classified relative to threshold for maximize target."""
        X_space = ParamSpace([Param_Continuous("x", 0, 10)])
        target = Target("y", aim="max", threshold=5.0)
        campaign = Campaign(X_space, target, task="characterization", seed=42)

        campaign.add_data(pd.concat([X_LINEAR, Y_LINEAR], axis=1))
        campaign.fit()

        evaluator = CharacterizationEvaluator(campaign)
        results = evaluator.classify_points(pd.DataFrame({"x": [1.0, 9.0]}), PI_range=0.0)

        pred_mask = results[target.name]["pred_mask"]
        assert pred_mask[0] == False  # x=1.0 → y≈2.0 < 5.0 → fail
        assert pred_mask[1] == True  # x=9.0 → y≈8.0 > 5.0 → pass

    def test_classification_correctness_minimize(self):
        """Test that points are correctly classified relative to threshold for minimize target."""
        X_space = ParamSpace([Param_Continuous("x", 0, 10)])
        target = Target("y", aim="min", threshold=5.0)
        campaign = Campaign(X_space, target, task="characterization", seed=42)

        campaign.add_data(pd.concat([X_LINEAR, Y_LINEAR], axis=1))
        campaign.fit()

        evaluator = CharacterizationEvaluator(campaign)
        results = evaluator.classify_points(pd.DataFrame({"x": [1.0, 9.0]}), PI_range=0.0)

        pred_mask = results[target.name]["pred_mask"]
        assert pred_mask[0] == True  # x=1.0 → y≈2.0 < 5.0 → pass
        assert pred_mask[1] == False  # x=9.0 → y≈8.0 > 5.0 → fail

    def test_joint_is_intersection(self):
        """Test that joint classification is the logical AND of individual targets."""
        X_space = ParamSpace([Param_Continuous("x", 0, 10)])
        targets = [Target("y1", aim="max", threshold=5.0), Target("y2", aim="min", threshold=3.0)]
        campaign = Campaign(X_space, targets, task="characterization", seed=42)

        campaign.add_data(pd.concat([X_LINEAR, Y_LINEAR_MULTI], axis=1))
        campaign.fit()

        evaluator = CharacterizationEvaluator(campaign)
        results = evaluator.classify_points(pd.DataFrame({"x": [1.0, 5.0, 9.0]}), PI_range=0.0)

        # Verify joint = y1 and y2
        expected_joint = results["y1"]["pred_mask"] & results["y2"]["pred_mask"]
        assert np.array_equal(results["Joint"]["pred_mask"], expected_joint)

    def test_confusion_matrix_arithmetic(self):
        """Test confusion matrix with known ground truth."""
        X_space = ParamSpace([Param_Continuous("x", 0, 10)])
        target = Target("y", aim="max", threshold=5.0)
        campaign = Campaign(X_space, target, task="characterization", seed=42)

        X_train = pd.DataFrame({"x": [1.0, 5.0, 9.0]})
        y_train = pd.DataFrame({"y": [2.0, 5.0, 8.0]})
        campaign.add_data(pd.concat([X_train, y_train], axis=1))
        campaign.fit()

        evaluator = CharacterizationEvaluator(campaign)

        # Test on training data where we know ground truth
        X_test = X_train
        y_true = np.array([2.0, 5.0, 8.0])

        results = evaluator.evaluate_with_ground_truth(X_test, y_true, PI_range=0.0)
        cm = results[target.name]["confusion_matrix"]

        # Ground truth: [Fail, Pass, Pass]
        # Predictions should match closely on training data
        # At minimum: TP + FP + FN + TN should equal total points
        assert cm["TP"] + cm["FP"] + cm["FN"] + cm["TN"] == 3

        # Jaccard = TP / (TP + FP + FN)
        expected_jaccard = (
            cm["TP"] / (cm["TP"] + cm["FP"] + cm["FN"]) if (cm["TP"] + cm["FP"] + cm["FN"]) > 0 else np.nan
        )
        if not np.isnan(expected_jaccard):
            assert abs(results[target.name]["jaccard"] - expected_jaccard) < 1e-10

    def test_initialization(self, simple_campaign):
        """Test evaluator initialization."""
        evaluator = CharacterizationEvaluator(simple_campaign)
        assert evaluator.campaign == simple_campaign
        assert evaluator.optimizer == simple_campaign.optimizer
        assert len(evaluator.targets) == 1

    def test_initialization_no_threshold(self):
        """Test that initialization fails without threshold."""
        X_space = ParamSpace([Param_Continuous("x1", 0, 1)])
        target = Target("y", aim="max")  # No threshold
        campaign = Campaign(X_space, target, task="optimization")

        X_train = X.iloc[[0, 2, 4]][["x1"]]
        y_train = pd.DataFrame({"y": [1.0, 2.0, 3.0]})
        campaign.add_data(pd.concat([X_train, y_train], axis=1))
        campaign.fit()

        with pytest.raises(ValueError, match="threshold"):
            CharacterizationEvaluator(campaign)

    def test_initialization_unfitted(self):
        """Test that initialization fails with unfitted campaign."""
        X_space = ParamSpace([Param_Continuous("x1", 0, 1)])
        target = Target("y", aim="max", threshold=5.0)
        campaign = Campaign(X_space, target, task="characterization")

        with pytest.raises(ValueError, match="fitted"):
            CharacterizationEvaluator(campaign)

    def test_fractions_sum(self, simple_campaign):
        """Test that pos_frac + neg_frac = classified_frac."""
        evaluator = CharacterizationEvaluator(simple_campaign)

        X_test = pd.DataFrame({"x1": [0.2, 0.5, 0.8], "x2": [0.3, 0.6, 0.9]})

        results = evaluator.classify_points(X_test, PI_range=0.7)
        target_result = results[evaluator.targets[0].name]

        # Core arithmetic property
        assert abs(target_result["classified_frac"] - (target_result["pos_frac"] + target_result["neg_frac"])) < 1e-10
        assert 0 <= target_result["classified_frac"] <= 1

    def test_sobol_sampling(self, simple_campaign):
        """Test that integer input triggers Sobol sampling of correct size."""
        evaluator = CharacterizationEvaluator(simple_campaign, seed=42)

        results = evaluator.classify_points(100, PI_range=0.7)

        # Verify correct number of points sampled
        assert results[evaluator.targets[0].name]["pred_mask"].size == 100

    def test_classify_points_different_PI_ranges(self, simple_campaign):
        """Test that different PI ranges give different results."""
        evaluator = CharacterizationEvaluator(simple_campaign)

        X_test = simple_campaign.X

        results_70 = evaluator.classify_points(X_test, PI_range=0.7)
        results_95 = evaluator.classify_points(X_test, PI_range=0.95)

        # 95% CI should classify less space (more conservative)
        target = evaluator.targets[0]
        assert results_95[target.name]["classified_frac"] <= results_70[target.name]["classified_frac"]

    def test_joint_pass_is_subset(self, multi_target_campaign):
        """Test that joint pass is a subset of each individual pass."""
        evaluator = CharacterizationEvaluator(multi_target_campaign)

        X_test = multi_target_campaign.X
        results = evaluator.classify_points(X_test, PI_range=0.7)

        # Joint pass fraction must be <= each individual pass fraction
        joint_pass = results["Joint"]["pos_frac"]
        for target in evaluator.targets:
            assert joint_pass <= results[target.name]["pos_frac"]

    def test_signed_threshold_maximize(self, simple_campaign):
        """Test that maximize target uses correct sign."""
        evaluator = CharacterizationEvaluator(simple_campaign)
        target = simple_campaign.target[0]

        mean_unsigned = np.array([4.0, 5.0, 6.0])
        std = np.array([0.5, 0.5, 0.5])

        mean_signed, signed_threshold, std_out = evaluator._prepare_signed_data(target, mean_unsigned, std)

        # For maximize: sign = 1
        assert np.array_equal(mean_signed, mean_unsigned)
        assert signed_threshold == target.threshold
        assert np.array_equal(std_out, std)

    def test_signed_threshold_minimize(self):
        """Test that minimize target uses correct sign."""
        X_space = ParamSpace([Param_Continuous("x1", 0, 1)])
        target = Target("y", aim="min", threshold=2.0)
        campaign = Campaign(X_space, target, task="characterization", seed=42)

        X_train = X.iloc[[0, 2, 4]][["x1"]]
        y_train = pd.DataFrame({"y": [3.0, 2.0, 1.0]})
        campaign.add_data(pd.concat([X_train, y_train], axis=1))
        campaign.fit()

        evaluator = CharacterizationEvaluator(campaign)

        mean_unsigned = np.array([3.0, 2.0, 1.0])
        std = np.array([0.5, 0.5, 0.5])

        mean_signed, signed_threshold, std_out = evaluator._prepare_signed_data(target, mean_unsigned, std)

        # For minimize: sign = -1
        assert np.array_equal(mean_signed, -mean_unsigned)
        assert signed_threshold == -target.threshold
        assert np.array_equal(std_out, std)

    def test_confusion_matrix(self):
        """Test confusion matrix computation."""
        pred_mask = np.array([True, True, False, False])
        true_mask = np.array([True, False, True, False])

        result = CharacterizationEvaluator._get_confusion_matrix(pred_mask, true_mask)

        assert result["TP"] == 1
        assert result["FP"] == 1
        assert result["FN"] == 1
        assert result["TN"] == 1
        assert result["jaccard"] == 1 / 3  # TP / (TP + FP + FN)

    def test_plan_sample_size(self, simple_campaign):
        """Test sample size planning."""
        evaluator = CharacterizationEvaluator(simple_campaign, seed=42)

        N = evaluator.plan_sample_size(pilot_ratio=0.1, epsilon=0.01, z=1.96, max_samples=10000)

        # Should return a reasonable sample size
        assert isinstance(N, int)
        assert 100 <= N <= 10000


class TestCampaignIntegration:
    """Test integration with Campaign class."""

    def test_automatic_characterization_analysis(self, simple_campaign):
        """Test that characterization columns are added automatically."""
        # Columns should be added after fit
        cis = ["70", "95"]
        fields = ["Pass", "Fail", "Classified"]
        for ci in cis:
            for field in fields:
                assert f"Characterization y {field} % ({ci}% CI)" in simple_campaign.data.columns

    def test_no_characterization_without_threshold(self):
        """Test that characterization columns are not added without threshold."""
        X_space = ParamSpace([Param_Continuous("x1", 0, 1)])
        target = Target("y", aim="max")
        campaign = Campaign(X_space, target, task="optimization", seed=42)

        X_train = X.iloc[[0, 2, 4]][["x1"]]
        y_train = pd.DataFrame({"y": [1.0, 2.0, 3.0]})
        campaign.add_data(pd.concat([X_train, y_train], axis=1))
        campaign.fit()

        # Should not have characterization columns
        char_cols = [col for col in campaign.data.columns if "Characterization" in col]
        assert len(char_cols) == 0

    def test_characterization_columns_are_constant(self, simple_campaign):
        """Test that characterization percentage columns have constant values."""
        # All rows should have the same value (percentage of space)
        pass_70 = simple_campaign.data["Characterization y Pass % (70% CI)"]
        assert pass_70.nunique() == 1

        fail_70 = simple_campaign.data["Characterization y Fail % (70% CI)"]
        assert fail_70.nunique() == 1

    def test_joint_columns_multi_target(self, multi_target_campaign):
        """Test that joint columns are added for multi-target."""
        assert "Characterization Joint Pass % (70% CI)" in multi_target_campaign.data.columns

    def test_no_joint_columns_single_target(self, simple_campaign):
        """Test that joint columns are not added for single target."""
        joint_cols = [col for col in simple_campaign.data.columns if "Joint" in col]
        assert len(joint_cols) == 0

    def test_no_response_max_for_characterization(self, simple_campaign):
        """Test that response max is not added for characterization campaigns."""
        max_cols = [col for col in simple_campaign.data.columns if "(max) (iter)" in col]
        assert len(max_cols) == 0

    def test_evaluate_characterization_no_threshold(self):
        """Test that evaluate_characterization fails without threshold."""
        X_space = ParamSpace([Param_Continuous("x1", 0, 1)])
        target = Target("y", aim="max")
        campaign = Campaign(X_space, target, task="optimization", seed=42)

        X_train = X.iloc[[0, 2, 4]][["x1"]]
        y_train = pd.DataFrame({"y": [1.0, 2.0, 3.0]})
        campaign.add_data(pd.concat([X_train, y_train], axis=1))
        campaign.fit()

        with pytest.raises(ValueError, match="threshold"):
            campaign.evaluate_characterization()

    def test_characterization_metrics_per_iteration(self, simple_campaign):
        """Each iteration's rows carry the metric from when that batch was fitted."""
        initial_pass = simple_campaign.data["Characterization y Pass % (70% CI)"].iloc[0]
        initial_iter = simple_campaign.data["Iteration"].iloc[0]

        # Add a second batch and refit
        X_new = X_NEW
        y_new = pd.DataFrame({"y": [3.5, 9.5]})
        simple_campaign.add_data(pd.concat([X_new, y_new], axis=1))
        simple_campaign.fit()

        # Original rows must still have the OLD metric value
        old_rows = simple_campaign.data[simple_campaign.data["Iteration"] == initial_iter]
        assert (old_rows["Characterization y Pass % (70% CI)"] == initial_pass).all()

        # New rows have a (potentially different) metric value and are not NaN
        new_iter = simple_campaign.data["Iteration"].max()
        new_rows = simple_campaign.data[simple_campaign.data["Iteration"] == new_iter]
        assert new_rows["Characterization y Pass % (70% CI)"].notna().all()

    def test_characterization_updates_on_new_data(self, simple_campaign):
        """Test that characterization columns update when new data is added."""
        # Get initial values
        initial_pass = simple_campaign.data["Characterization y Pass % (70% CI)"].iloc[0]

        # Add more data
        X_new = X_NEW
        y_new = pd.DataFrame({"y": [3.5, 8.0]})
        new_data = pd.concat([X_new, y_new], axis=1)

        simple_campaign.add_data(new_data)
        simple_campaign.fit()

        # Values should potentially change (though might be the same by chance)
        # Just check that columns still exist and are valid
        assert "Characterization y Pass % (70% CI)" in simple_campaign.data.columns
        final_pass = simple_campaign.data["Characterization y Pass % (70% CI)"].iloc[0]
        assert 0 <= final_pass <= 100

    def test_task_type_warns_if_not_specified(self):
        """Test that a warning is raised when task is not specified."""
        X_space = ParamSpace([Param_Continuous("x1", 0, 1)])
        target = Target("y", aim="max", threshold=5.0)

        with pytest.warns(UserWarning, match="task not specified"):
            campaign = Campaign(X_space, target)

        # Defaults to OPTIMIZATION when not specified
        from obsidian.utils import TaskType

        assert campaign.task == TaskType.OPTIMIZATION

    def test_task_type_no_warning_if_specified(self):
        """Test that no warning is raised when task is explicitly set."""
        from obsidian.utils import TaskType
        import warnings

        X_space = ParamSpace([Param_Continuous("x1", 0, 1)])
        target = Target("y", aim="max", threshold=5.0)

        with warnings.catch_warnings():
            warnings.simplefilter("error", UserWarning)
            campaign = Campaign(X_space, target, task="characterization")

        assert campaign.task == TaskType.CHARACTERIZATION

    def test_explicit_task_type(self):
        """Test that explicit task type can be set."""
        from obsidian.utils import TaskType

        X_space = ParamSpace([Param_Continuous("x1", 0, 1)])
        target = Target("y", aim="max", threshold=5.0)

        campaign_char = Campaign(X_space, target, task="characterization")
        assert campaign_char.task == TaskType.CHARACTERIZATION
        assert campaign_char._is_characterization

        campaign_opt = Campaign(X_space, target, task="optimization")
        assert campaign_opt.task == TaskType.OPTIMIZATION
        assert not campaign_opt._is_characterization

    def test_task_type_save_load(self):
        """Test that task type is saved and loaded."""
        X_space = ParamSpace([Param_Continuous("x1", 0, 1)])
        target = Target("y", aim="max", threshold=5.0)
        campaign = Campaign(X_space, target, task="characterization", seed=42)

        X_train = X.iloc[[0, 2, 4]][["x1"]]
        y_train = pd.DataFrame({"y": [3.0, 6.0, 4.0]})
        campaign.add_data(pd.concat([X_train, y_train], axis=1))
        campaign.fit()

        # Save state
        state = campaign.save_state()
        assert "task" in state
        assert state["task"] == "characterization"

        # Load state
        campaign_loaded = Campaign.load_state(state)
        assert campaign_loaded.task.value == "characterization"
        assert campaign_loaded._is_characterization


class TestTrackingOnlyTargets:
    """Test that evaluator correctly ignores tracking-only targets."""

    @pytest.fixture
    def campaign_with_tracking(self):
        """Campaign with one active threshold target and one tracking-only target."""
        X_space = ParamSpace([Param_Continuous("x1", 0, 1), Param_Continuous("x2", 0, 1)])
        targets = [
            Target("y1", aim="max", threshold=5.0),
            Target("y2", aim="min", tracking_only=True),  # no threshold, tracking only
        ]
        campaign = Campaign(X_space, targets, task="characterization", seed=42)

        X_train = X
        y_train = pd.DataFrame({"y1": [3.0, 4.5, 6.0, 7.5, 4.0], "y2": [1.0, 1.5, 2.0, 2.5, 1.2]})
        campaign.add_data(pd.concat([X_train, y_train], axis=1))
        campaign.fit()
        return campaign

    def test_evaluator_ignores_tracking_only(self, campaign_with_tracking):
        """tracking_only targets are excluded from characterization."""
        evaluator = CharacterizationEvaluator(campaign_with_tracking)
        assert len(evaluator.targets) == 1
        assert evaluator.targets[0].name == "y1"

    def test_classify_points_ignores_tracking_only(self, campaign_with_tracking):
        """classify_points returns results only for non-tracking-only targets."""
        evaluator = CharacterizationEvaluator(campaign_with_tracking)
        results = evaluator.classify_points(200, PI_range=0.7)

        assert len([k for k in results.keys() if k not in ("X_samples", "X_feasible")]) == 1
        assert "Joint" not in results

    def test_characterization_columns_exclude_tracking_only(self, campaign_with_tracking):
        """campaign.data columns are only added for non-tracking-only targets."""
        assert "Characterization y1 Pass % (70% CI)" in campaign_with_tracking.data.columns
        assert "Characterization y2 Pass % (70% CI)" not in campaign_with_tracking.data.columns

    def test_threshold_added_on_fly_is_picked_up(self, campaign_with_tracking):
        """Adding a threshold to a non-tracking-only target is immediately reflected."""
        evaluator = CharacterizationEvaluator(campaign_with_tracking)
        assert len(evaluator.targets) == 1

        # Give y2 a threshold AND make it non-tracking-only
        campaign_with_tracking.target[1].threshold = 2.0
        campaign_with_tracking.target[1].tracking_only = False
        assert len(evaluator.targets) == 2
        assert evaluator.targets[1].name == "y2"

    def test_raises_if_no_active_target_has_threshold(self):
        """Evaluator raises ValueError when no active target has a threshold."""
        X_space = ParamSpace([Param_Continuous("x1", 0, 1)])
        target = Target("y1", aim="max")  # no threshold
        campaign = Campaign(X_space, target, task="optimization", seed=42)
        X_train = X.iloc[[0, 2, 4]][["x1"]]
        y_train = pd.DataFrame({"y1": [3.0, 6.0, 4.0]})
        campaign.add_data(pd.concat([X_train, y_train], axis=1))
        campaign.fit()

        with pytest.raises(ValueError, match="threshold"):
            CharacterizationEvaluator(campaign)


# Analytical feasible regions for hypercube mock testing.
# Each region takes a DataFrame with 'x1', 'x2' columns and returns a boolean mask.

# Box: axis-aligned rectangle, largest inscribed square has half-width 0.3 at center (0.5, 0.6)
REGION_BOX = lambda X: (X["x1"] >= 0.2) & (X["x1"] <= 0.8) & (X["x2"] >= 0.3) & (X["x2"] <= 0.9)

# Circle: disk of radius 0.4 centered at (0.5, 0.5). Largest inscribed square has half-width 0.4/sqrt(2).
REGION_CIRCLE = lambda X: (X["x1"] - 0.5) ** 2 + (X["x2"] - 0.5) ** 2 <= 0.4**2

# L-shape (concave): union of two rectangles forming an L.
#   Rect A: [0.1, 0.9] x [0.1, 0.4]
#   Rect B: [0.1, 0.4] x [0.1, 0.9]
# Largest inscribed square fits in Rect A (or B) with half-width 0.15.
REGION_L_SHAPE = lambda X: (
    ((X["x1"] >= 0.1) & (X["x1"] <= 0.9) & (X["x2"] >= 0.1) & (X["x2"] <= 0.4))
    | ((X["x1"] >= 0.1) & (X["x1"] <= 0.4) & (X["x2"] >= 0.1) & (X["x2"] <= 0.9))
)


def _make_mock_classify(region_fn):
    """Build a mock classify_points replacement from an analytical feasible region fn."""

    def mock_classify(X, PI_range=0.7, return_samples=False):
        pred_mask = region_fn(X)
        return {
            "y": {
                "pred_mask": pred_mask,
                "pos_frac": pred_mask.mean(),
                "neg_frac": (~pred_mask).mean(),
                "classified_frac": 1.0,
            },
            "X_samples": X,
            "X_feasible": X[pred_mask],
        }

    return mock_classify


class TestHypercubeComputation:
    """Tests for compute_largest_hypercube method.

    Tests use analytical feasible regions (box, circle, L-shape) instead of GP fits
    to verify numerical correctness of the hypercube computation.
    """

    def test_box_region_numerical_correctness(self, simple_campaign):
        """Box region [0.2,0.8]x[0.3,0.9]: largest square has half-width 0.3."""
        evaluator = CharacterizationEvaluator(simple_campaign)
        evaluator.classify_points = _make_mock_classify(REGION_BOX)

        result = evaluator.compute_largest_hypercube(X=1000, PI_range=0.7)

        # Expected: half-width 0.3 (x1 limits), volume 0.36
        assert abs(result["volume"] - 0.36) / 0.36 < 0.02
        assert abs(result["center"]["x1"] - 0.5) < 0.05
        assert abs(result["center"]["x2"] - 0.6) < 0.05

        hw_x1 = (result["bounds"]["upper"]["x1"] - result["bounds"]["lower"]["x1"]) / 2
        hw_x2 = (result["bounds"]["upper"]["x2"] - result["bounds"]["lower"]["x2"]) / 2
        assert abs(hw_x1 - hw_x2) < 0.01  # uniform weights
        assert abs(hw_x1 - 0.3) < 0.015

    def test_circle_region_numerical_correctness(self, simple_campaign):
        """Disk of radius 0.4: largest inscribed square has half-width 0.4/sqrt(2)."""
        evaluator = CharacterizationEvaluator(simple_campaign)
        evaluator.classify_points = _make_mock_classify(REGION_CIRCLE)

        result = evaluator.compute_largest_hypercube(X=2000, center={"x1": 0.5, "x2": 0.5}, PI_range=0.7)

        expected_hw = 0.4 / np.sqrt(2)
        expected_volume = (2 * expected_hw) ** 2  # 0.32

        # Conservative check may slightly over-estimate due to sample density in gaps
        assert abs(result["volume"] - expected_volume) / expected_volume < 0.05
        hw = (result["bounds"]["upper"]["x1"] - result["bounds"]["lower"]["x1"]) / 2
        assert abs(hw - expected_hw) / expected_hw < 0.05

    def test_l_shape_region_numerical_correctness(self, simple_campaign):
        """L-shaped (concave) region: hypercube centered in horizontal arm.

        Arm is [0.1, 0.9] x [0.1, 0.4] — thickness 0.3 → max square half-width 0.15.
        We pin the center inside that arm (0.5, 0.25) so the search is well-defined;
        with auto-center the centroid lands in the concave pocket where the
        inscribed square is much smaller.
        """
        evaluator = CharacterizationEvaluator(simple_campaign)
        evaluator.classify_points = _make_mock_classify(REGION_L_SHAPE)

        result = evaluator.compute_largest_hypercube(X=2000, center={"x1": 0.5, "x2": 0.25}, PI_range=0.7)

        expected_hw = 0.15  # arm thickness / 2
        expected_volume = (2 * expected_hw) ** 2  # 0.09

        hw_x1 = (result["bounds"]["upper"]["x1"] - result["bounds"]["lower"]["x1"]) / 2
        hw_x2 = (result["bounds"]["upper"]["x2"] - result["bounds"]["lower"]["x2"]) / 2

        # Conservative: should not exceed the arm's inscribable square
        assert result["volume"] <= expected_volume * 1.05
        # Uniform weights → equal half-widths
        assert abs(hw_x1 - hw_x2) < 0.015
        assert abs(hw_x1 - expected_hw) / expected_hw < 0.1

    def test_fixed_center_is_respected(self, simple_campaign):
        """User-specified center is used verbatim (not re-centered)."""
        evaluator = CharacterizationEvaluator(simple_campaign)
        evaluator.classify_points = _make_mock_classify(REGION_BOX)

        result = evaluator.compute_largest_hypercube(X=1000, center={"x1": 0.5, "x2": 0.6}, PI_range=0.7)

        assert result["center"]["x1"] == 0.5
        assert result["center"]["x2"] == 0.6

    def test_auto_center_uses_centroid(self, simple_campaign):
        """When center=None, centroid of feasible samples is used."""
        evaluator = CharacterizationEvaluator(simple_campaign)
        evaluator.classify_points = _make_mock_classify(REGION_BOX)

        result = evaluator.compute_largest_hypercube(X=2000, PI_range=0.7)

        # Box is [0.2,0.8]x[0.3,0.9] → centroid at (0.5, 0.6)
        assert abs(result["center"]["x1"] - 0.5) < 0.02
        assert abs(result["center"]["x2"] - 0.6) < 0.02

    def test_dim_weights_give_correct_aspect_ratio(self, simple_campaign):
        """dim_weights={'x1': 2, 'x2': 1} → half-width ratio 2:1."""
        evaluator = CharacterizationEvaluator(simple_campaign)
        evaluator.classify_points = _make_mock_classify(REGION_BOX)

        result = evaluator.compute_largest_hypercube(
            X=2000, center={"x1": 0.5, "x2": 0.6}, dim_weights={"x1": 2, "x2": 1}, PI_range=0.7
        )

        hw_x1 = (result["bounds"]["upper"]["x1"] - result["bounds"]["lower"]["x1"]) / 2
        hw_x2 = (result["bounds"]["upper"]["x2"] - result["bounds"]["lower"]["x2"]) / 2

        # Ratio should be 2:1
        assert abs(hw_x1 / hw_x2 - 2.0) < 0.1
        # Box has x1 width 0.6 (centered at 0.5) and x2 width 0.6 (centered at 0.6)
        # With 2:1 ratio and uniform box, x2 is limiting: hw_x2 <= 0.3, hw_x1 <= 0.3*2 = 0.6
        # x1 is bounded by box at ±0.3 from 0.5, so hw_x1 ≈ 0.3 → hw_x2 ≈ 0.15
        assert abs(hw_x1 - 0.3) < 0.02
        assert abs(hw_x2 - 0.15) < 0.02

    def test_result_keys_when_nothing_pinned(self, simple_campaign):
        """Continuous-only problem with nothing fixed: both keys present and empty."""
        evaluator = CharacterizationEvaluator(simple_campaign)
        evaluator.classify_points = _make_mock_classify(REGION_BOX)

        result = evaluator.compute_largest_hypercube(X=500, PI_range=0.7)

        assert result["fixed_dims"] == {}
        assert result["categorical_values"] == {}

    def test_fixed_dims_zero_width(self, simple_campaign):
        """Continuous fixed dims have zero half-width and are reported in 'fixed_dims'."""
        evaluator = CharacterizationEvaluator(simple_campaign)
        evaluator.classify_points = _make_mock_classify(REGION_BOX)

        result = evaluator.compute_largest_hypercube(
            X=1000, center={"x1": 0.5, "x2": 0.6}, fixed_dims={"x2": 0.6}, PI_range=0.7
        )

        # Continuous pinning lands in 'fixed_dims'; no categorical params here.
        assert result["fixed_dims"] == {"x2": 0.6}
        assert result["categorical_values"] == {}

        hw = (result["bounds"]["upper"] - result["bounds"]["lower"]) / 2
        assert hw["x2"] == 0.0
        assert result["center"]["x2"] == 0.6
        # Along x1, feasible range is [0.2, 0.8] → half-width 0.3
        assert abs(hw["x1"] - 0.3) < 0.02

    def test_bounds_respect_parameter_space(self, simple_campaign):
        """Hypercube bounds are clipped to parameter-space limits, not beyond."""
        evaluator = CharacterizationEvaluator(simple_campaign)
        # Box is [0.2,0.8]x[0.3,0.9] but we force center near corner → hypercube should clip
        evaluator.classify_points = _make_mock_classify(REGION_BOX)

        result = evaluator.compute_largest_hypercube(X=1000, center={"x1": 0.8, "x2": 0.9}, PI_range=0.7)

        # Bounds should stay in [0, 1]^2
        assert result["bounds"]["lower"]["x1"] >= 0
        assert result["bounds"]["lower"]["x2"] >= 0
        assert result["bounds"]["upper"]["x1"] <= 1
        assert result["bounds"]["upper"]["x2"] <= 1
        # At (0.8, 0.9) with parameter space [0,1]^2, max half-width is min(0.2, 0.1) = 0.1
        hw = (result["bounds"]["upper"] - result["bounds"]["lower"]) / 2
        assert hw["x1"] <= 0.1 + 1e-6
        assert hw["x2"] <= 0.1 + 1e-6

    def test_convergence(self, simple_campaign):
        """Binary search converges well before max_iter with analytical region."""
        evaluator = CharacterizationEvaluator(simple_campaign)
        evaluator.classify_points = _make_mock_classify(REGION_BOX)

        result = evaluator.compute_largest_hypercube(X=500, center={"x1": 0.5, "x2": 0.6}, tolerance=1e-4, max_iter=50)

        assert 0 < result["convergence_iters"] < 50

    def test_volume_equals_product_of_widths(self, simple_campaign):
        """Volume reported matches ∏ (2 * half_width) across free dimensions."""
        evaluator = CharacterizationEvaluator(simple_campaign)
        evaluator.classify_points = _make_mock_classify(REGION_BOX)

        result = evaluator.compute_largest_hypercube(X=500, center={"x1": 0.5, "x2": 0.6}, PI_range=0.7)

        hw = (result["bounds"]["upper"] - result["bounds"]["lower"]) / 2
        expected = (2 * hw["x1"]) * (2 * hw["x2"])
        assert abs(result["volume"] - expected) < 1e-6

    def test_empty_feasible_region_raises(self, simple_campaign):
        """Too few feasible points raises ValueError."""
        evaluator = CharacterizationEvaluator(simple_campaign)
        # Region that accepts nothing → no feasible points
        evaluator.classify_points = _make_mock_classify(lambda X: pd.Series(False, index=X.index))

        with pytest.raises(ValueError, match="Too few feasible points"):
            evaluator.compute_largest_hypercube(X=500, center={"x1": 0.5, "x2": 0.5})

    def test_all_fixed_params_raises(self, simple_campaign):
        """Fixing all parameters raises ValueError."""
        evaluator = CharacterizationEvaluator(simple_campaign)
        evaluator.classify_points = _make_mock_classify(REGION_BOX)

        with pytest.raises(ValueError, match="All parameters are fixed"):
            evaluator.compute_largest_hypercube(X=500, center={"x1": 0.5, "x2": 0.6}, fixed_dims={"x1": 0.5, "x2": 0.6})

    def test_missing_center_parameter_raises(self, simple_campaign):
        """Partial center spec raises ValueError."""
        evaluator = CharacterizationEvaluator(simple_campaign)
        evaluator.classify_points = _make_mock_classify(REGION_BOX)

        with pytest.raises(ValueError, match="missing parameters"):
            evaluator.compute_largest_hypercube(X=500, center={"x1": 0.5})

    def test_multi_target_uses_joint_region(self, multi_target_campaign):
        """Multi-target campaigns use joint feasible region (integration)."""
        evaluator = CharacterizationEvaluator(multi_target_campaign)

        result = evaluator.compute_largest_hypercube(X=500, PI_range=0.7)

        assert result["volume"] >= 0
        assert "center" in result

    def test_different_pi_ranges_give_different_volumes(self, simple_campaign):
        """Wider PI → more uncertainty → smaller feasible region → smaller volume (integration)."""
        evaluator = CharacterizationEvaluator(simple_campaign)

        result_70 = evaluator.compute_largest_hypercube(X=500, center={"x1": 0.5, "x2": 0.5}, PI_range=0.7)
        result_95 = evaluator.compute_largest_hypercube(X=500, center={"x1": 0.5, "x2": 0.5}, PI_range=0.95)

        assert result_95["volume"] <= result_70["volume"]

    def test_discrete_parameters_enumeration(self):
        """Test hypercube with discrete parameters enumerates combinations."""
        X_space = ParamSpace([Param_Continuous("x", 0, 10), Param_Categorical("color", ["red", "blue"])])
        target = Target("y", aim="max", threshold=4.0)
        campaign = Campaign(X_space, target, task="characterization", seed=42)

        # Red points are better than blue
        X_train = pd.DataFrame({"x": [1, 3, 5, 7, 9, 1, 3, 5, 7, 9], "color": ["red"] * 5 + ["blue"] * 5})
        y_train = pd.DataFrame({"y": [6, 7, 8, 7, 6, 3, 3.5, 4, 3.5, 3]})  # red is higher
        campaign.add_data(pd.concat([X_train, y_train], axis=1))
        campaign.fit()

        evaluator = CharacterizationEvaluator(campaign)
        result = evaluator.compute_largest_hypercube(X=1000, PI_range=0.7)

        # Winning categorical combo shows up in 'categorical_values', not 'fixed_dims'
        assert result["categorical_values"] == {"color": "red"}
        assert result["fixed_dims"] == {}
        assert result["volume"] > 0
        assert "center" in result

    def test_discrete_all_fixed(self):
        """Test that when all discrete params are fixed, it works like continuous."""
        X_space = ParamSpace([Param_Continuous("x", 0, 10), Param_Categorical("color", ["red", "blue"])])
        target = Target("y", aim="max", threshold=4.0)
        campaign = Campaign(X_space, target, task="characterization", seed=42)

        X_train = pd.DataFrame({"x": [1.0, 3.0, 5.0, 7.0, 9.0], "color": ["red"] * 5})
        y_train = pd.DataFrame({"y": [6.0, 7.0, 8.0, 7.0, 6.0]})
        campaign.add_data(pd.concat([X_train, y_train], axis=1))
        campaign.fit()

        evaluator = CharacterizationEvaluator(campaign)
        result = evaluator.compute_largest_hypercube(X=1000, fixed_dims={"color": "red"}, PI_range=0.7)

        # User-specified categorical pinning shows up in 'categorical_values',
        # and 'fixed_dims' stays empty because no continuous dim was fixed.
        assert result["categorical_values"] == {"color": "red"}
        assert result["fixed_dims"] == {}
        assert result["volume"] > 0

    def test_mixed_continuous_fix_and_categorical_enumeration(self):
        """fixed_dims with a continuous pin splits correctly across the result keys.

        'x1' (continuous) ends up in 'fixed_dims'; the chosen 'color' (categorical,
        enumerated since not user-pinned) ends up in 'categorical_values'. A
        second free continuous dim 'x2' ensures the hypercube is non-degenerate.
        """
        X_space = ParamSpace(
            [
                Param_Continuous("x1", 0, 10),
                Param_Continuous("x2", 0, 10),
                Param_Categorical("color", ["red", "blue"]),
            ]
        )
        target = Target("y", aim="max", threshold=4.0)
        campaign = Campaign(X_space, target, task="characterization", seed=42)

        # Red feasible, blue infeasible. x2 varies so we have a free continuous dim.
        X_train = pd.DataFrame(
            {
                "x1": [1.0, 3.0, 5.0, 7.0, 9.0] * 2,
                "x2": [2.0, 4.0, 6.0, 4.0, 2.0] * 2,
                "color": ["red"] * 5 + ["blue"] * 5,
            }
        )
        y_train = pd.DataFrame({"y": [6.0, 7.0, 8.0, 7.0, 6.0, 2.0, 2.5, 3.0, 2.5, 2.0]})
        campaign.add_data(pd.concat([X_train, y_train], axis=1))
        campaign.fit()

        evaluator = CharacterizationEvaluator(campaign)
        result = evaluator.compute_largest_hypercube(X=1000, fixed_dims={"x1": 5.0}, PI_range=0.7)

        assert result["fixed_dims"] == {"x1": 5.0}
        assert result["categorical_values"] == {"color": "red"}


class TestRandomizedStraddleRNG:
    """Regression tests for randomized-straddle (RANDSTR) beta seeding behavior."""

    def test_beta_data_dependent(self):
        """Randomized-straddle beta must be idempotent within an iteration but vary across iterations.

        Regression test: previously the beta draw was frozen for the entire campaign because the
        acquisition ran inside ``with_tmp_seed(self.seed)`` and re-seeded from the same constant
        every call. Beta is now derived from (effective seed, n_obs), so it is reproducible for a
        given data state yet advances as new observations arrive.
        """
        from botorch.models import SingleTaskGP

        from obsidian.acquisition.characterization import qRandomizedStraddle
        from obsidian.rng import with_tmp_seed

        def _dummy_model():
            train_X = torch.rand(5, 2, dtype=torch.double)
            train_Y = torch.rand(5, 1, dtype=torch.double)
            return SingleTaskGP(train_X, train_Y)

        self_seed = 42  # constant self.seed, mimicking fix_random_state=True across suggests

        def build_beta(n_obs):
            # The optimizer builds the acquisition inside with_tmp_seed(self.seed).
            with with_tmp_seed(self_seed):
                aq = qRandomizedStraddle(model=_dummy_model(), threshold=0.0, n_obs=n_obs)
                return torch.as_tensor(aq.beta).flatten().clone()

        # Idempotent within an iteration: same data (n_obs) -> identical beta.
        beta_a = build_beta(5)
        beta_b = build_beta(5)
        assert torch.allclose(beta_a, beta_b)

        # Varies across iterations: more observations -> different beta.
        beta_next = build_beta(6)
        assert not torch.allclose(beta_a, beta_next)

    def test_reported_aq_value_reproducible(self):
        """The reported aq value must be reproducible across identical suggests (fix_random_state=True).

        Regression test: the acquisition is instantiated twice per suggest -- once inside the seeded
        optimization wrapper (picks the candidate) and once during evaluation (computes the reported
        aq value). Previously the evaluation build ran outside any seed context, so for the randomized
        straddle it drew a different beta (and used an unseeded sampler), making the reported aq value
        both non-reproducible and inconsistent with the optimized candidate. The evaluation is now run
        under the same resolved seed as the optimization.
        """
        X_train = pd.DataFrame({"x1": [0.1, 0.3, 0.5, 0.7, 0.9], "x2": [0.2, 0.4, 0.6, 0.8, 0.1]})
        Y_train = pd.DataFrame({"y": [3.0, 4.5, 6.0, 7.5, 4.0]})

        X_space = ParamSpace([Param_Continuous("x1", 0, 1), Param_Continuous("x2", 0, 1)])
        target = Target("y", aim="max", threshold=5.0)

        opt = BayesianOptimizer(X_space, seed=42, task="characterization", fix_random_state=True)
        opt.fit(pd.concat([X_train, Y_train], axis=1), target)

        _, eval1 = opt.suggest(acquisition=["RANDSTR"])
        _, eval2 = opt.suggest(acquisition=["RANDSTR"])

        np.testing.assert_allclose(eval1["aq Value"].values, eval2["aq Value"].values)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
