"""Tests for ``plot_2d_response_map`` (reduced dimension heat map)."""

import matplotlib

matplotlib.use("Agg")  # noqa: E402

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest

from obsidian.campaign import Campaign
from obsidian.parameters import Param_Continuous, ParamSpace, Target
from obsidian.plotting.mpl import plot_2d_response_map


def _make_simple_campaign(
    threshold: float | None = 5.0,
    tracking_only: bool = False,
    task: str = "characterization",
) -> Campaign:
    X_space = ParamSpace(
        [
            Param_Continuous("x1", 0.0, 1.0),
            Param_Continuous("x2", 0.0, 1.0),
        ]
    )
    target = Target("y", aim="max", threshold=threshold, tracking_only=tracking_only)
    campaign = Campaign(X_space, target, task=task, seed=42)

    X_train = pd.DataFrame({"x1": [0.1, 0.3, 0.5, 0.7, 0.9], "x2": [0.2, 0.4, 0.6, 0.8, 0.1]})
    y_train = pd.DataFrame({"y": [3.0, 4.5, 6.0, 7.5, 4.0]})
    campaign.add_data(pd.concat([X_train, y_train], axis=1))
    campaign.fit()
    return campaign


def _make_multi_campaign() -> Campaign:
    X_space = ParamSpace(
        [
            Param_Continuous("x1", 0.0, 1.0),
            Param_Continuous("x2", 0.0, 1.0),
            Param_Continuous("x3", 0.0, 1.0),
        ]
    )
    targets = [
        Target("y1", aim="max", threshold=5.0),
        Target("y2", aim="min", threshold=2.0),
    ]
    campaign = Campaign(X_space, targets, task="characterization", seed=42)
    X_train = pd.DataFrame(
        {
            "x1": [0.1, 0.3, 0.5, 0.7, 0.9],
            "x2": [0.2, 0.4, 0.6, 0.8, 0.1],
            "x3": [0.5, 0.5, 0.5, 0.5, 0.5],
        }
    )
    y_train = pd.DataFrame(
        {
            "y1": [3.0, 4.5, 6.0, 7.5, 4.0],
            "y2": [4.0, 3.0, 1.5, 1.0, 2.5],
        }
    )
    campaign.add_data(pd.concat([X_train, y_train], axis=1))
    campaign.fit()
    return campaign


@pytest.fixture
def fitted_campaign():
    return _make_simple_campaign()


@pytest.fixture
def fitted_campaign_no_threshold():
    return _make_simple_campaign(threshold=None, task="optimization")


@pytest.fixture
def multi_campaign():
    return _make_multi_campaign()


@pytest.fixture(autouse=True)
def _close_figures():
    yield
    plt.close("all")


ROW_DEFAULT = {"x1": [0.0, 0.5, 1.0]}
COL_DEFAULT = {"x2": [0.0, 0.5, 1.0]}
FIXED_X3: dict[str, float | str] = {"x3": 0.5}


class TestRendering:
    def test_passfail_mode_renders(self, fitted_campaign):
        fig, axes = plot_2d_response_map(fitted_campaign, row_params=ROW_DEFAULT, col_params=COL_DEFAULT)
        assert fig is not None
        assert axes.shape == (1, 1)

    def test_continuous_mode_renders(self, fitted_campaign_no_threshold):
        fig, axes = plot_2d_response_map(
            fitted_campaign_no_threshold,
            row_params=ROW_DEFAULT,
            col_params=COL_DEFAULT,
            mode="continuous",
        )
        assert axes.shape == (1, 1)

    def test_confidence_mode_has_four_levels(self, fitted_campaign):
        fig, axes = plot_2d_response_map(
            fitted_campaign,
            row_params=ROW_DEFAULT,
            col_params=COL_DEFAULT,
            mode="confidence",
        )
        assert axes.shape == (1, 1)
        images = axes[0, 0].get_images()
        assert len(images) == 1
        arr = images[0].get_array()
        unique = set(np.unique(arr).astype(int).tolist())
        assert unique.issubset({0, 1, 2, 3})

    def test_multi_target_passfail_includes_joint(self, multi_campaign):
        fig, axes = plot_2d_response_map(
            multi_campaign,
            row_params=ROW_DEFAULT,
            col_params=COL_DEFAULT,
            fixed_params=FIXED_X3,
            include_joint=True,
        )
        # 2 targets + 1 joint panel
        assert axes.shape == (1, 3)

    def test_multi_target_passfail_omits_joint(self, multi_campaign):
        fig, axes = plot_2d_response_map(
            multi_campaign,
            row_params=ROW_DEFAULT,
            col_params=COL_DEFAULT,
            fixed_params=FIXED_X3,
            include_joint=False,
        )
        assert axes.shape == (1, 2)

    def test_grid_shape_matches_row_and_col_counts(self, fitted_campaign):
        row_params = {"x1": [0.0, 0.25, 0.5, 0.75, 1.0]}  # R = 5
        fig, axes = plot_2d_response_map(fitted_campaign, row_params=row_params, col_params=COL_DEFAULT)
        arr = axes[0, 0].get_images()[0].get_array()
        assert arr.shape == (5, 3)


# Hierarchical row/col specification.
class TestHierarchical:
    def test_nested_levels_compose(self, multi_campaign):
        # Use 2 levels on each axis (R = 2*3 = 6, C = 2*2 = 4)
        fig, axes = plot_2d_response_map(
            multi_campaign,
            row_params={"x1": [0.0, 0.5, 1.0], "x3": [0.25, 0.75]},
            col_params={"x2": [0.25, 0.75]},
            target_names=["y1"],
            mode="passfail",
        )
        arr = axes[0, 0].get_images()[0].get_array()
        assert arr.shape == (6, 2)


# Validation: every campaign X param must appear in exactly one of
# row_params / col_params / fixed_params. row/col take lists, fixed takes scalars.
class TestParamValidation:
    def test_uncovered_campaign_param_rejected(self, multi_campaign):
        # x3 is in neither row, col, nor fixed.
        with pytest.raises(ValueError, match="Missing"):
            plot_2d_response_map(
                multi_campaign,
                row_params={"x1": [0.0, 1.0]},
                col_params={"x2": [0.0, 1.0]},
            )

    def test_param_not_in_campaign_rejected(self, fitted_campaign):
        with pytest.raises(ValueError, match="not in campaign"):
            plot_2d_response_map(
                fitted_campaign,
                row_params={"nope": [0.0, 1.0]},
                col_params={"x2": [0.0, 1.0]},
            )

    def test_param_in_both_row_and_col_rejected(self, fitted_campaign):
        with pytest.raises(ValueError, match="more than one"):
            plot_2d_response_map(
                fitted_campaign,
                row_params={"x1": [0.0, 1.0]},
                col_params={"x1": [0.0, 1.0]},
            )

    def test_param_in_both_row_and_fixed_rejected(self, multi_campaign):
        with pytest.raises(ValueError, match="more than one"):
            plot_2d_response_map(
                multi_campaign,
                row_params={"x1": [0.0, 1.0]},
                col_params={"x2": [0.0, 1.0]},
                fixed_params={"x1": 0.5, "x3": 0.5},
            )

    def test_row_value_list_must_be_non_empty(self, fitted_campaign):
        with pytest.raises(ValueError, match="at least one"):
            plot_2d_response_map(
                fitted_campaign,
                row_params={"x1": []},
                col_params={"x2": [0.0, 1.0]},
            )

    def test_row_value_must_be_a_list_not_scalar(self, fitted_campaign):
        with pytest.raises(ValueError, match="list of values"):
            plot_2d_response_map(
                fitted_campaign,
                row_params={"x1": 0.5},  # type: ignore
                col_params={"x2": [0.0, 1.0]},
            )

    def test_row_dict_must_be_non_empty(self, fitted_campaign):
        with pytest.raises(ValueError, match="non-empty"):
            plot_2d_response_map(
                fitted_campaign,
                row_params={},
                col_params={"x2": [0.0, 1.0]},
            )

    def test_continuous_mode_requires_single_target(self, multi_campaign):
        with pytest.raises(ValueError, match="exactly one target"):
            plot_2d_response_map(
                multi_campaign,
                row_params=ROW_DEFAULT,
                col_params=COL_DEFAULT,
                fixed_params=FIXED_X3,
                mode="continuous",
            )


# Threshold detection.
class TestThresholdValidation:
    def test_passfail_mode_requires_threshold(self):
        X_space = ParamSpace([Param_Continuous("x1", 0.0, 1.0), Param_Continuous("x2", 0.0, 1.0)])
        targets = [
            Target("y1", aim="max", threshold=5.0),
            Target("y2", aim="max"),  # no threshold
        ]
        campaign = Campaign(X_space, targets, task="characterization", seed=42)
        X_train = pd.DataFrame({"x1": [0.1, 0.3, 0.5, 0.7, 0.9], "x2": [0.2, 0.4, 0.6, 0.8, 0.1]})
        y_train = pd.DataFrame({"y1": [3.0, 4.5, 6.0, 7.5, 4.0], "y2": [1.0, 2.0, 3.0, 4.0, 5.0]})
        campaign.add_data(pd.concat([X_train, y_train], axis=1))
        campaign.fit()

        with pytest.raises(ValueError, match="Thresholds are required"):
            plot_2d_response_map(
                campaign,
                row_params={"x1": [0.0, 0.5, 1.0]},
                col_params={"x2": [0.0, 0.5, 1.0]},
                target_names=["y1", "y2"],
            )


if __name__ == "__main__":
    pytest.main([__file__, "-q"])
