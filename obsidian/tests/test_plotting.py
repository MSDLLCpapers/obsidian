from obsidian import Campaign
from obsidian.plotting import parity_plot, factor_plot, surface_plot, visualize_inputs

import pytest
from obsidian.tests.utils import DEFAULT_PATH
import json


with open(DEFAULT_PATH) as json_file:
    obj_dict = json.load(json_file)

campaign = Campaign.load_state(obj_dict)
optimizer = campaign.optimizer


@pytest.mark.fast
def test_parity_plot():
    fig = parity_plot(optimizer)


plots_and_objects = [pytest.param(parity_plot, optimizer, marks=pytest.mark.fast),
                     pytest.param(visualize_inputs, campaign, marks=pytest.mark.fast),
                     pytest.param(factor_plot, optimizer, marks=pytest.mark.fast),
                     (surface_plot, optimizer)]


@pytest.mark.parametrize('plot_type, acting_object', plots_and_objects)
def test_plots_fast(plot_type, acting_object):
    fig = plot_type(acting_object)


feature_ids = [i for i in range(len(optimizer.X_space))]


@pytest.mark.parametrize('feature_id', feature_ids)
def test_factor_plot(feature_id):
    fig = factor_plot(optimizer, feature_id=feature_id)


def test_factor_plot_options():
    X_ref = optimizer.X_train.iloc[-1, :].to_frame().T
    fig = factor_plot(optimizer, f_transform=True, X_ref=X_ref)


@pytest.mark.slow
@pytest.mark.parametrize('feature_id', feature_ids)
def test_surface_plot(feature_id):
    fig = surface_plot(optimizer, feature_ids=[0, feature_id])


@pytest.mark.slow
def test_surface_plot_options():
    fig = surface_plot(optimizer, plot_bands=False, plot_data=True)
    fig = surface_plot(optimizer, f_transform=True)


if __name__ == '__main__':
    pytest.main([__file__, '-m', 'not slow'])
