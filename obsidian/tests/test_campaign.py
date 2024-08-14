"""PyTests for obsidian.campaign"""

from obsidian.parameters import Target
from obsidian.experiment import Simulator
from obsidian.experiment.benchmark import two_leaves, shifted_parab
from obsidian.campaign import Campaign, Explainer, calc_ofat_ranges
from obsidian.objectives import Identity_Objective
from obsidian.plotting import plot_interactions, plot_ofat_ranges
from obsidian.exceptions import IncompatibleObjectiveError, UnfitError

from obsidian.tests.param_configs import X_sp_cont_ndims, X_sp_default
from obsidian.tests.utils import DEFAULT_MOO_PATH, equal_state_dicts

import pandas as pd
import pytest
import json
 
# Avoid using TkAgg which causes Tcl issues during testing
import matplotlib
matplotlib.use('inline')

target_test = [
    [Target(name='Response 1', f_transform='Standard', aim='max'),
     Target(name='Response 2', f_transform='Standard', aim='max')],
    Target(name='Response', f_transform='Standard', aim='max')
]


@pytest.mark.parametrize('X_space, sim_fcn, target',
                         [(X_sp_cont_ndims[2], two_leaves, target_test[0]),
                          (X_sp_default, shifted_parab, target_test[1])])
def test_campaign_basics(X_space, sim_fcn, target):
    # Standard usage
    campaign = Campaign(X_space, target)
    simulator = Simulator(X_space, sim_fcn, eps=0.05)
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
    Z0['Iteration'] = 5
    campaign.add_data(Z0)
    campaign.y
    campaign.fit()
    campaign.response_max

    # Serialize, deserialize, re-serialize
    obj_dict = campaign.save_state()
    campaign2 = Campaign.load_state(obj_dict)
    obj_dict2 = campaign2.save_state()
    assert equal_state_dicts(obj_dict, obj_dict2), 'Error during serialization'
    
    
# Load default
with open(DEFAULT_MOO_PATH) as json_file:
    obj_dict = json.load(json_file)
campaign = Campaign.load_state(obj_dict)
X_space = campaign.X_space
target = campaign.target


def test_explain():
    # Standard usage
    exp = Explainer(campaign.optimizer)
    exp.shap_explain(n=50)
    exp.__repr__

    # Test SHAP plots
    exp.shap_summary()
    exp.shap_summary_bar()
    
    # Test PDP-ICE, with options
    exp.shap_pdp_ice(ind=0, ice_color_var=None, npoints=10)
    exp.shap_pdp_ice(ind=0, npoints=10)
    exp.shap_pdp_ice(ind=(0, 1), npoints=5)

    # Test pairwise SHAP analysis, with options
    X_new = campaign.X.iloc[0, :]
    X_ref = campaign.X.loc[1, :]
    df_shap_value_new, fig_bar, fig_line = exp.shap_single_point(X_new)
    df_shap_value_new, fig_bar, fig_line = exp.shap_single_point(X_new, X_ref=X_ref)

    # Test sensitivity analysis, with options
    df_sens = exp.sensitivity()
    df_sens = exp.sensitivity(X_ref=X_ref)


X_ref_test = [None,
              campaign.X.iloc[campaign.y.idxmax()['Response 1'], :]]


@pytest.mark.parametrize('X_ref', X_ref_test)
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


# VALIDATION TESTS - Force errors to be raised in object usage

@pytest.mark.fast
def test_campaign_validation():
    
    # Missing X names
    random_data = pd.DataFrame(data={'A': [1, 2, 3], 'B': [4, 5, 6]})
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
        exp.shap_single_point(X_new=campaign.X_space.mean())
    
    random_data = pd.DataFrame(data={'A': [1], 'B': [4]})
    long_data = pd.DataFrame(data={'Parameter 1': [1, 2], 'Parameter 2': [1, 2]})
    
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
        exp.shap_single_point(X_new=campaign.X_space.mean(), X_ref=random_data)

    # Missing X names
    with pytest.raises(ValueError):
        exp.sensitivity(X_ref=random_data)
    
    # X_ref > 1 row
    with pytest.raises(ValueError):
        exp.sensitivity(X_ref=long_data)


if __name__ == '__main__':
    pytest.main([__file__, '-m', 'not slow'])
