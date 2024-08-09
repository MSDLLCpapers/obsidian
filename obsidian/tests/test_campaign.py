
from obsidian.tests.param_configs import X_sp_cont_ndims, X_sp_default
from obsidian.parameters import Target
from obsidian.experiment import Simulator
from obsidian.experiment.benchmark import two_leaves, shifted_parab
from obsidian.campaign import Campaign
from obsidian.objectives import Identity_Objective, Scalar_WeightedNorm, Feature_Objective, \
    Objective_Sequence, Utopian_Distance, Index_Objective, Bounded_Target

from obsidian.tests.utils import DEFAULT_MOO_PATH
import json

import pandas as pd
import pytest


target_test = [
    [Target(name='Response 1', f_transform='Standard', aim='max'),
     Target(name='Response 2', f_transform='Standard', aim='max')],
    Target(name='Response', f_transform='Standard', aim='max')
]


@pytest.mark.parametrize('X_space, sim_fcn, target',
                         [(X_sp_cont_ndims[2], two_leaves, target_test[0]),
                          (X_sp_default, shifted_parab, target_test[1])])
def test_campaign(X_space, sim_fcn, target):
    campaign = Campaign(X_space, target)
    simulator = Simulator(X_space, sim_fcn, eps=0.05)
    X0 = campaign.suggest()
    y0 = simulator.simulate(X0)
    Z0 = pd.concat([X0, y0], axis=1)
    campaign.m_exp
    campaign.add_data(Z0)
    campaign.clear_data()
    Z0['Iteration'] = 5
    campaign.add_data(Z0)
    campaign.y
    campaign.fit()

    obj_dict = campaign.save_state()
    campaign2 = Campaign.load_state(obj_dict)
    campaign2.__repr__()
    
    campaign2.set_objective(Identity_Objective())
    campaign2.suggest()
    
    
with open(DEFAULT_MOO_PATH) as json_file:
    obj_dict = json.load(json_file)

campaign = Campaign.load_state(obj_dict)
X_space = campaign.X_space
target = campaign.target

test_objs = [Identity_Objective(), Scalar_WeightedNorm(weights=[1, 1]), Feature_Objective(X_space, indices=[0], coeff=[1]),
             Objective_Sequence([Utopian_Distance([1], target[0]), Index_Objective()]),
             Bounded_Target(bounds=[(0, 1)], targets=target[0])]


@pytest.mark.parametrize('obj', test_objs)
def test_campaign_objectives(obj):
    campaign.set_objective(obj)
    campaign.objective.__repr__()
    
    obj_dict = campaign.save_state()
    campaign2 = Campaign.load_state(obj_dict)
    campaign2.save_state()
    campaign2.__repr__()
       
        
if __name__ == '__main__':
    pytest.main([__file__, '-m', 'not slow'])
