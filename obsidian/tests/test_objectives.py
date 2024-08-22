"""PyTests for obsidian.campaign"""

from obsidian.campaign import Campaign
from obsidian.objectives import (
    Identity_Objective,
    Scalar_WeightedNorm,
    Scalar_WeightedSum,
    Scalar_Chebyshev,
    Feature_Objective,
    Objective_Sequence,
    Utopian_Distance,
    Index_Objective,
    Bounded_Target
)

from obsidian.tests.utils import DEFAULT_MOO_PATH, equal_state_dicts
from obsidian.exceptions import IncompatibleObjectiveError

import pandas as pd
import pytest
import json

# Load default
with open(DEFAULT_MOO_PATH) as json_file:
    obj_dict = json.load(json_file)
campaign = Campaign.load_state(obj_dict)
X_space = campaign.X_space
target = campaign.target

# Run very short optimizations for testing
test_config = {'optim_samples': 2, 'optim_restarts': 2}

test_objs = [Identity_Objective(mo=len(target) > 1),
             Scalar_WeightedNorm(weights=[1, 1]),
             Feature_Objective(X_space, ind=(0,), coeff=(1,)),
             Objective_Sequence([Utopian_Distance([1], target[0]), Index_Objective()]),
             Bounded_Target(bounds=[(0, 1)]*len(target), targets=target),
             None]

utopian = Utopian_Distance(utopian=[10, 10], targets=target)

test_scalars = [Scalar_WeightedSum(weights=[0.5, 0.5]),
                Objective_Sequence([utopian, Scalar_WeightedSum(weights=[0.5, 0.5])]),
                Scalar_WeightedNorm(weights=[0.5, 0.5]),
                Objective_Sequence([utopian, Scalar_WeightedNorm(weights=[0.5, 0.5], neg=True)]),
                Scalar_Chebyshev(weights=[0.5, 0.5]),
                Scalar_Chebyshev(weights=[0.5, 0.5], augment=False),
                Objective_Sequence([utopian, Scalar_Chebyshev(weights=[0.5, 0.5])])]


@pytest.mark.parametrize('obj', test_objs + test_scalars)
def test_campaign_objectives(obj):
    # Set objective, read, examine output
    campaign.set_objective(obj)
    if campaign.objective:
        campaign.objective.__repr__()
    campaign.o

    # Serialize, deserialize, re-serialize
    obj_dict = campaign.save_state()
    campaign2 = Campaign.load_state(obj_dict)
    obj_dict2 = campaign2.save_state()
    assert equal_state_dicts(obj_dict, obj_dict2), 'Error during serialization'
    
    
@pytest.mark.parametrize('m_batch', [pytest.param(1, marks=pytest.mark.fast), 2, pytest.param(5, marks=pytest.mark.slow)])
@pytest.mark.parametrize('obj', test_objs)
def test_objective_suggestions(m_batch, obj):
    optimizer = campaign.optimizer
    if obj is not None:
        if obj._is_mo and optimizer.n_response == 1 and not isinstance(obj, Feature_Objective):
            with pytest.raises(IncompatibleObjectiveError):
                X_suggest, eval_suggest = optimizer.suggest(m_batch=m_batch, objective=obj,  **test_config)
        else:
            X_suggest, eval_suggest = optimizer.suggest(m_batch=m_batch, objective=obj,  **test_config)
    else:
        X_suggest, eval_suggest = optimizer.suggest(m_batch=m_batch, **test_config)
    df_suggest = pd.concat([X_suggest, eval_suggest], axis=1)
    
 
if __name__ == '__main__':
    pytest.main([__file__, '-m', 'not slow'])
