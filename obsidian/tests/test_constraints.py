"""PyTests for obsidian.constraints"""

from obsidian.campaign import Campaign
from obsidian.constraints import (
    OutConstraint_Blank,
    InConstraint_Generic,
    InConstraint_ConstantDim,
    OutConstraint_L1
)
from obsidian.tests.utils import DEFAULT_MOO_PATH

import pandas as pd
import pytest
import json

# Load defaults
with open(DEFAULT_MOO_PATH) as json_file:
    obj_dict = json.load(json_file)
campaign = Campaign.load_state(obj_dict)

optimizer = campaign.optimizer
X_space = campaign.X_space
target = campaign.target

test_ineq = [[InConstraint_Generic(X_space, indices=[0, 1], coeff=[1, 1], rhs=5)]]
test_nleq = [[InConstraint_ConstantDim(X_space, dim=0, tol=0.1)]]
test_out = [[OutConstraint_Blank(target)], [OutConstraint_L1(target, offset=1)]]

# Run very short optimizations for testing
test_config = {'optim_samples': 2, 'optim_restarts': 2}


@pytest.mark.parametrize('ineq_constraints', test_ineq)
def test_ineq_constraints(ineq_constraints):
    X_suggest, eval_suggest = optimizer.suggest(ineq_constraints=ineq_constraints,
                                                **test_config)
    df_suggest = pd.concat([X_suggest, eval_suggest], axis=1)
    
    
@pytest.mark.parametrize('nleq_constraints', test_nleq)
def test_nleq_constraints(nleq_constraints):
    X_suggest, eval_suggest = optimizer.suggest(nleq_constraints=nleq_constraints,
                                                **test_config)
    df_suggest = pd.concat([X_suggest, eval_suggest], axis=1)
    
    
@pytest.mark.parametrize('out_constraints', test_out)
def test_out_constraints(out_constraints):
    X_suggest, eval_suggest = optimizer.suggest(out_constraints=out_constraints,
                                                **test_config)
    df_suggest = pd.concat([X_suggest, eval_suggest], axis=1)
    
    
@pytest.mark.slow
def test_combo_constraints():
    X_suggest, eval_suggest = optimizer.suggest(ineq_constraints=test_ineq[0],
                                                nleq_constraints=test_nleq[0],
                                                out_constraints=test_out[0],
                                                **test_config)
    df_suggest = pd.concat([X_suggest, eval_suggest], axis=1)
    
    
if __name__ == '__main__':
    pytest.main([__file__, '-m', 'not slow'])
