"""PyTests for obsidian.constraints"""

from obsidian.campaign import Campaign
from obsidian.constraints import (
    Linear_Constraint,
    BatchVariance_Constraint,
    OutConstraint_Blank,
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

test_linear = [
    Linear_Constraint(X_space, ind=[0], weights=[1], rhs=5, equality=True),
    Linear_Constraint(X_space, ind=[0, 1], weights=[1, 1], rhs=5)
]
test_nonlinear = [BatchVariance_Constraint(X_space, ind=0, tol=0.1)]
test_out = [[OutConstraint_Blank(target)], [OutConstraint_L1(target, offset=1)]]

# Run very short optimizations for testing
test_config = {'optim_samples': 2, 'optim_restarts': 2}


@pytest.mark.parametrize('out_constraints', test_out)
def test_out_constraints(out_constraints):
    X_suggest, eval_suggest = optimizer.suggest(out_constraints=out_constraints,
                                                **test_config)
    df_suggest = pd.concat([X_suggest, eval_suggest], axis=1)


@pytest.mark.parametrize('lin_const', test_linear)
def test_ineq_constraints(lin_const):
    optimizer.X_space.constrain_inputs(lin_const)
    X_suggest, eval_suggest = optimizer.suggest(**test_config)
    df_suggest = pd.concat([X_suggest, eval_suggest], axis=1)
    optimizer.X_space.clear_constraints()
    
    
@pytest.mark.parametrize('nl_const', test_nonlinear)
def test_nleq_constraints(nl_const):
    optimizer.X_space.constrain_inputs(nl_const)
    X_suggest, eval_suggest = optimizer.suggest(**test_config)
    df_suggest = pd.concat([X_suggest, eval_suggest], axis=1)
    optimizer.X_space.clear_constraints()

    
@pytest.mark.slow
def test_combo_constraints():
    X_suggest, eval_suggest = optimizer.suggest(ineq_constraints=test_linear[1],
                                                nleq_constraints=test_nonlinear[0],
                                                out_constraints=test_out[0],
                                                **test_config)
    df_suggest = pd.concat([X_suggest, eval_suggest], axis=1)
    optimizer.X_space.clear_constraints()
    
    
if __name__ == '__main__':
    pytest.main([__file__, '-m', 'not slow'])
