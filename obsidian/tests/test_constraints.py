"""PyTests for obsidian.constraints"""

from obsidian.campaign import Campaign
from obsidian.constraints import (
    Linear_Constraint,
    BatchVariance_Constraint,
    Blank_Constraint,
    L1_Constraint,
    ThresholdConstraint,
    UpperBoundConstraint,
    LowerBoundConstraint,
    InRangeConstraint,
)
from obsidian.tests.utils import DEFAULT_MOO_PATH

import pandas as pd
import pytest
import json

# Load defaults
with open(DEFAULT_MOO_PATH) as json_file:
    obj_dict = json.load(json_file)
    # Override seeds for determinism with new RNG control
    obj_dict['seed'] = 114514
    obj_dict['optimizer']['opt_attrs']['seed'] = 114514
campaign = Campaign.load_state(obj_dict)

optimizer = campaign.optimizer
X_space = campaign.X_space
target = campaign.target

test_linear = [
    Linear_Constraint(X_space, ind=[0], weights=[1], rhs=5, equality=True),
    Linear_Constraint(X_space, ind=[0, 1], weights=[1, 1], rhs=5)
]
test_nonlinear = [BatchVariance_Constraint(X_space, ind=0, tol=0.1)]
test_out = [Blank_Constraint(target), L1_Constraint(target, offset=1)]

# Run very short optimizations for testing
test_config = {'optim_samples': 2, 'optim_restarts': 2}


@pytest.mark.parametrize('out_const', test_out)
def test_out_constraints(out_const):
    out_const.__repr__()
    campaign.constrain_outputs(out_const)
    X_suggest, eval_suggest = campaign.optimizer.suggest(**test_config)
    df_suggest = pd.concat([X_suggest, eval_suggest], axis=1)
    campaign.clear_output_constraints()


@pytest.mark.parametrize('lin_const', test_linear)
def test_ineq_constraints(lin_const):
    lin_const.__repr__()
    optimizer.X_space.constrain_inputs(lin_const)
    X_suggest, eval_suggest = optimizer.suggest(**test_config)
    df_suggest = pd.concat([X_suggest, eval_suggest], axis=1)
    optimizer.X_space.clear_constraints()
    
    
@pytest.mark.parametrize('nl_const', test_nonlinear)
def test_nleq_constraints(nl_const):
    nl_const.__repr__()
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


# ============================================================================
# Threshold Constraint Tests
# ============================================================================

def test_threshold_constraint_instantiation():
    """Test that all threshold constraint classes instantiate correctly."""
    target_name = target[0].name

    # Upper bound only
    upper_const = UpperBoundConstraint(target, target_name, upper=95.0)
    assert upper_const.has_upper
    assert not upper_const.has_lower
    assert upper_const.target_name == target_name

    # Lower bound only
    lower_const = LowerBoundConstraint(target, target_name, lower=70.0)
    assert lower_const.has_lower
    assert not lower_const.has_upper
    assert lower_const.target_name == target_name

    # Range bound
    range_const = InRangeConstraint(target, target_name, lower=70.0, upper=95.0)
    assert range_const.has_lower
    assert range_const.has_upper
    assert range_const.target_name == target_name

    # Base class with optional bounds
    threshold_const = ThresholdConstraint(target, target_name, lower=70.0)
    assert threshold_const.has_lower
    assert not threshold_const.has_upper


def test_threshold_constraint_repr():
    """Test string representations."""
    target_name = target[0].name

    upper_const = UpperBoundConstraint(target, target_name, upper=95.0)
    repr_str = repr(upper_const)
    assert 'UpperBoundConstraint' in repr_str
    assert target_name in repr_str
    assert 'upper=95.0' in repr_str


def test_threshold_constraint_validation():
    """Test input validation."""
    target_name = target[0].name

    # Must provide at least one bound
    with pytest.raises(ValueError, match="At least one of 'lower' or 'upper' must be specified"):
        ThresholdConstraint(target, target_name)

    # Target name must exist
    with pytest.raises(ValueError, match="Target .* not found"):
        UpperBoundConstraint(target, "NonexistentTarget", upper=95.0)

    # Range bound: lower must be less than upper
    with pytest.raises(ValueError, match="Lower bound .* must be less than upper bound"):
        InRangeConstraint(target, target_name, lower=95.0, upper=70.0)


def test_threshold_constraint_forward():
    """Test constraint forward pass logic."""
    import torch
    target_name = target[0].name

    # Upper bound: output <= 95
    upper_const = UpperBoundConstraint(target, target_name, upper=95.0)
    constraint_fn = upper_const.forward(scale=False)

    # Test with dummy samples (single output)
    samples = torch.tensor([[90.0], [95.0], [100.0]])
    violations = constraint_fn(samples)

    # 90 <= 95: feasible (negative)
    assert violations[0].item() < 0
    # 95 <= 95: feasible (zero or negative)
    assert violations[1].item() <= 0
    # 100 <= 95: infeasible (positive)
    assert violations[2].item() > 0

    # Lower bound: output >= 70
    lower_const = LowerBoundConstraint(target, target_name, lower=70.0)
    constraint_fn = lower_const.forward(scale=False)

    samples = torch.tensor([[65.0], [70.0], [75.0]])
    violations = constraint_fn(samples)

    # 65 >= 70: infeasible (positive)
    assert violations[0].item() > 0
    # 70 >= 70: feasible (zero or negative)
    assert violations[1].item() <= 0
    # 75 >= 70: feasible (negative)
    assert violations[2].item() < 0

    # Range bound: 70 <= output <= 95
    range_const = InRangeConstraint(target, target_name, lower=70.0, upper=95.0)
    constraint_fn = range_const.forward(scale=False)

    samples = torch.tensor([[65.0], [80.0], [100.0]])
    violations = constraint_fn(samples)

    # 65: below lower bound, infeasible
    assert violations[0].item() > 0
    # 80: within range, feasible
    assert violations[1].item() < 0
    # 100: above upper bound, infeasible
    assert violations[2].item() > 0


def test_threshold_constraint_with_campaign():
    """Test threshold constraints work with campaign.suggest()."""
    target_name = target[0].name

    # Create constraint
    constraint = InRangeConstraint(target, target_name, lower=0.3, upper=0.8)

    # Apply to campaign
    campaign.constrain_outputs(constraint)

    # Generate suggestions (should respect constraint)
    X_suggest, eval_suggest = campaign.optimizer.suggest(**test_config)
    df_suggest = pd.concat([X_suggest, eval_suggest], axis=1)

    # Clean up
    campaign.clear_output_constraints()

    # Verify we got suggestions (constraint didn't break optimization)
    assert len(X_suggest) > 0


# Test parametrized threshold constraints
test_threshold = [
    UpperBoundConstraint(target, target[0].name, upper=0.9),
    LowerBoundConstraint(target, target[0].name, lower=0.2),
    InRangeConstraint(target, target[0].name, lower=0.2, upper=0.9),
]


@pytest.mark.parametrize('threshold_const', test_threshold)
def test_threshold_constraints_parametrized(threshold_const):
    """Parametrized test for all threshold constraint types."""
    threshold_const.__repr__()
    campaign.constrain_outputs(threshold_const)
    X_suggest, eval_suggest = campaign.optimizer.suggest(**test_config)
    df_suggest = pd.concat([X_suggest, eval_suggest], axis=1)
    campaign.clear_output_constraints()


if __name__ == '__main__':
    pytest.main([__file__, '-m', 'not slow'])
