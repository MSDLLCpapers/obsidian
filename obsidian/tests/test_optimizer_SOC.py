"""PyTests for obsidian.optimizer under single-output characterization usage"""

from obsidian.tests.param_configs import X_sp_cont_small

from obsidian.parameters import Target
from obsidian.experiment import ExpDesigner, Simulator
from obsidian.optimizer import BayesianOptimizer
from obsidian.experiment.benchmark import shifted_parab

import pandas as pd
import pytest


# Shared test config for speed
test_config = {"optim_samples": 8, "optim_restarts": 2}

# Fixture: fitted optimizer for single-output characterization
@pytest.fixture()
def X_space():
    return X_sp_cont_small


@pytest.fixture()
def Z0(X_space):
    designer = ExpDesigner(X_space, seed=0)
    X0 = designer.initialize(m_initial=len(X_space) * 2, method="LHS")
    simulator = Simulator(X_space, shifted_parab, eps=0.05, rng=0)
    y0 = simulator.simulate(X0)
    return pd.concat([X0, y0], axis=1)


@pytest.fixture()
def optimizer(X_space, Z0):
    target = Target(name="Response", f_transform="Standard", aim="max")
    opt = BayesianOptimizer(X_space, surrogate="GP", task="characterization", seed=0, verbose=0)
    opt.fit(Z0, target=target)
    return opt


# Single-objective characterization acquisitions (threshold required)
soc_aqs = [
    pytest.param({"STR": {"threshold": 5.0}}, marks=pytest.mark.fast),
    pytest.param({"RANDSTR": {"threshold": 5.0}}, marks=pytest.mark.fast),
    pytest.param({"RANDSTR": {"threshold": 5.0, "beta": 2.0}}, id="RANDSTR-fixed-beta"),
]


@pytest.mark.parametrize("aq", soc_aqs)
def test_soc_suggest(optimizer, aq):
    """Test that single-output characterization acquisitions produce valid suggestions"""
    X_suggest, eval_suggest = optimizer.suggest(m_batch=2, acquisition=[aq], **test_config)
    assert len(X_suggest) == 2
    assert not X_suggest.isnull().any().any()


@pytest.mark.fast
def test_soc_suggest_batch(optimizer):
    """Test batch suggestion with RANDSTR"""
    X_suggest, eval_suggest = optimizer.suggest(
        m_batch=3, acquisition=[{"RANDSTR": {"threshold": 5.0}}], **test_config
    )
    assert len(X_suggest) == 3


@pytest.mark.fast
def test_soc_determinism(X_space, Z0):
    """Test that SOC acquisitions are deterministic with fix_random_state=True"""
    target = Target(name="Response", f_transform="Standard", aim="max")
    opt = BayesianOptimizer(X_space, task="characterization", seed=42, fix_random_state=True)
    opt.fit(Z0, target=target)

    X1, _ = opt.suggest(m_batch=2, acquisition=[{"RANDSTR": {"threshold": 5.0}}], **test_config)
    X2, _ = opt.suggest(m_batch=2, acquisition=[{"RANDSTR": {"threshold": 5.0}}], **test_config)
    pd.testing.assert_frame_equal(X1, X2)


@pytest.mark.fast
def test_soc_no_threshold_error(X_space, Z0):
    """Test that missing threshold raises an error"""
    target = Target(name="Response", f_transform="Standard", aim="max")  # No threshold
    opt = BayesianOptimizer(X_space, surrogate="GP", task="characterization", seed=0, verbose=0)
    opt.fit(Z0, target=target)

    with pytest.raises(ValueError, match="No threshold specified"):
        opt.suggest(m_batch=1, acquisition=[{"RANDSTR": {}}], **test_config)


@pytest.mark.fast
def test_soc_dict_threshold_warning(X_space, Z0):
    """Test that dict threshold for single-objective raises a warning"""
    target = Target(name="Response", f_transform="Standard", aim="max")
    opt = BayesianOptimizer(X_space, surrogate="GP", task="characterization", seed=0, verbose=0)
    opt.fit(Z0, target=target)

    with pytest.warns(UserWarning, match="redundant"):
        opt.suggest(m_batch=1, acquisition=[{"RANDSTR": {"threshold": {"Response": 5.0}}}], **test_config)


@pytest.mark.fast
def test_soc_invalid_threshold_type(X_space, Z0):
    """Test that invalid threshold type raises an error"""
    target = Target(name="Response", f_transform="Standard", aim="max")
    opt = BayesianOptimizer(X_space, surrogate="GP", task="characterization", seed=0, verbose=0)
    opt.fit(Z0, target=target)

    with pytest.raises(ValueError, match="Invalid threshold type"):
        opt.suggest(m_batch=1, acquisition=[{"RANDSTR": {"threshold": "invalid"}}], **test_config)


@pytest.mark.fast
def test_soc_beta_list(optimizer):
    """Test RANDSTR with beta as list (per-batch)"""
    X_suggest, _ = optimizer.suggest(
        m_batch=2, acquisition=[{"RANDSTR": {"threshold": 5.0, "beta": [1.5, 2.0]}}], **test_config
    )
    assert len(X_suggest) == 2


@pytest.mark.fast
def test_soc_beta_invalid_shape(optimizer):
    """Test RANDSTR with invalid beta shape raises an error"""
    with pytest.raises(ValueError, match="beta must be"):
        optimizer.suggest(
            m_batch=2, acquisition=[{"RANDSTR": {"threshold": 5.0, "beta": [1.5, 2.0, 2.5]}}], **test_config
        )


@pytest.mark.fast
def test_soc_threshold_from_target(X_space, Z0):
    """Test using threshold from Target object (no explicit threshold in hyperparameters)"""
    target = Target(name="Response", f_transform="Standard", aim="max", threshold=5.0)
    opt = BayesianOptimizer(X_space, surrogate="GP", task="characterization", seed=0, verbose=0)
    opt.fit(Z0, target=target)

    # Don't pass threshold in hyperparameters - should use target.threshold
    X_suggest, _ = opt.suggest(m_batch=1, acquisition=[{"RANDSTR": {}}], **test_config)
    assert len(X_suggest) == 1


@pytest.mark.fast
def test_str_no_beta(optimizer):
    """Test that STR uses default beta = 1.96**2 (squared parameter; σ-multiplier 1.96) when not specified"""
    X_suggest, _ = optimizer.suggest(
        m_batch=1, acquisition=[{"STR": {"threshold": 5.0}}], **test_config
    )
    assert len(X_suggest) == 1


@pytest.mark.fast
def test_randstr_with_generator(optimizer):
    """Test RANDSTR with explicit generator (line 399)"""
    import torch
    gen = torch.Generator().manual_seed(42)
    X_suggest, _ = optimizer.suggest(
        m_batch=1,
        acquisition=[{"RANDSTR": {"threshold": 5.0, "generator": gen}}],
        **test_config
    )
    assert len(X_suggest) == 1


@pytest.mark.fast
def test_randstr_scalar_beta(optimizer):
    """Test RANDSTR with scalar beta (line 186)"""
    X_suggest, _ = optimizer.suggest(
        m_batch=1,
        acquisition=[{"RANDSTR": {"threshold": 5.0, "beta": 2.5}}],
        **test_config
    )
    assert len(X_suggest) == 1


if __name__ == "__main__":
    pytest.main([__file__, "-m", "fast"])
