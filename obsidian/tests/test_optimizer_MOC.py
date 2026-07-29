"""PyTests for obsidian.optimizer under multi-output characterization usage"""

import pandas as pd
import pytest

from obsidian.experiment import ExpDesigner, Simulator
from obsidian.experiment.benchmark import two_leaves
from obsidian.optimizer import BayesianOptimizer
from obsidian.parameters import Target
from obsidian.tests.param_configs import X_sp_cont_ndims

# Shared test config for speed
test_config = {"optim_samples": 8, "optim_restarts": 2}

target = [
    Target(name="Response 1", f_transform="Standard", aim="max"),
    Target(name="Response 2", f_transform="Standard", aim="max"),
]


@pytest.fixture()
def X_space():
    return X_sp_cont_ndims[2]


@pytest.fixture()
def Z0(X_space):
    designer = ExpDesigner(X_space, seed=0)
    X0 = designer.initialize(m_initial=len(X_space) * 2, method="LHS")
    simulator = Simulator(X_space, two_leaves, eps=0.05, rng=0)
    y0 = simulator.simulate(X0)
    return pd.concat([X0, y0], axis=1)


@pytest.fixture()
def optimizer(X_space, Z0):
    opt = BayesianOptimizer(X_space, surrogate="GP", task="characterization", seed=0, verbose=0)
    opt.fit(Z0, target=target)
    return opt


# Multi-objective characterization acquisitions (threshold list required)
moc_aqs = [
    pytest.param({"MSTR": {"threshold": [0.5, 0.5]}}, marks=pytest.mark.fast),
    pytest.param({"JAREX": {"threshold": [0.5, 0.5]}}, marks=pytest.mark.fast),
    pytest.param({"MRANDSTR": {"threshold": [0.5, 0.5]}}, id="MRANDSTR-alias"),
    pytest.param({"JAREX": {"threshold": [0.5, 0.5], "beta": [1.5, 1.5]}}, id="JAREX-fixed-beta"),
    pytest.param({"JAREX": {"threshold": [0.5, 0.5], "kernel_reduction": "sum"}}, id="JAREX-sum-kernel"),
    pytest.param({"JAREX": {"threshold": [0.5, 0.5], "kernel_reduction": "max"}}, id="JAREX-max-kernel"),
]


@pytest.mark.parametrize("aq", moc_aqs)
def test_moc_suggest(optimizer, aq):
    """Test that multi-output characterization acquisitions produce valid suggestions"""
    X_suggest, eval_suggest = optimizer.suggest(m_batch=2, acquisition=[aq], **test_config)
    assert len(X_suggest) == 2
    assert not X_suggest.isnull().any().any()


@pytest.mark.fast
def test_moc_jarex_alias(optimizer):
    """Test that MRANDSTR and JAREX produce equivalent behavior (same implementation)"""
    from obsidian.acquisition import registry

    assert registry.aq_class_dict["JAREX"] is registry.aq_class_dict["MRANDSTR"]


@pytest.mark.fast
def test_moc_determinism(X_space, Z0):
    """Test that MOC acquisitions are deterministic with fix_random_state=True"""
    opt = BayesianOptimizer(X_space, task="characterization", seed=42, fix_random_state=True)
    opt.fit(Z0, target=target)

    X1, _ = opt.suggest(m_batch=2, acquisition=[{"JAREX": {"threshold": [0.5, 0.5]}}], **test_config)
    X2, _ = opt.suggest(m_batch=2, acquisition=[{"JAREX": {"threshold": [0.5, 0.5]}}], **test_config)
    pd.testing.assert_frame_equal(X1, X2)


@pytest.mark.fast
def test_moc_wrong_threshold_count(optimizer):
    """Test that mismatched threshold count raises an error"""
    # _parse_thresholds raises ValueError for a list whose length does not match the number of targets.
    # Match on a substring of the real message so the test fails fast on unrelated bugs
    with pytest.raises(ValueError, match=r"Threshold list has \d+ values but \d+ targets"):
        optimizer.suggest(m_batch=1, acquisition=[{"MSTR": {"threshold": [0.5]}}], **test_config)


@pytest.mark.fast
def test_moc_dict_threshold(optimizer):
    """Test dict threshold specification"""
    X_suggest, _ = optimizer.suggest(
        m_batch=2, acquisition=[{"JAREX": {"threshold": {"Response 1": 0.5, "Response 2": 0.5}}}], **test_config
    )
    assert len(X_suggest) == 2


@pytest.mark.fast
def test_moc_dict_threshold_mismatch(optimizer):
    """Test that dict threshold with wrong keys raises an error"""
    with pytest.raises(ValueError, match="do not exactly match"):
        optimizer.suggest(
            m_batch=1, acquisition=[{"JAREX": {"threshold": {"Wrong Name": 0.5, "Response 2": 0.5}}}], **test_config
        )


@pytest.mark.fast
def test_moc_no_threshold_error(X_space, Z0):
    """Test that missing threshold raises an error"""
    targets_no_threshold = [
        Target(name="Response 1", f_transform="Standard", aim="max"),
        Target(name="Response 2", f_transform="Standard", aim="max"),
    ]
    opt = BayesianOptimizer(X_space, surrogate="GP", task="characterization", seed=0, verbose=0)
    opt.fit(Z0, target=targets_no_threshold)

    with pytest.raises(ValueError, match="No threshold specified"):
        opt.suggest(m_batch=1, acquisition=[{"JAREX": {}}], **test_config)


@pytest.mark.fast
def test_moc_sync_beta_warning(optimizer):
    """Test that sync_objective_beta for multi-objective emits a UserWarning."""
    # The sync_beta warning is emitted during _normalize_beta when sync_objective_beta=True
    # for multi-objective and beta_in is None (so it samples beta).
    with pytest.warns(UserWarning, match="Synchronized beta"):
        optimizer.suggest(
            m_batch=2,
            acquisition=[{"JAREX": {"threshold": [0.5, 0.5], "sync_objective_beta": True}}],
            **test_config,
        )


@pytest.mark.fast
def test_moc_beta_per_target(optimizer):
    """Test JAREX with beta per target"""
    X_suggest, _ = optimizer.suggest(
        m_batch=1, acquisition=[{"JAREX": {"threshold": [0.5, 0.5], "beta": [1.5, 2.0]}}], **test_config
    )
    assert len(X_suggest) == 1


@pytest.mark.fast
def test_moc_beta_per_batch(optimizer):
    """Test JAREX with beta per batch"""
    X_suggest, _ = optimizer.suggest(
        m_batch=2, acquisition=[{"JAREX": {"threshold": [0.5, 0.5], "beta": [1.5, 2.0]}}], **test_config
    )
    assert len(X_suggest) == 2


@pytest.mark.fast
def test_moc_beta_full_matrix(optimizer):
    """Test JAREX with full beta matrix (m_batch x num_targets)"""
    X_suggest, _ = optimizer.suggest(
        m_batch=2, acquisition=[{"JAREX": {"threshold": [0.5, 0.5], "beta": [1.5, 2.0, 1.8, 2.2]}}], **test_config
    )
    assert len(X_suggest) == 2


@pytest.mark.fast
def test_moc_beta_invalid_shape(optimizer):
    """Test JAREX with invalid beta shape raises an error"""
    with pytest.raises(ValueError, match="beta must be"):
        optimizer.suggest(
            m_batch=2, acquisition=[{"JAREX": {"threshold": [0.5, 0.5], "beta": [1.5, 2.0, 2.5]}}], **test_config
        )


@pytest.mark.fast
def test_moc_invalid_kernel_reduction(optimizer):
    """Test that invalid kernel_reduction raises an error"""
    with pytest.raises(ValueError, match="kernel_reduction must be"):
        optimizer.suggest(
            m_batch=1, acquisition=[{"JAREX": {"threshold": [0.5, 0.5], "kernel_reduction": "invalid"}}], **test_config
        )


@pytest.mark.fast
def test_moc_weights(optimizer):
    """Test JAREX with custom weights"""
    X_suggest, _ = optimizer.suggest(
        m_batch=2, acquisition=[{"JAREX": {"threshold": [0.5, 0.5], "weights": [1.0, 2.0]}}], **test_config
    )
    assert len(X_suggest) == 2


@pytest.mark.fast
def test_moc_weights_invalid_shape(optimizer):
    """Test JAREX with invalid weights shape raises an error"""
    with pytest.raises(ValueError, match="weights must be"):
        optimizer.suggest(
            m_batch=1, acquisition=[{"JAREX": {"threshold": [0.5, 0.5], "weights": [1.0]}}], **test_config
        )


@pytest.mark.fast
def test_moc_threshold_from_targets(X_space, Z0):
    """Test using threshold from Target objects (no explicit threshold in hyperparameters)"""
    targets_with_threshold = [
        Target(name="Response 1", f_transform="Standard", aim="max", threshold=0.5),
        Target(name="Response 2", f_transform="Standard", aim="max", threshold=0.5),
    ]
    opt = BayesianOptimizer(X_space, surrogate="GP", task="characterization", seed=0, verbose=0)
    opt.fit(Z0, target=targets_with_threshold)

    # Don't pass threshold in hyperparameters - should use targets' thresholds
    X_suggest, _ = opt.suggest(m_batch=1, acquisition=[{"JAREX": {}}], **test_config)
    assert len(X_suggest) == 1


@pytest.mark.fast
def test_mstr_no_beta(optimizer):
    """Test that MSTR uses default beta = 1.96**2 (squared parameter; σ-multiplier 1.96) when not specified"""
    X_suggest, _ = optimizer.suggest(m_batch=1, acquisition=[{"MSTR": {"threshold": [0.5, 0.5]}}], **test_config)
    assert len(X_suggest) == 1


@pytest.mark.fast
def test_jarex_no_weights(optimizer):
    """Test JAREX with no weights specified (line 486 - default weights)"""
    # When weights is None, it should use torch.ones_like(threshold)
    X_suggest, _ = optimizer.suggest(m_batch=1, acquisition=[{"JAREX": {"threshold": [0.5, 0.5]}}], **test_config)
    assert len(X_suggest) == 1


@pytest.mark.fast
def test_jarex_weights_numpy(optimizer):
    """Test JAREX with weights as numpy array (line 489)"""
    import numpy as np

    X_suggest, _ = optimizer.suggest(
        m_batch=1, acquisition=[{"JAREX": {"threshold": [0.5, 0.5], "weights": np.array([1.0, 2.0])}}], **test_config
    )
    assert len(X_suggest) == 1


@pytest.mark.fast
def test_jarex_kernel_reduction_max(optimizer):
    """Test JAREX with kernel_reduction='max' (line 582)"""
    X_suggest, _ = optimizer.suggest(
        m_batch=1, acquisition=[{"JAREX": {"threshold": [0.5, 0.5], "kernel_reduction": "max"}}], **test_config
    )
    assert len(X_suggest) == 1


@pytest.mark.fast
def test_jarex_kernel_reduction_sum_with_weights(optimizer):
    """Test JAREX with kernel_reduction='sum' and explicit weights (line 587)"""
    X_suggest, _ = optimizer.suggest(
        m_batch=1,
        acquisition=[{"JAREX": {"threshold": [0.5, 0.5], "kernel_reduction": "sum", "weights": [1.0, 1.0]}}],
        **test_config
    )
    assert len(X_suggest) == 1


@pytest.mark.fast
def test_jarex_with_generator(optimizer):
    """Test JAREX with explicit generator (line 651)"""
    import torch

    gen = torch.Generator().manual_seed(42)
    X_suggest, _ = optimizer.suggest(
        m_batch=1, acquisition=[{"JAREX": {"threshold": [0.5, 0.5], "generator": gen}}], **test_config
    )
    assert len(X_suggest) == 1


if __name__ == "__main__":
    pytest.main([__file__, "-m", "fast"])
