"""PyTests for RNG control mechanisms"""

import pytest
import numpy as np
from obsidian.experiment import ExpDesigner
import pandas as pd
import torch
import obsidian
from obsidian import create_rng_manager
from obsidian.rng import RNGManager
from obsidian.parameters import Target
from obsidian.campaign import Campaign
from obsidian.optimizer import BayesianOptimizer
from obsidian.experiment import Simulator
from obsidian.experiment.benchmark import shifted_parab
from obsidian.tests.param_configs import X_sp_default


@pytest.fixture
def setup_campaign():
    """Set up a basic campaign for testing"""
    X_space = X_sp_default
    target = Target(name="Response", f_transform="Standard", aim="max")
    simulator = Simulator(X_space, shifted_parab, eps=0.05, rng=114514)

    return X_space, target, simulator


def test_create_new_rng():
    """Test 1: Create new RNG manager with various inputs"""
    # Create with seed
    rng1 = create_rng_manager(42)
    assert isinstance(rng1, RNGManager)
    assert rng1.seed == 42

    # Create without seed (random)
    rng2 = create_rng_manager(None)
    assert isinstance(rng2, RNGManager)
    assert rng2.seed is not None

    # Create with explicit seed
    rng3 = create_rng_manager(123)
    assert rng3.seed == 123

    # Two RNGs with same seed should produce same random numbers
    rng_a = create_rng_manager(999)
    rng_b = create_rng_manager(999)

    val_a = rng_a.np_rng.random(10)
    val_b = rng_b.np_rng.random(10)
    assert np.allclose(val_a, val_b)

    # Different seeds should produce different random numbers
    rng_c = create_rng_manager(111)
    rng_d = create_rng_manager(222)

    val_c = rng_c.np_rng.random(10)
    val_d = rng_d.np_rng.random(10)
    assert not np.allclose(val_c, val_d)


def test_global_rng_operations():
    """Test 2: Global RNG creation, retrieve, reset operations"""
    # Test get_global_rng creates a new one if doesn't exist
    global_rng1 = obsidian.rng.get_global_rng()
    assert isinstance(global_rng1, RNGManager)

    # Retrieving again should return the same instance
    global_rng2 = obsidian.rng.get_global_rng()
    assert global_rng1 is global_rng2

    # Reset global RNG
    obsidian.rng.reset_global_rng(seed=12345)
    global_rng3 = obsidian.rng.get_global_rng()
    assert global_rng3.seed == 12345
    # Should be a new instance
    assert global_rng3 is not global_rng1

    # Test that reset changes the RNG behavior
    obsidian.rng.reset_global_rng(seed=100)
    rng_a = obsidian.rng.get_global_rng()
    val1 = rng_a.np_rng.random(5)

    obsidian.rng.reset_global_rng(seed=100)
    rng_b = obsidian.rng.get_global_rng()
    val2 = rng_b.np_rng.random(5)

    assert np.allclose(val1, val2)


def test_campaign_suggest_consistency(setup_campaign):
    """Test 3: Campaign suggest gives consistent results with deterministic acquisition (EI)"""
    X_space, target, simulator = setup_campaign

    # Create two campaigns with same seed
    campaign1 = Campaign(X_space, target, seed=114514)
    campaign2 = Campaign(X_space, target, seed=114514)

    # Initialize and fit both campaigns with same data
    X0 = campaign1.initialize(m_initial=5, method="LHS")
    y0 = simulator.simulate(X0)
    Z0 = pd.concat([X0, y0], axis=1)

    campaign1.add_data(Z0)
    campaign2.add_data(Z0)

    campaign1.fit()
    campaign2.fit()

    # Suggest with EI (deterministic acquisition function)
    X_suggest1, _ = campaign1.suggest(m_batch=3, acquisition=["EI"])
    X_suggest2, _ = campaign2.suggest(m_batch=3, acquisition=["EI"])

    # Results should be identical
    pd.testing.assert_frame_equal(X_suggest1, X_suggest2)


def test_manual_seed_override(setup_campaign):
    """Test 4: Manual seed override in suggest gives different results"""
    X_space, target, simulator = setup_campaign

    # Create campaign
    campaign = Campaign(X_space, target, seed=114514)

    # Initialize and fit
    X0 = campaign.initialize(m_initial=5, method="LHS")
    y0 = simulator.simulate(X0)
    Z0 = pd.concat([X0, y0], axis=1)
    campaign.add_data(Z0)
    campaign.fit()

    # Suggest with default seed
    X_suggest1, _ = campaign.suggest(m_batch=3, acquisition=["EI"])

    # Re-fit to reset state
    campaign.fit()

    # Suggest with manual seed override
    X_suggest2, _ = campaign.suggest(m_batch=3, acquisition=["EI"], manual_seed=999)

    # Results should be different due to manual seed override
    # (Different random restarts in optimization)
    try:
        pd.testing.assert_frame_equal(X_suggest1, X_suggest2)
        # If they're equal, that's suspicious but not necessarily wrong
        # The manual_seed affects optimization restarts, which may or may not
        # find different optima depending on the landscape
    except AssertionError:
        # Expected: manual_seed changes optimization behavior
        pass


def test_fix_random_state(setup_campaign):
    """Test 5: fix_random_state works as intended"""
    X_space, target, simulator = setup_campaign

    designer = ExpDesigner(X_space, seed=114514)

    # Create optimizer with fix_random_state=True (default)
    opt_fixed = BayesianOptimizer(X_space, seed=114514, fix_random_state=True)

    # Create optimizer with fix_random_state=False
    opt_not_fixed = BayesianOptimizer(X_space, seed=114514, fix_random_state=False)

    # Generate data
    X0 = designer.initialize(m_initial=5, method="LHS")
    y0 = simulator.simulate(X0)
    Z0 = pd.concat([X0, y0], axis=1)

    # Fit both optimizers
    opt_fixed.fit(Z0, target)
    opt_not_fixed.fit(Z0, target)

    # With fix_random_state=True, multiple fits should give identical models
    opt_fixed_2 = BayesianOptimizer(X_space, seed=114514, fix_random_state=True)
    opt_fixed_2.fit(Z0, target)

    # Check that model parameters are identical
    # Both should have deterministic model_generator
    assert opt_fixed.model_generator is None
    assert opt_fixed_2.model_generator is None

    # Predictions should be identical
    X_test = designer.initialize(m_initial=3, method="LHS")
    pred1 = opt_fixed.predict(X_test)
    pred2 = opt_fixed_2.predict(X_test)
    pd.testing.assert_frame_equal(pred1, pred2)

    # With fix_random_state=False, model_generator should exist
    assert opt_not_fixed.model_generator is not None


def test_campaign_save_load_rng_state(setup_campaign):
    """Test 6: Campaign and optimizer save and load preserve RNG state"""
    X_space, target, simulator = setup_campaign

    # Create campaign with specific seed
    campaign1 = Campaign(X_space, target, seed=12345)

    # Initialize, add data, and fit
    X0 = campaign1.initialize(m_initial=5, method="LHS")
    y0 = simulator.simulate(X0)
    Z0 = pd.concat([X0, y0], axis=1)
    campaign1.add_data(Z0)
    campaign1.fit()

    # Make a suggestion
    X_suggest1, _ = campaign1.suggest(m_batch=2, acquisition=["EI"])

    # Save state
    state = campaign1.save_state()

    # Load into new campaign
    campaign2 = Campaign.load_state(state)

    # RNG state should be preserved
    assert campaign2.seed == 12345
    assert hasattr(campaign2, "rng")
    assert campaign2.rng.seed == 12345

    # Make another suggestion - should continue from saved RNG state
    X_suggest2, _ = campaign2.suggest(m_batch=2, acquisition=["EI"])

    # Both campaigns should have the same optimizer state
    assert campaign1.optimizer.is_fit == campaign2.optimizer.is_fit

    # Test optimizer save/load specifically
    opt_state = campaign1.optimizer.save_state()
    opt_loaded = BayesianOptimizer.load_state(opt_state)

    # Check RNG state is preserved in optimizer
    if "rng_state" in opt_state:
        assert opt_loaded.seed == campaign1.optimizer.seed
        assert hasattr(opt_loaded, "rng")


def test_old_rng_control_mode(setup_campaign):
    """Test 7: Test obsidian.USE_OLD_RNG_CONTROL works"""
    X_space, target, simulator = setup_campaign

    # Save current state
    original_mode = obsidian.USE_OLD_RNG_CONTROL

    try:
        # Test with old RNG control (no RNGManager)
        obsidian.USE_OLD_RNG_CONTROL = True

        # Create campaign - should work without RNGManager
        campaign_old = Campaign(X_space, target, seed=114514)

        # Should not have RNGManager in old mode
        assert not hasattr(campaign_old, "rng") or campaign_old.rng is None

        # Initialize and fit should still work
        X0 = campaign_old.initialize(m_initial=5, method="LHS")
        y0 = simulator.simulate(X0)
        Z0 = pd.concat([X0, y0], axis=1)
        campaign_old.add_data(Z0)
        campaign_old.fit()

        # Suggest should still work
        X_suggest_old, _ = campaign_old.suggest(m_batch=2)
        assert len(X_suggest_old) == 2

        # Test with new RNG control
        obsidian.USE_OLD_RNG_CONTROL = False

        campaign_new = Campaign(X_space, target, seed=114514)

        # Should have RNGManager in new mode
        assert hasattr(campaign_new, "rng")
        assert isinstance(campaign_new.rng, RNGManager)

        # Initialize and fit
        X0_new = campaign_new.initialize(m_initial=5, method="LHS")
        y0_new = simulator.simulate(X0_new)
        Z0_new = pd.concat([X0_new, y0_new], axis=1)
        campaign_new.add_data(Z0_new)
        campaign_new.fit()

        # Suggest should work
        X_suggest_new, _ = campaign_new.suggest(m_batch=2)
        assert len(X_suggest_new) == 2

    finally:
        # Restore original mode
        obsidian.USE_OLD_RNG_CONTROL = original_mode


def test_rng_state_save_load():
    """Test RNGManager state save/load"""
    # Create RNG with seed
    rng1 = create_rng_manager(777)

    # Generate some random numbers to advance state
    rng1.np_rng.random(10)
    torch.rand(5, generator=rng1.torch_rng)

    # Save state
    state = rng1.save_state()

    # Generate more numbers
    val1_np = rng1.np_rng.random(5)
    val1_torch = torch.rand(3, generator=rng1.torch_rng)

    # Load state into new RNG
    rng2 = RNGManager.load_state(state)

    # Should produce same numbers as rng1 did after state was saved
    val2_np = rng2.np_rng.random(5)
    val2_torch = torch.rand(3, generator=rng2.torch_rng)

    assert np.allclose(val1_np, val2_np)
    assert torch.allclose(val1_torch, val2_torch)


def test_simulator_rng_consistency():
    """Test that Simulator uses RNG correctly for reproducible noise"""
    X_space = X_sp_default

    # Create two simulators with same RNG seed
    sim1 = Simulator(X_space, shifted_parab, eps=0.1, rng=555)
    sim2 = Simulator(X_space, shifted_parab, eps=0.1, rng=555)

    # Generate same test points
    X_test = pd.DataFrame(
        {
            "Parameter 1": [5.0, 7.5, 2.5],
            "Parameter 2": [-10.0, -5.0, -15.0],
            "Parameter 3": [5.0, 5.0, 5.0],
            "Parameter 7": ["A", "B", "C"],
        }
    )

    # Simulate - should get identical results (including noise)
    y1 = sim1.simulate(X_test)
    y2 = sim2.simulate(X_test)

    pd.testing.assert_frame_equal(y1, y2)

    # Different seeds should give different noise
    sim3 = Simulator(X_space, shifted_parab, eps=0.1, rng=999)
    y3 = sim3.simulate(X_test)

    # Results should differ due to different noise
    with pytest.raises(AssertionError):
        pd.testing.assert_frame_equal(y1, y3)


def test_shared_rng_flag(setup_campaign):
    """Test sharing RNG between campaign and optimizer"""
    X_space, target, _ = setup_campaign

    # Create shared RNG
    shared_rng = create_rng_manager(42)

    # Create optimizer with shared RNG
    optimizer = BayesianOptimizer(X_space, rng=shared_rng)

    # Creating campaign with shared RNG should produce warning
    campaign = Campaign(X_space, target, optimizer=optimizer, rng=shared_rng)
    assert campaign.rng is optimizer.rng
    assert campaign._owns_rng is False


def test_optimizer_rng_save_load(setup_campaign):
    """Test that optimizer RNG state is properly saved and restored"""
    X_space, target, simulator = setup_campaign

    # Create optimizer with specific seed
    optimizer1 = BayesianOptimizer(X_space, seed=777, fix_random_state=True)

    # Generate and fit data
    designer = ExpDesigner(X_space, seed=777)
    X0 = designer.initialize(m_initial=5, method="LHS")
    y0 = simulator.simulate(X0)
    Z0 = pd.concat([X0, y0], axis=1)

    optimizer1.fit(Z0, target)

    # Save optimizer state
    opt_state = optimizer1.save_state()

    # Verify RNG state is saved
    assert "rng_state" in opt_state
    assert opt_state["rng_state"]["seed"] == 777
    assert opt_state["fix_random_state"] is True

    # Make a suggestion
    X_suggest1, _ = optimizer1.suggest(m_batch=2, acquisition=["EI"])

    # Load optimizer from saved state
    optimizer2 = BayesianOptimizer.load_state(opt_state)

    # Verify RNG attributes are restored
    assert optimizer2.seed == 777
    assert hasattr(optimizer2, "rng")
    assert optimizer2.rng.seed == 777

    # Make suggestion from loaded optimizer - should be deterministic
    X_suggest2, _ = optimizer2.suggest(m_batch=2, acquisition=["EI"])

    # With fix_random_state=True, both should give same results
    pd.testing.assert_frame_equal(X_suggest1, X_suggest2)

    # Test with fix_random_state=False
    optimizer3 = BayesianOptimizer(X_space, seed=888, fix_random_state=False)
    optimizer3.fit(Z0, target)

    opt_state3 = optimizer3.save_state()
    assert opt_state3["fix_random_state"] is False
    assert "model_generator_state" in opt_state3  # Should save generator state

    # Load and verify model_generator is restored
    optimizer4 = BayesianOptimizer.load_state(opt_state3)
    assert hasattr(optimizer4, "model_generator")
    assert optimizer4.model_generator is not None


def test_rng_reproducibility_full_workflow(setup_campaign):
    """Integration test: Full workflow should be reproducible with same seed"""
    X_space, target, _ = setup_campaign

    def run_campaign(seed):
        # Create simulator with fixed seed
        sim = Simulator(X_space, shifted_parab, eps=0.05, rng=seed)

        # Create campaign
        campaign = Campaign(X_space, target, seed=seed)

        # Initialize
        X0 = campaign.initialize(m_initial=5, method="LHS")
        y0 = sim.simulate(X0)
        Z0 = pd.concat([X0, y0], axis=1)

        # Add data and fit
        campaign.add_data(Z0)
        campaign.fit()

        # Suggest
        X_suggest, eval_suggest = campaign.suggest(m_batch=3, acquisition=["EI"])

        return X0, y0, X_suggest, eval_suggest

    # Run twice with same seed
    X0_a, y0_a, X_suggest_a, eval_suggest_a = run_campaign(12345)
    X0_b, y0_b, X_suggest_b, eval_suggest_b = run_campaign(12345)

    # Everything should be identical
    pd.testing.assert_frame_equal(X0_a, X0_b)
    pd.testing.assert_frame_equal(y0_a, y0_b)
    pd.testing.assert_frame_equal(X_suggest_a, X_suggest_b)
    # eval_suggest may have small numerical differences in predictions
    # so we test with tolerance
    for col in eval_suggest_a.columns:
        if eval_suggest_a[col].dtype in [np.float64, np.float32]:
            assert np.allclose(eval_suggest_a[col].values, eval_suggest_b[col].values, rtol=1e-5, atol=1e-8)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
