"""PyTests for RNGManager core functionality

There are other tests that are related to RNG behavior (e.g. reproducibility of acquisition function suggestions), but
this file focuses on testing the core RNGManager class. Refer to other test files for more integrated tests of RNG
behavior within the campaign workflow.
"""

import os
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch

import obsidian
from obsidian import create_rng_manager
from obsidian.campaign import Campaign
from obsidian.experiment import Simulator
from obsidian.experiment.benchmark import shifted_parab
from obsidian.parameters import Target
from obsidian.rng import RNGManager, derive_seed, validate_seed
from obsidian.tests.param_configs import X_sp_default


@pytest.fixture
def setup_campaign():
    """Set up a basic campaign for testing"""
    X_space = X_sp_default
    target = Target(name="Response", f_transform="Standard", aim="max")
    simulator = Simulator(X_space, shifted_parab, eps=0.05, rng=114514)

    return X_space, target, simulator


def test_create_new_rng():
    """Test RNGManager creation with various inputs"""
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


def test_rng_state_save_load():
    """Test RNGManager state save and load"""
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


def test_old_rng_control_mode(setup_campaign):
    """Test backward compatibility: obsidian.USE_OLD_RNG_CONTROL flag"""
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


# Fully independent subprocess test to ensure reproducibility
def test_rng_subprocess_isolation():
    """Test that campaigns in separate processes produce identical results (no global state pollution)"""
    worker_script = Path(__file__).parent / "_rng_subprocess_worker.py"
    seed = 77777
    n_cycles = 3

    with tempfile.TemporaryDirectory() as tmpdir:
        output1 = os.path.join(tmpdir, "run1.csv")
        output2 = os.path.join(tmpdir, "run2.csv")

        # Run twice in separate subprocesses
        subprocess.run(
            [sys.executable, str(worker_script), str(seed), str(n_cycles), output1], check=True, capture_output=True
        )
        subprocess.run(
            [sys.executable, str(worker_script), str(seed), str(n_cycles), output2], check=True, capture_output=True
        )

        # Load and compare results
        data1 = pd.read_csv(output1)
        data2 = pd.read_csv(output2)

        pd.testing.assert_frame_equal(data1, data2)


def test_derive_seed():
    """derive_seed is deterministic in (base, key), stays in range, and is a no-op for key=None."""
    # key=None returns the base seed unchanged (modulo range)
    assert derive_seed(12345, None) == 12345
    assert derive_seed(2**31, None) == 2**31 % (2**31 - 1)

    # deterministic: same (base, key) -> same result
    assert derive_seed(42, 5) == derive_seed(42, 5)
    # sensitive to the key
    assert derive_seed(42, 5) != derive_seed(42, 6)
    # sensitive to the base seed
    assert derive_seed(42, 5) != derive_seed(43, 5)

    # always a valid 32-bit-range seed
    for base, key in [(42, 5), (0, "abc"), (2**31 - 2, 999999)]:
        s = derive_seed(base, key)
        assert 0 <= s < 2**31 - 1


def test_validate_seed():
    """validate_seed passes valid/None seeds and rejects out-of-range or non-integer input."""
    # None (auto) and in-range integers pass through unchanged
    assert validate_seed(None) is None
    assert validate_seed(0) == 0
    assert validate_seed(2**32 - 1) == 2**32 - 1
    assert validate_seed(np.int64(123)) == 123

    # out of numpy's [0, 2**32) range -> ValueError naming the argument
    for bad in [-1, 2**32, 2**33]:
        with pytest.raises(ValueError, match="manual_seed"):
            validate_seed(bad)

    # non-integer (incl. bool, which is a subclass of int) -> TypeError
    for bad in [3.5, True, "5"]:
        with pytest.raises(TypeError, match="manual_seed"):
            validate_seed(bad)

    # custom argument name surfaces in the message
    with pytest.raises(ValueError, match="my_seed"):
        validate_seed(-1, name="my_seed")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
