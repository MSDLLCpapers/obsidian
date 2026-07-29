"""Shared pytest fixtures for obsidian tests"""

import gc
import os

# Limit PyTorch threads only when using pytest-xdist (when -n flag is used)
# xdist sets PYTEST_XDIST_WORKER env var for each worker process
# With xdist: limit threads per worker to avoid N_workers × N_threads explosion
# Without xdist: allow full thread usage for faster single-process execution
if "PYTEST_XDIST_WORKER" in os.environ:
    num_threads = 2
    os.environ.setdefault("OMP_NUM_THREADS", str(num_threads))
    os.environ.setdefault("MKL_NUM_THREADS", str(num_threads))
    os.environ.setdefault("OPENBLAS_NUM_THREADS", str(num_threads))

# Force a non-interactive Matplotlib backend for the whole test suite so that
# plotting tests don't fail on headless CI runners (no Tk/Tcl display available).
import matplotlib
matplotlib.use("Agg")

import pytest
import torch

import obsidian
from obsidian.config import TORCH_DTYPE

# Apply torch thread limits only when running with xdist
if "PYTEST_XDIST_WORKER" in os.environ:
    torch.set_num_threads(2)
    torch.set_num_interop_threads(2)


@pytest.fixture(autouse=True)
def set_default_dtype():
    torch.set_default_dtype(TORCH_DTYPE)
    yield


@pytest.fixture(autouse=True)
def fast_model_fitting():
    """
    Override MULTI_STARTS to False for faster testing.

    In production, MULTI_STARTS=True runs all max_attempts restarts and picks the best model.
    For tests, this adds significant overhead without testing additional functionality.
    """
    from obsidian.surrogates import botorch

    original = botorch.MULTI_STARTS
    botorch.MULTI_STARTS = False
    yield
    botorch.MULTI_STARTS = original


@pytest.fixture(params=[False, True], ids=["new_rng", "old_rng"])
def rng_mode(request):
    """
    Fixture to test both RNG control modes.

    Parametrizes tests to run with both:
    - new_rng: USE_OLD_RNG_CONTROL = False (default, RNGManager-based)
    - old_rng: USE_OLD_RNG_CONTROL = True (legacy, direct seeding)

    Automatically resets to default after each test.
    """
    original = obsidian.USE_OLD_RNG_CONTROL
    obsidian.USE_OLD_RNG_CONTROL = request.param
    yield request.param
    obsidian.USE_OLD_RNG_CONTROL = original


@pytest.fixture(autouse=True)
def cleanup_memory():
    """
    Clean up memory after each test to prevent accumulation.

    This fixture runs after every test and forces garbage collection
    to free unreferenced PyTorch tensors, BoTorch models, and other objects.
    Helps prevent OOM issues on CI runners with limited memory.
    """
    yield
    gc.collect()

    # Aggressive cleanup only on CI runners due to limited memory
    if os.environ.get("GITHUB_ACTIONS") == "true":
        gc.collect()
        # Force malloc to release memory back to OS (Linux-specific)
        try:
            import ctypes

            ctypes.CDLL("libc.so.6").malloc_trim(0)
        except Exception:
            # Not available on all platforms
            pass


def pytest_addoption(parser):
    """Register optional numerical baseline workflows."""
    parser.addoption(
        "--capture-baseline",
        action="store",
        default=None,
        help="Directory path to store numerical baselines captured during tests.",
    )
    parser.addoption(
        "--compare-baseline",
        action="store",
        default=None,
        help="Directory path containing previously captured numerical baselines.",
    )
    # Default tolerance rationale: results are float64 (TORCH_DTYPE = torch.double)
    # and round-trip losslessly through CSV, so the comparison is precision-limited
    # only by computation, not storage. Measured cross-version drift (Python
    # 3.10 vs 3.12, identical pinned deps) is at machine-epsilon level (<=1e-14).
    # 1e-9 sits ~5 orders of magnitude above that noise floor (so it won't flake on
    # last-bit reduction-order differences) while staying sensitive enough to catch
    # any real algorithmic regression, which shifts results by far more than 1e-9.
    parser.addoption(
        "--baseline-rtol",
        action="store",
        type=float,
        default=1e-9,
        help="Relative tolerance for baseline comparison (default: 1e-9).",
    )
    parser.addoption(
        "--baseline-atol",
        action="store",
        type=float,
        default=1e-9,
        help="Absolute tolerance for baseline comparison (default: 1e-9).",
    )


@pytest.fixture(autouse=True)
def numerical_baseline_hook(request, monkeypatch):
    """Optionally capture or compare numerical results across test runs.

    Intercepts key numerical operations to ensure cross-version reproducibility:
    - RNG operations (design initialization, simulation)
    - Optimization (Campaign.suggest, which internally drives Optimizer.suggest,
      and AdvExpDesigner.optimize_design)
    - Design generation (ExpDesigner, AdvExpDesigner)

    Comparison tolerances are configurable via --baseline-rtol / --baseline-atol
    (both default to 1e-9; see pytest_addoption for the rationale).

    Enabled only when --capture-baseline or --compare-baseline is provided.
    """
    import hashlib
    from pathlib import Path
    import numpy as np
    import pandas as pd

    capture_dir = request.config.getoption("--capture-baseline")
    compare_dir = request.config.getoption("--compare-baseline")
    rtol = request.config.getoption("--baseline-rtol")
    atol = request.config.getoption("--baseline-atol")

    if not capture_dir and not compare_dir:
        yield
        return

    # Counter for multiple calls of same method in one test
    call_counters = {}

    def _extract_dataframe(result):
        """Extract DataFrame from various return types."""
        if isinstance(result, tuple) and len(result) >= 1:
            result = result[0]
        if isinstance(result, pd.DataFrame):
            return result.copy()
        return pd.DataFrame(result)

    def _make_identifier(*parts):
        """Create identifier from method name and key parameters."""
        # Filter out None and empty strings
        clean_parts = [str(p) for p in parts if p is not None and p != ""]
        return "_".join(clean_parts)

    def _baseline_path(base_dir, nodeid, method_id, call_num):
        """Generate unique path for each intercepted call."""
        slug = (
            nodeid.replace(os.sep, "_")
            .replace("/", "_")
            .replace("::", "__")
            .replace("[", "_")
            .replace("]", "_")
        )
        digest = hashlib.sha1(nodeid.encode("utf-8")).hexdigest()[:8]
        filename = f"{slug}__{digest}__{method_id}_{call_num}.csv"
        return Path(base_dir) / filename

    def _capture_or_compare(result_df, nodeid, method_id):
        """Capture or compare a DataFrame result."""
        # Track call number for this method in this test
        key = f"{nodeid}::{method_id}"
        call_num = call_counters.get(key, 0)
        call_counters[key] = call_num + 1

        if capture_dir:
            out_path = _baseline_path(capture_dir, nodeid, method_id, call_num)
            out_path.parent.mkdir(parents=True, exist_ok=True)
            tmp_path = out_path.with_suffix(".tmp")
            result_df.to_csv(tmp_path, index=False)
            os.replace(tmp_path, out_path)

        if compare_dir:
            in_path = _baseline_path(compare_dir, nodeid, method_id, call_num)
            if not in_path.exists():
                # A missing baseline means the set/order of captured calls diverged
                # between the reference and compare runs. That is itself a
                # reproducibility failure, so fail hard rather than skipping (a skip
                # would let the comparison report success despite the divergence).
                pytest.fail(f"No baseline for {method_id} (call {call_num}) in {nodeid}: {in_path}")

            baseline_df = pd.read_csv(in_path)

            # Numeric columns: compare with (configurable) tolerance. Verify the
            # numeric column set/order matches first, so an added/removed/reordered
            # column produces a clear message instead of a misaligned value compare.
            actual_num_df = result_df.select_dtypes(include=[np.number])
            baseline_num_df = baseline_df.select_dtypes(include=[np.number])
            if not actual_num_df.columns.equals(baseline_num_df.columns):
                raise AssertionError(
                    f"Numeric column mismatch in {method_id} (call {call_num}) for {nodeid}: "
                    f"{list(actual_num_df.columns)} != {list(baseline_num_df.columns)}"
                )
            np.testing.assert_allclose(
                actual_num_df.to_numpy(),
                baseline_num_df.to_numpy(),
                atol=atol,
                rtol=rtol,
                err_msg=f"Numerical drift in {method_id} (call {call_num}) for {nodeid}",
            )

            # Non-numeric columns (categoricals/strings, common in designs):
            # require exact equality so drift there is not silently ignored.
            actual_other = result_df.select_dtypes(exclude=[np.number]).astype(str)
            baseline_other = baseline_df.select_dtypes(exclude=[np.number]).astype(str)
            if not actual_other.columns.equals(baseline_other.columns):
                raise AssertionError(
                    f"Non-numeric column mismatch in {method_id} (call {call_num}) for {nodeid}: "
                    f"{list(actual_other.columns)} != {list(baseline_other.columns)}"
                )
            np.testing.assert_array_equal(
                actual_other.to_numpy(),
                baseline_other.to_numpy(),
                err_msg=f"Categorical drift in {method_id} (call {call_num}) for {nodeid}",
            )

    # === Intercept Campaign.suggest (Bayesian optimization) ===
    from obsidian.campaign import Campaign
    original_campaign_suggest = Campaign.suggest

    def patched_campaign_suggest(self, **optim_kwargs):
        result = original_campaign_suggest(self, **optim_kwargs)
        m_batch = optim_kwargs.get('m_batch', 1)
        acquisition = optim_kwargs.get('acquisition')
        acq_str = str(acquisition[0]) if isinstance(acquisition, list) and acquisition else str(acquisition)
        method_id = _make_identifier("Campaign.suggest", acq_str, f"m{m_batch}")

        # When the optimizer is fit, suggest() returns (X, eval); the eval frame
        # carries acquisition values / predictions, so capture it separately
        # (under its own identifier) rather than dropping it.
        _capture_or_compare(_extract_dataframe(result), request.node.nodeid, method_id)
        if isinstance(result, tuple) and len(result) >= 2 and isinstance(result[1], pd.DataFrame):
            _capture_or_compare(result[1].copy(), request.node.nodeid, method_id + "_eval")
        return result

    # === Intercept ExpDesigner.initialize (RNG: basic design generation) ===
    from obsidian.experiment import ExpDesigner
    original_designer_initialize = ExpDesigner.initialize

    def patched_designer_initialize(self, m_initial=None, method="LHS", **kwargs):
        result = original_designer_initialize(self, m_initial=m_initial, method=method, **kwargs)
        # Unseeded designers (self.seed is None) are intentionally random and
        # produce different output on every run, so they can't serve as a stable
        # baseline -- skip them rather than recording non-reproducible noise.
        if getattr(self, "seed", None) is not None:
            m_str = f"m{m_initial}" if m_initial is not None else "default"
            method_id = _make_identifier("ExpDesigner.initialize", method, m_str)
            _capture_or_compare(result, request.node.nodeid, method_id)
        return result

    # === Intercept AdvExpDesigner.initialize (RNG: advanced design generation) ===
    from obsidian.experiment.advanced_design import AdvExpDesigner
    original_adv_initialize = AdvExpDesigner.initialize

    def patched_adv_initialize(self, m_initial=None, method="LHS", **kwargs):
        result = original_adv_initialize(self, m_initial=m_initial, method=method, **kwargs)
        # Skip unseeded designers (see patched_designer_initialize).
        if getattr(self, "seed", None) is not None:
            m_str = f"m{m_initial}" if m_initial is not None else "default"
            method_id = _make_identifier("AdvExpDesigner.initialize", method, m_str)
            _capture_or_compare(result, request.node.nodeid, method_id)
        return result

    # === Intercept AdvExpDesigner.optimize_design (Design optimization) ===
    original_optimize_design = AdvExpDesigner.optimize_design

    def patched_optimize_design(self, n_trials=10, n_samples=20, metrics_to_optimize=None, **kwargs):
        result = original_optimize_design(self, n_trials=n_trials, n_samples=n_samples,
                                          metrics_to_optimize=metrics_to_optimize, **kwargs)
        # Skip unseeded designers: initialize(method='Optimized') derives a random
        # seed_start from self.seed when it is None, so the captured design would be
        # non-reproducible (see patched_designer_initialize).
        if getattr(self, "seed", None) is not None:
            # Result is (best_design, metrics_df)
            best_design = result[0] if isinstance(result, tuple) else result
            metrics_str = "_".join(metrics_to_optimize[:2]) if metrics_to_optimize else "default"
            method_id = _make_identifier("AdvExpDesigner.optimize_design", metrics_str, f"t{n_trials}")
            _capture_or_compare(best_design, request.node.nodeid, method_id)
        return result

    # === Intercept Simulator.simulate (RNG: response simulation) ===
    from obsidian.experiment import Simulator
    original_simulate = Simulator.simulate

    def patched_simulate(self, X_prop):
        result = original_simulate(self, X_prop)
        method_id = _make_identifier("Simulator.simulate", f"n{len(X_prop)}")
        _capture_or_compare(result, request.node.nodeid, method_id)
        return result

    # Apply all patches
    monkeypatch.setattr(Campaign, "suggest", patched_campaign_suggest)
    monkeypatch.setattr(ExpDesigner, "initialize", patched_designer_initialize)
    monkeypatch.setattr(AdvExpDesigner, "initialize", patched_adv_initialize)
    monkeypatch.setattr(AdvExpDesigner, "optimize_design", patched_optimize_design)
    monkeypatch.setattr(Simulator, "simulate", patched_simulate)

    yield
