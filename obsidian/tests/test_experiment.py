"""PyTests for obsidian.experiment"""

import json

from obsidian.tests.param_configs import test_X_space

from obsidian.experiment import ExpDesigner, Simulator
from obsidian.experiment.benchmark import rosenbrock, shifted_parab, ackley
from obsidian.exceptions import UnsupportedError
from obsidian.parameters import Param_Continuous, ParamSpace

import numpy as np
import pytest

test_methods = ["DOE_full", "DOE_res4", "CCD", "Sobol", "Random", "Custom", "LHS"]


@pytest.mark.fast
@pytest.mark.parametrize("X_space", test_X_space)
@pytest.mark.parametrize("method", test_methods)
def test_designer(X_space, method):
    designer = ExpDesigner(X_space, seed=0)
    m = 20
    # Use a seeded generator so the custom samples are deterministic across runs
    # (the global np.random state is not reproducible and made this flaky).
    custom_X = np.random.default_rng(0).uniform(0, 1, size=(m, X_space.n_dim))
    if method == "Custom":
        X0 = designer.initialize(m_initial=m, method=method, sample_custom=custom_X)
    elif method in ["DOE_full", "DOE_res4", "CCD"] and X_space.X_discrete:
        # DOE/CCD methods do not support discrete parameters and must reject them
        with pytest.raises(UnsupportedError):
            designer.initialize(m_initial=m, method=method)
        return
    else:
        X0 = designer.initialize(m_initial=m, method=method)

    if method == "LHS":
        X0 = designer.initialize(m_initial=X_space.n_dim * 10, method=method)
        for param in X_space:
            if param in X_space.X_discrete:
                assert np.all([level in X0[param.name].unique() for level in param.categories])
    designer.__repr__()


test_functions = [shifted_parab, rosenbrock, ackley]


@pytest.fixture(params=test_X_space)
def X_s(request):
    return request.param


@pytest.fixture
def X0(X_s):
    designer = ExpDesigner(X_s, seed=0)
    X0 = designer.initialize(m_initial=20, method="LHS")
    return X0


@pytest.mark.fast
@pytest.mark.parametrize("function", test_functions)
def test_simulator(function, X0, X_s):
    simulator = Simulator(X_s, function)
    simulator.simulate(X0)
    simulator.__repr__()


@pytest.mark.fast
def test_exp_designer_save_load():
    """ExpDesigner.save_state() / .load_state() must preserve X_space and seed.

    Tests:
    - Saved state is JSON-safe
    - Standalone load reconstructs X_space and seed from embedded payload
    - Caller-supplied overrides take precedence over embedded values
    - Non-default seed survives save and load
    """
    X_space = ParamSpace([
        Param_Continuous("A", 0, 1),
        Param_Continuous("B", -5, 5),
    ])
    designer = ExpDesigner(X_space, seed=17)

    # Save and verify JSON-safe
    state = designer.save_state()
    json.dumps(state)  # must not raise
    assert state["name"] == "ExpDesigner"
    assert state["seed"] == 17

    # Standalone load reconstructs everything
    designer2 = ExpDesigner.load_state(state)
    assert designer2.seed == 17
    assert designer2.X_space.n_dim == X_space.n_dim
    assert list(designer2.X_space.X_names) == ["A", "B"]

    # Overrides take precedence
    other_space = ParamSpace([Param_Continuous("Z", 0, 1)])
    designer3 = ExpDesigner.load_state(state, X_space=other_space, seed=99)
    assert designer3.X_space is other_space
    assert designer3.seed == 99


if __name__ == "__main__":
    pytest.main([__file__, "-m", "fast"])
