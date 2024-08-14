"""PyTests for obsidian.experiment"""

from obsidian.tests.param_configs import test_X_space

from obsidian.experiment import ExpDesigner, Simulator
from obsidian.experiment.benchmark import rosenbrock, shifted_parab, ackley

import numpy as np
import pytest

test_methods = ['DOE_full', 'DOE_res4', 'Sobol', 'Random', 'Custom', 'LHS']


@pytest.mark.fast
@pytest.mark.parametrize('X_space', test_X_space)
@pytest.mark.parametrize('method', test_methods)
def test_designer(X_space, method):
    designer = ExpDesigner(X_space, seed=0)
    m = 20
    custom_X = np.random.uniform(0, 1, size=(m, X_space.n_dim))
    if method == 'Custom':
        X0 = designer.initialize(m_initial=m, method=method, sample_custom=custom_X)
    elif method in ['DOE_full', 'DOE_res4'] and X_space.X_discrete:
        pass
    else:
        X0 = designer.initialize(m_initial=m, method=method)
        
    if method == 'LHS':
        X0 = designer.initialize(m_initial=X_space.n_dim*10, method=method)
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
    X0 = designer.initialize(m_initial=20, method='LHS')
    return X0


@pytest.mark.fast
@pytest.mark.parametrize('function', test_functions)
def test_simulator(function, X0, X_s):
    simulator = Simulator(X_s, function)
    simulator.simulate(X0)
    simulator.__repr__()


if __name__ == '__main__':
    pytest.main([__file__, '-m', 'fast'])
