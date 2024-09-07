"""PyTests for obsidian.optimizer under single-output usage"""

from obsidian.tests.param_configs import X_sp_default, X_sp_cont_small, X_sp_cat_small

from obsidian.parameters import Target
from obsidian.experiment import ExpDesigner, Simulator
from obsidian.optimizer import BayesianOptimizer
from obsidian.experiment.benchmark import shifted_parab
from obsidian.tests.utils import approx_equal, equal_state_dicts

import pandas as pd
import numpy as np
import pytest


# Test a variety of preset parameter spaces
@pytest.fixture(params=[X_sp_default, X_sp_cont_small, X_sp_cat_small])
def X_space(request):
    return request.param


@pytest.fixture()
def Z0(X_space):
    designer = ExpDesigner(X_space, seed=1)
    X0 = designer.initialize(m_initial=len(X_space)*2, method='LHS')
    simulator = Simulator(X_space, shifted_parab, eps=0.05)
    y0 = simulator.simulate(X0)
    Z0 = pd.concat([X0, y0], axis=1)
    return Z0


@pytest.mark.fast
@pytest.mark.parametrize('f_transform', ['Standard', 'Logit_MinMax', 'Logit_Percentage', 'Identity'])
def test_f_transform(X_space, Z0, f_transform):
    # Test all of the transforms with fitting at min/max
    optimizer = BayesianOptimizer(X_space, surrogate='GP', seed=0, verbose=0)
    target = Target(name='Response', f_transform=f_transform, aim='max')
    target.__repr__()
    optimizer.fit(Z0, target=target)
    
    # Verify equivalence of f and f_inv
    y_train = optimizer.y_train
    f_train = target.transform_f(y_train)
    f_train_inv = target.transform_f(f_train, inverse=True)
    assert approx_equal(y_train.values.flatten(), f_train_inv.values.flatten())
    
    target = Target(name='Response', f_transform=f_transform, aim='min')
    optimizer.fit(Z0, target=target)

    # Verify equivalence of f and f_inv
    y_train = optimizer.y_train
    f_train = target.transform_f(y_train)
    f_train_inv = target.transform_f(f_train, inverse=True)
    assert approx_equal(y_train.values.flatten(), f_train_inv.values.flatten())


target = Target(name='Response', f_transform='Standard', aim='max')


@pytest.mark.parametrize('surrogate', [pytest.param('GP', marks=pytest.mark.fast),
                                       'GPflat',
                                       'GPprior',
                                       pytest.param('DKL', marks=pytest.mark.slow),
                                       'DNN'])
def test_optimizer_fit(X_space, surrogate, Z0, serial_test=True):
    optimizer = BayesianOptimizer(X_space, surrogate=surrogate, seed=0, verbose=0)
    
    if surrogate == 'GPflat' and not X_space.X_cont:
        # GPflat will fail will a purely categorical space because the design matrix is not p.d.
        return
    
    tol = 1e-2 if surrogate == 'DNN' else 1e-5
    
    optimizer.fit(Z0, target=target)
    if serial_test:
        obj_dict = optimizer.save_state()
        optimizer_2 = BayesianOptimizer.load_state(obj_dict)
        obj_dict2 = optimizer_2.save_state()
        assert equal_state_dicts(obj_dict, obj_dict2)
        optimizer_2.__repr__()
        y_pred = optimizer.predict(optimizer.X_train)
        y_pred_2 = optimizer_2.predict(optimizer.X_train)
        y_error = ((y_pred_2-y_pred)/y_pred.max(axis=0)).values
        assert abs(y_error).max() < tol, 'Prediction error in loading parameters of saved optimizer'


# Generate a baseline optimizer to use for future tests
base_X_space = X_sp_default
optimizer = BayesianOptimizer(base_X_space, surrogate='GP', seed=0, verbose=0)
designer = ExpDesigner(base_X_space, seed=0)
X0 = designer.initialize(m_initial=6, method='LHS')
simulator = Simulator(base_X_space, shifted_parab, eps=0.05)
y0 = simulator.simulate(X0)
Z0_base = pd.concat([X0, y0], axis=1)
optimizer.fit(Z0_base, target=target)

# Run very short optimizations for testing
test_config = {'optim_samples': 2, 'optim_restarts': 2}


def test_fit_nan():
    Z0_sample = Z0_base.copy()
    for col in Z0_sample.columns:
        Z0_sample.loc[Z0_sample.sample(frac=0.1).index, col] = np.nan
    optimizer_nan = BayesianOptimizer(base_X_space, surrogate='GP', seed=0, verbose=0)
    optimizer_nan.fit(Z0_sample, target=target)


@pytest.mark.fast
def test_optimizer_pending():
    X_suggest, eval_suggest = optimizer.suggest(m_batch=2, **test_config)
    X_suggest, eval_suggest = optimizer.suggest(m_batch=1, **test_config, X_pending=X_suggest)
    X_suggest, eval_suggest = optimizer.suggest(m_batch=1, **test_config, X_pending=X_suggest, eval_pending=eval_suggest)


@pytest.mark.parametrize('m_batch', [pytest.param(1, marks=pytest.mark.fast), 3])
@pytest.mark.parametrize('fixed_var', [None, {'Parameter 1': 5}])
def test_optimizer_suggest(m_batch, fixed_var):
    X_suggest, eval_suggest = optimizer.suggest(m_batch=m_batch, fixed_var=fixed_var,
                                                acquisition=['EI', 'SF'], **test_config)
    df_suggest = pd.concat([X_suggest, eval_suggest], axis=1)


def test_suggest_searchspace():
    optimizer.X_space[0].set_search(2, 8)
    optimizer.X_space[3].set_search(['A', 'C'])
    
    X_suggest, eval_suggest = optimizer.suggest(m_batch=2, **test_config)
    df_suggest = pd.concat([X_suggest, eval_suggest], axis=1)
    
    optimizer.X_space.open_search()


test_aqs = ['NEI',
            'EI',
            {'EI': {'inflate': 0.05}},
            {'EI': {'inflate': -0.05}},
            'PI',
            {'PI': {'inflate': 0.05}},
            {'PI': {'inflate': -0.05}},
            'UCB',
            {'UCB': {'beta': 2}},
            {'UCB': {'beta': 0}},
            'NIPV',
            'SR',
            'Mean',
            'SF',
            ]


@pytest.mark.parametrize('aq', test_aqs)
def test_optimizer_aqs(aq):
    X_suggest, eval_suggest = optimizer.suggest(m_batch=2, acquisition=[aq], **test_config)
    df_suggest = pd.concat([X_suggest, eval_suggest], axis=1)


@pytest.mark.fast
def test_optimizer_maximize():
    X_suggest, eval_suggest = optimizer.maximize(**test_config)
    df_suggest = pd.concat([X_suggest, eval_suggest], axis=1)


if __name__ == '__main__':
    pytest.main([__file__, '-m', 'not slow'])
