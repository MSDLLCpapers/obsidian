from obsidian.tests.param_configs import X_sp_default, X_sp_cont_small, X_sp_cat_small

from obsidian.parameters import Target
from obsidian.experiment import ExpDesigner, Simulator
from obsidian.experiment.benchmark import shifted_parab
from obsidian.optimizer import BayesianOptimizer
from obsidian.constraints import OutConstraint_Blank, InConstraint_Generic, InConstraint_ConstantDim, OutConstraint_L1
from obsidian.objectives import Identity_Objective, Feature_Objective, Objective_Sequence, Utopian_Distance, Bounded_Target
from obsidian.tests.utils import approx_equal
from obsidian.campaign.explainer import Explainer

import pandas as pd
import numpy as np
import pytest


@pytest.fixture(params=[X_sp_default, X_sp_cont_small, X_sp_cat_small])
def X_space(request):
    return request.param


@pytest.fixture()
def Z0(X_space):
    designer = ExpDesigner(X_space, seed=1)
    X0 = designer.initialize(m_initial=8, method='LHS')
    simulator = Simulator(X_space, shifted_parab, eps=0.05)
    y0 = simulator.simulate(X0)
    Z0 = pd.concat([X0, y0], axis=1)
    return Z0


@pytest.mark.fast
@pytest.mark.parametrize('f_transform', ['Standard', 'Logit_MinMax', 'Logit_Percentage', 'Identity'])
def test_f_transform(X_space, Z0, f_transform):
    optimizer = BayesianOptimizer(X_space, surrogate='GP', seed=0, verbose=0)
    target = Target(name='Response', f_transform=f_transform, aim='max')
    optimizer.fit(Z0, target=target)

    y_train = optimizer.y_train
    f_train = target.transform_f(y_train)
    f_train_inv = target.transform_f(f_train, inverse=True)
    target.__repr__()
    
    assert approx_equal(y_train.values.flatten(), f_train_inv.values.flatten())


target = Target(name='Response', f_transform='Standard', aim='max')


@pytest.mark.parametrize('surrogate', [pytest.param('GP', marks=pytest.mark.fast), 'GPflat',
                                       'GPprior', pytest.param('DKL', marks=pytest.mark.slow)])
def test_optimizer_fit(X_space, surrogate, Z0, serial_test=True):

    optimizer = BayesianOptimizer(X_space, surrogate=surrogate, seed=0, verbose=0)
    
    if surrogate == 'GPflat' and not X_space.X_cont:
        # GPflat will fail will a purely categorical space because the design matrix is not p.d.
        return
    
    optimizer.fit(Z0, target=target)
    if serial_test:
        save = optimizer.save_state()
        optimizer_2 = BayesianOptimizer.load_state(save)
        optimizer_2.__repr__()
        y_pred = optimizer.predict(optimizer.X_train)
        y_pred_2 = optimizer_2.predict(optimizer.X_train)
        y_error = ((y_pred_2-y_pred)/y_pred).values
        assert abs(y_error).max() < 1e-5, 'Prediction error in loading parameters of saved optimizer'


# Generate a baseline optimizer to use for future tests
base_X_space = X_sp_default
optimizer = BayesianOptimizer(base_X_space, surrogate='GP', seed=0, verbose=0)
designer = ExpDesigner(base_X_space, seed=0)
X0 = designer.initialize(m_initial=6, method='LHS')
simulator = Simulator(base_X_space, shifted_parab, eps=0.05)
y0 = simulator.simulate(X0)
Z0_base = pd.concat([X0, y0], axis=1)
optimizer.fit(Z0_base, target=target)

test_config = {'optim_samples': 4, 'optim_restarts': 2}


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


test_aqs = ['NEI', 'EI', {'EI': {'inflate': 0.05}}, {'EI': {'inflate': -0.05}},
            'PI', 'UCB', {'UCB': {'beta': 2}}, {'UCB': {'beta': 0}},
            'NIPV',
            'SF', 'RS', 'Mean', 'SR']


@pytest.mark.parametrize('aq', test_aqs)
def test_optimizer_aqs_SOO(aq):
    X_suggest, eval_suggest = optimizer.suggest(m_batch=2, acquisition=[aq], **test_config)
    df_suggest = pd.concat([X_suggest, eval_suggest], axis=1)


@pytest.mark.fast
def test_optimizer_maximize():
    X_suggest, eval_suggest = optimizer.maximize(**test_config)
    df_suggest = pd.concat([X_suggest, eval_suggest], axis=1)


test_ineq = [[InConstraint_Generic(base_X_space, indices=[0, 1], coeff=[1, 1], rhs=5)]]
test_nleq = [[InConstraint_ConstantDim(base_X_space, dim=0, tol=0.1)]]
test_out = [[OutConstraint_Blank(target)], [OutConstraint_L1(target, offset=1)]]


@pytest.mark.parametrize('ineq_constraints', test_ineq)
def test_ineq_constraints(ineq_constraints):
    X_suggest, eval_suggest = optimizer.suggest(ineq_constraints=ineq_constraints,
                                                **test_config)
    df_suggest = pd.concat([X_suggest, eval_suggest], axis=1)
    
    
@pytest.mark.parametrize('nleq_constraints', test_nleq)
def test_nleq_constraints(nleq_constraints):
    with pytest.raises(Exception):
        X_suggest, eval_suggest = optimizer.suggest(nleq_constraints=nleq_constraints,
                                                    **test_config)
        df_suggest = pd.concat([X_suggest, eval_suggest], axis=1)
    
    
@pytest.mark.parametrize('out_constraints', test_out)
def test_out_constraints(out_constraints):
    X_suggest, eval_suggest = optimizer.suggest(out_constraints=out_constraints,
                                                **test_config)
    df_suggest = pd.concat([X_suggest, eval_suggest], axis=1)


@pytest.mark.slow
def test_combo_constraints():
    X_suggest, eval_suggest = optimizer.suggest(ineq_constraints=test_ineq[0],
                                                nleq_constraints=None,
                                                out_constraints=test_out[0],
                                                **test_config)
    df_suggest = pd.concat([X_suggest, eval_suggest], axis=1)


@pytest.mark.parametrize('m_batch', [pytest.param(1, marks=pytest.mark.fast), 2, pytest.param(5, marks=pytest.mark.slow)])
@pytest.mark.parametrize('obj', [Identity_Objective(),
                                 Feature_Objective(X_space=base_X_space, indices=[0, 1], coeff=[1, 1]),
                                 Utopian_Distance([10], target),
                                 Bounded_Target([(0.8, 1.0)], target),
                                 Objective_Sequence([Identity_Objective(), Identity_Objective()])])
def test_objective(m_batch, obj):
    X_suggest, eval_suggest = optimizer.suggest(m_batch=m_batch, objective=obj, **test_config)
    df_suggest = pd.concat([X_suggest, eval_suggest], axis=1)


@pytest.mark.fast
def test_explain():
    model_exp = Explainer(optimizer)
    model_exp.shap_explain(n=10)
    df_sens = model_exp.cal_sensitivity(dx=0.01)


if __name__ == '__main__':
    pytest.main([__file__, '-m', 'not slow'])
