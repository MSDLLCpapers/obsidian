
from obsidian.tests.param_configs import X_sp_cont_ndims

from obsidian.parameters import Target
from obsidian.experiment import ExpDesigner, Simulator
from obsidian.experiment.benchmark import two_leaves
from obsidian.optimizer import BayesianOptimizer
from obsidian.constraints import OutConstraint_Blank, InConstraint_Generic, InConstraint_ConstantDim
from obsidian.objectives import Identity_Objective, Index_Objective, Utopian_Distance, Objective_Sequence, Bounded_Target, \
      Scalar_WeightedSum, Scalar_WeightedNorm, Scalar_Chebyshev
      
import pandas as pd
import numpy as np
import pytest


@pytest.fixture()
def X_space():
    return X_sp_cont_ndims[2]


@pytest.fixture()
def Z0(X_space):
    designer = ExpDesigner(X_space, seed=0)
    X0 = designer.initialize(m_initial=6, method='LHS')
    simulator = Simulator(X_space, two_leaves, eps=0.05)
    y0 = simulator.simulate(X0)
    Z0 = pd.concat([X0, y0], axis=1)
    return Z0


target = [
    Target(name='Response 1', f_transform='Standard', aim='max'),
    Target(name='Response 2', f_transform='Standard', aim='max')
]


@pytest.mark.parametrize('surrogate', [pytest.param('GP', marks=pytest.mark.fast), 'GPflat', pytest.param('DKL', marks=pytest.mark.slow)])
def test_optimizer_fit(X_space, surrogate, Z0, serial_test=True):
    optimizer = BayesianOptimizer(X_space, surrogate=surrogate, seed=0, verbose=0)
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
base_X_space = X_sp_cont_ndims[2]
optimizer = BayesianOptimizer(base_X_space, surrogate='GP', seed=0, verbose=0)
designer = ExpDesigner(base_X_space, seed=0)
X0 = designer.initialize(m_initial=6, method='LHS')
simulator = Simulator(base_X_space, two_leaves, eps=0.05)
y0 = simulator.simulate(X0)
Z0_base = pd.concat([X0, y0], axis=1)
optimizer.fit(Z0_base, target=target)

# Run very short and bad optimizations for testing, but test all MOO aqs
test_config = {'optim_samples': 4, 'optim_restarts': 2}


def test_fit_nan():
    Z0_sample = Z0_base.copy()
    for col in Z0_sample.columns:
        Z0_sample.loc[Z0_sample.sample(frac=0.1).index, col] = np.nan
    optimizer_nan = BayesianOptimizer(base_X_space, surrogate='GP', seed=0, verbose=0)
    optimizer_nan.fit(Z0_sample, target=target)


@pytest.mark.parametrize('m_batch', [pytest.param(1, marks=pytest.mark.fast), 3])
@pytest.mark.parametrize('fixed_var', [None, {'Parameter 1': 5}])
def test_optimizer_suggest(m_batch, fixed_var):
    X_suggest, eval_suggest = optimizer.suggest(m_batch=m_batch, fixed_var=fixed_var,
                                                acquisition=['NEHVI', 'SF'], **test_config)
    df_suggest = pd.concat([X_suggest, eval_suggest], axis=1)


test_aqs = ['NEHVI', {'NEHVI': {'ref_point': [0.1, 0.1]}},
            'EHVI', {'EHVI': {'ref_point': [0.1, 0.1]}},
            {'NParEGO': {'scalarization_weights': [5, 1]}},
            'Mean', 'SF', 'RS']


@pytest.mark.parametrize('aq', test_aqs)
def test_optimizer_aqs(aq):
    X_suggest, eval_suggest = optimizer.suggest(m_batch=2,  acquisition=[aq], **test_config)
    df_suggest = pd.concat([X_suggest, eval_suggest], axis=1)
   

utopian = Utopian_Distance(utopian=[10, 10], targets=target)

test_scalars = [Scalar_WeightedSum(weights=[0.5, 0.5]),
                Objective_Sequence([utopian, Scalar_WeightedSum(weights=[0.5, 0.5])]),
                Scalar_WeightedNorm(weights=[0.5, 0.5]),
                Objective_Sequence([utopian, Scalar_WeightedNorm(weights=[0.5, 0.5], neg=True)]),
                Scalar_Chebyshev(weights=[0.5, 0.5]),
                Scalar_Chebyshev(weights=[0.5, 0.5], augment=False),
                Objective_Sequence([utopian, Scalar_Chebyshev(weights=[0.5, 0.5])])]


@pytest.mark.parametrize('scalar', test_scalars)
def test_optimizer_scalar(scalar):
    X_suggest, eval_suggest = optimizer.suggest(m_batch=2, objective=scalar, **test_config)
    df_suggest = pd.concat([X_suggest, eval_suggest], axis=1)


@pytest.mark.fast
def test_optimizer_maximize():
    X_suggest, eval_suggest = optimizer.maximize(**test_config)
    df_suggest = pd.concat([X_suggest, eval_suggest], axis=1)


test_ineq = [[InConstraint_Generic(base_X_space, indices=[0, 1], coeff=[1, 1], rhs=5)]]
test_nleq = [[InConstraint_ConstantDim(base_X_space, dim=0, tol=0.1)]]
test_out = [[OutConstraint_Blank(target)]]


@pytest.mark.parametrize('ineq_constraints', test_ineq)
def test_ineq_constraints(ineq_constraints):
    X_suggest, eval_suggest = optimizer.suggest(ineq_constraints=ineq_constraints,
                                                **test_config)
    df_suggest = pd.concat([X_suggest, eval_suggest], axis=1)
    
    
@pytest.mark.parametrize('nleq_constraints', test_nleq)
def test_nleq_constraints(nleq_constraints):
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
                                                nleq_constraints=test_nleq[0],
                                                out_constraints=test_out[0],
                                                **test_config)
    df_suggest = pd.concat([X_suggest, eval_suggest], axis=1)


@pytest.mark.parametrize('m_batch', [pytest.param(1, marks=pytest.mark.fast), 2, pytest.param(5, marks=pytest.mark.slow)])
@pytest.mark.parametrize('obj', [Identity_Objective(mo=True),
                                 Scalar_WeightedSum(weights=[0.5, 0.5]),
                                 Utopian_Distance([10, 10], target),
                                 Bounded_Target([(0.8, 1.0), None], target),
                                 Bounded_Target([(0.8, 1.0), (0.8, 1.0)], target),
                                 Objective_Sequence([Identity_Objective(mo=True), Index_Objective()])])
def test_objective(m_batch, obj):
    X_suggest, eval_suggest = optimizer.suggest(m_batch=m_batch, objective=obj,  **test_config)
    df_suggest = pd.concat([X_suggest, eval_suggest], axis=1)


if __name__ == '__main__':
    pytest.main([__file__, '-m', 'not slow'])
