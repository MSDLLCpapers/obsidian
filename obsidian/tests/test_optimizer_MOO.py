"""PyTests for obsidian.optimizer under multi-output usage"""

from obsidian.tests.param_configs import X_sp_cont_ndims

from obsidian.parameters import Target
from obsidian.experiment import ExpDesigner, Simulator
from obsidian.optimizer import BayesianOptimizer
from obsidian.experiment.benchmark import two_leaves
from obsidian.tests.utils import equal_state_dicts
from botorch.models.ensemble import EnsembleModel

from numpy.testing import assert_array_equal
import pandas as pd
import numpy as np
import pytest
import warnings


@pytest.fixture()
def X_space():
    return X_sp_cont_ndims[2]


@pytest.fixture()
def Z0(X_space):
    designer = ExpDesigner(X_space, seed=0)
    X0 = designer.initialize(m_initial=len(X_space)*2, method='LHS')
    simulator = Simulator(X_space, two_leaves, eps=0.05, rng=0)
    y0 = simulator.simulate(X0)
    Z0 = pd.concat([X0, y0], axis=1)
    return Z0


target = [
    Target(name='Response 1', f_transform='Standard', aim='max'),
    Target(name='Response 2', f_transform='Standard', aim='max')
]


@pytest.mark.parametrize('surrogate', [pytest.param('GP', marks=pytest.mark.fast),
                                       'GPflat',
                                       'GPprior',
                                       pytest.param('DKL', marks=pytest.mark.slow),
                                       'DNN'])
def test_optimizer_fit(X_space, surrogate, Z0, serial_test=True):
    optimizer = BayesianOptimizer(X_space, surrogate=surrogate, seed=0, verbose=3)
    
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
        y_error = ((y_pred_2-y_pred)/y_pred.max(axis=0))

        # Only check the mean for ensemble models, as the lb/ub are not deterministic
        if isinstance(optimizer.surrogate[0].torch_model, EnsembleModel):
            check_cols = [col for col in y_pred if '(pred)' in col]
            y_error = y_error[check_cols]
        
        assert abs(y_error.values).max() < tol, 'Prediction error in loading parameters of saved optimizer'


# Generate a baseline optimizer to use for future tests
base_X_space = X_sp_cont_ndims[2]
optimizer = BayesianOptimizer(base_X_space, surrogate='GP', seed=0, verbose=0)
designer = ExpDesigner(base_X_space, seed=0)
X0 = designer.initialize(m_initial=6, method='LHS')
simulator = Simulator(base_X_space, two_leaves, eps=0.05, rng=0)
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
                                                acquisition=['NEHVI', 'SF'], **test_config)
    df_suggest = pd.concat([X_suggest, eval_suggest], axis=1)


def test_suggest_searchspace():
    optimizer.X_space[0].set_search(2, 8)
    
    X_suggest, eval_suggest = optimizer.suggest(m_batch=2, **test_config)
    df_suggest = pd.concat([X_suggest, eval_suggest], axis=1)
    
    optimizer.X_space.open_search()
    

test_aqs = ['NEHVI',
            {'NEHVI': {'ref_point': [0.1, 0.1]}},
            'EHVI',
            {'EHVI': {'ref_point': [0.1, 0.1]}},
            'NParEGO',
            {'NParEGO': {'scalarization_weights': [5, 1]}},
            'Mean',
            'SF',
            'RS',
            ]


@pytest.mark.parametrize('aq', test_aqs)
def test_optimizer_aqs(aq):
    X_suggest, eval_suggest = optimizer.suggest(m_batch=2,  acquisition=[aq], **test_config)
    df_suggest = pd.concat([X_suggest, eval_suggest], axis=1)
   

@pytest.mark.fast
def test_optimizer_maximize():
    X_suggest, eval_suggest = optimizer.maximize(**test_config)
    df_suggest = pd.concat([X_suggest, eval_suggest], axis=1)


@pytest.mark.fast
def test_suggest_ignores_tracking_targets_by_default():
    optimizer_tracking = BayesianOptimizer(base_X_space, surrogate='GP', seed=0, verbose=0)
    tracking_targets = [
        Target(name='Response 1', f_transform='Standard', aim='max'),
        Target(name='Response 2', f_transform='Standard', aim='max', tracking_only=True),
    ]
    optimizer_tracking.fit(Z0_base, target=tracking_targets)

    _, eval_suggest = optimizer_tracking.suggest(m_batch=1, **test_config)
    assert eval_suggest['aq Method'].iloc[0] == 'NEI'


@pytest.mark.fast
def test_suggest_warns_tracking_only_target_when_explicitly_requested():
    optimizer_tracking = BayesianOptimizer(base_X_space, surrogate='GP', seed=0, verbose=0)
    tracking_targets = [
        Target(name='Response 1', f_transform='Standard', aim='max'),
        Target(name='Response 2', f_transform='Standard', aim='max', tracking_only=True),
    ]
    optimizer_tracking.fit(Z0_base, target=tracking_targets)

    with pytest.warns(UserWarning, match='tracking-only'):
        _, eval_suggest = optimizer_tracking.suggest(
            m_batch=1,
            target=Target(name='Response 2', f_transform='Standard', aim='max', tracking_only=True),
            **test_config,
        )
    assert eval_suggest['aq Method'].iloc[0] == 'NEI'


@pytest.mark.fast
def test_suggest_rejects_unknown_target_when_target_provided():
    optimizer_tracking = BayesianOptimizer(base_X_space, surrogate='GP', seed=0, verbose=0)
    tracking_targets = [
        Target(name='Response 1', f_transform='Standard', aim='max'),
        Target(name='Response 2', f_transform='Standard', aim='max', tracking_only=True),
    ]
    optimizer_tracking.fit(Z0_base, target=tracking_targets)

    with pytest.raises(NameError):
        optimizer_tracking.suggest(
            m_batch=1,
            target=Target(name='Response 3', f_transform='Standard', aim='max'),
            **test_config,
        )


def test_maximize_includes_tracking_only_without_warning():
    optimizer_tracking = BayesianOptimizer(base_X_space, surrogate='GP', seed=0, verbose=0)
    tracking_targets = [
        Target(name='Response 1', f_transform='Standard', aim='max'),
        Target(name='Response 2', f_transform='Standard', aim='max', tracking_only=True),
    ]
    optimizer_tracking.fit(Z0_base, target=tracking_targets)

    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter('always')
        X_all, _ = optimizer_tracking.maximize(**test_config)
    assert X_all.shape[0] == 2
    assert not any('tracking-only' in str(w_i.message).lower() for w_i in w)


@pytest.mark.fast
def test_save_load_preserves_tracking_only_targets():
    optimizer_tracking = BayesianOptimizer(base_X_space, surrogate='GP', seed=0, verbose=0)
    tracking_targets = [
        Target(name='Response 1', f_transform='Standard', aim='max'),
        Target(name='Response 2', f_transform='Standard', aim='max', tracking_only=True),
    ]
    optimizer_tracking.fit(Z0_base, target=tracking_targets)

    state = optimizer_tracking.save_state()
    optimizer_loaded = BayesianOptimizer.load_state(state)

    assert optimizer_loaded.target[1].tracking_only == True

    X_all, _ = optimizer_loaded.maximize(**test_config)
    assert X_all.shape[0] == 2


@pytest.mark.fast
def test_save_load_unfitted_optimizer(X_space):
    """Test that unfitted optimizers can be saved and loaded"""
    # Create unfitted optimizer
    opt1 = BayesianOptimizer(X_space, surrogate='GP', seed=42, verbose=0)

    # Save state before fitting
    state = opt1.save_state()

    # Verify state structure
    assert 'X_space' in state
    assert 'surrogate_spec' in state
    assert 'is_fitted' in state
    assert state['is_fitted'] == False
    assert 'target' not in state  # Should not be present for unfitted
    assert 'model_states' not in state  # Should not be present for unfitted

    # Load the unfitted optimizer
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        opt2 = BayesianOptimizer.load_state(state)
        # Verify warning was issued
        assert len(w) == 1
        assert issubclass(w[0].category, UserWarning)
        assert "unfitted" in str(w[0].message).lower()

    # Verify properties
    assert opt2.is_fit == False
    assert not hasattr(opt2, 'target')
    assert not hasattr(opt2, 'surrogate')
    assert opt2.surrogate_type == opt1.surrogate_type
    assert opt2.surrogate_hps == opt1.surrogate_hps
    assert opt2.seed == opt1.seed

    # Verify it can still be fitted
    designer = ExpDesigner(X_space, seed=1)
    X0 = designer.initialize(m_initial=len(X_space)*2, method='LHS')
    simulator = Simulator(X_space, two_leaves, eps=0.05, rng=1)
    y0 = simulator.simulate(X0)
    Z0 = pd.concat([X0, y0], axis=1)
    targets = [
        Target(name='Response 1', f_transform='Standard', aim='max'),
        Target(name='Response 2', f_transform='Standard', aim='max')
    ]
    opt2.fit(Z0, target=targets)
    assert opt2.is_fit == True

    # Save and load again after fitting
    state2 = opt2.save_state()
    assert state2['is_fitted'] == True
    assert 'target' in state2
    assert 'model_states' in state2


@pytest.mark.fast
def test_load_legacy_fitted_state(X_space, Z0):
    """Test that old saved states (without is_fitted flag) still load correctly"""
    # Create and fit optimizer
    opt1 = BayesianOptimizer(X_space, surrogate='GP', seed=42, verbose=0)
    targets = [
        Target(name='Response 1', f_transform='Standard', aim='max'),
        Target(name='Response 2', f_transform='Standard', aim='max')
    ]
    opt1.fit(Z0, target=targets)

    # Save state
    state = opt1.save_state()

    # Simulate legacy state by removing is_fitted flag
    if 'is_fitted' in state:
        del state['is_fitted']

    # Should still load correctly
    opt2 = BayesianOptimizer.load_state(state)
    assert opt2.is_fit == True  # Inferred from presence of model_states

    # Predictions should match
    y_pred = opt1.predict(opt1.X_train)
    y_pred_2 = opt2.predict(opt2.X_train)
    assert_array_equal(y_pred.values, y_pred_2.values)


@pytest.mark.fast
def test_unfitted_to_fitted_cycle(X_space, Z0):
    """Test complete cycle: create -> save -> load -> fit -> save -> load"""
    targets = [
        Target(name='Response 1', f_transform='Standard', aim='max'),
        Target(name='Response 2', f_transform='Standard', aim='max')
    ]

    # 1. Create unfitted optimizer
    opt1 = BayesianOptimizer(X_space, surrogate='GP', seed=42, verbose=0)

    # 2. Save unfitted state
    state_unfitted = opt1.save_state()
    assert state_unfitted['is_fitted'] == False

    # 3. Load unfitted state
    with warnings.catch_warnings(record=True):
        warnings.simplefilter("always")
        opt2 = BayesianOptimizer.load_state(state_unfitted)
    assert opt2.is_fit == False

    # 4. Fit the loaded optimizer
    opt2.fit(Z0, target=targets)
    assert opt2.is_fit == True

    # 5. Save fitted state
    state_fitted = opt2.save_state()
    assert state_fitted['is_fitted'] == True

    # 6. Load fitted state
    opt3 = BayesianOptimizer.load_state(state_fitted)
    assert opt3.is_fit == True

    # 7. Verify predictions match
    y_pred_2 = opt2.predict(opt2.X_train)
    y_pred_3 = opt3.predict(opt3.X_train)
    assert_array_equal(y_pred_2.values, y_pred_3.values)


@pytest.mark.fast
def test_unfitted_state_round_trip(X_space):
    """Test that unfitted optimizer state round-trips correctly"""
    opt1 = BayesianOptimizer(X_space, surrogate='GP', seed=42, verbose=0)

    # Save and load twice
    state1 = opt1.save_state()
    with warnings.catch_warnings(record=True):
        warnings.simplefilter("always")
        opt2 = BayesianOptimizer.load_state(state1)
    state2 = opt2.save_state()

    # States should be identical
    assert equal_state_dicts(state1, state2)


@pytest.mark.fast
def test_unfitted_optimizer_operations(X_space):
    """Test that unfitted optimizer properly raises errors for operations requiring fit"""
    from obsidian.exceptions import UnfitError

    opt = BayesianOptimizer(X_space, surrogate='GP', seed=42, verbose=0)

    # Save should work
    state = opt.save_state()
    assert state['is_fitted'] == False

    # Create a dummy DataFrame for predictions
    designer = ExpDesigner(X_space, seed=1)
    X_test = designer.initialize(m_initial=1, method='LHS')

    # These operations should fail on unfitted optimizer
    with pytest.raises(UnfitError):
        opt.predict(X_test)

    with pytest.raises(UnfitError):
        opt.suggest(m_batch=1)

    with pytest.raises(UnfitError):
        opt.evaluate(X_test)


@pytest.mark.fast
def test_save_load_warnings(X_space, Z0, capsys):
    """Test that warnings are issued correctly"""
    # Test 1: Loading unfitted state issues UserWarning
    opt_unfitted = BayesianOptimizer(X_space, surrogate='GP', seed=42, verbose=0)
    state_unfitted = opt_unfitted.save_state()

    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        BayesianOptimizer.load_state(state_unfitted)
        assert len(w) == 1
        assert issubclass(w[0].category, UserWarning)
        assert "unfitted" in str(w[0].message).lower()

    # Test 2: Saving unfitted optimizer prints info message (when verbose)
    opt_verbose = BayesianOptimizer(X_space, surrogate='GP', seed=42, verbose=1)
    opt_verbose.save_state()
    captured = capsys.readouterr()
    assert "unfitted" in captured.out.lower() or "configuration" in captured.out.lower()

    # Test 3: Loading legacy state prints info message
    targets = [
        Target(name='Response 1', f_transform='Standard', aim='max'),
        Target(name='Response 2', f_transform='Standard', aim='max')
    ]
    opt_fitted = BayesianOptimizer(X_space, surrogate='GP', seed=42, verbose=0)
    opt_fitted.fit(Z0, target=targets)
    state_legacy = opt_fitted.save_state()
    del state_legacy['is_fitted']  # Make it look like legacy

    BayesianOptimizer.load_state(state_legacy)
    captured = capsys.readouterr()
    assert "older version" in captured.out.lower() or "missing" in captured.out.lower()


if __name__ == '__main__':
    pytest.main([__file__, '-m', 'not slow'])
