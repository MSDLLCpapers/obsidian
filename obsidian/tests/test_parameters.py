from obsidian.tests.param_configs import test_X_space
from obsidian.experiment import ExpDesigner
from obsidian.parameters import (
    Param_Continuous,
    Param_Observational,
    Param_Discrete_Numeric,
    Param_Categorical,
    Param_Ordinal,
    Param_Discrete,
    Task,
    ParamSpace
)

import numpy as np
from pandas.testing import assert_frame_equal
import pytest


@pytest.fixture(params=test_X_space)
def X_space(request):
    return request.param


@pytest.fixture
def X0(X_space):
    designer = ExpDesigner(X_space, seed=0)
    X0 = designer.initialize(m_initial=20, method='LHS')
    return X0


@pytest.mark.fast
def test_param_loading(X_space):
    X_space_dict = X_space.save_state()
    X_space_new = ParamSpace.load_state(X_space_dict)
    for param in X_space_new:
        param.__repr__()
    X_space_new.__repr__()
    assert X_space_dict == X_space_new.save_state(), 'Error during serialization of parameter space'


@pytest.mark.fast
def test_param_mapping(X0, X_space):
    X0_map = X_space.unit_map(X0)
    X0_demap = X_space.unit_demap(X0_map)
    assert_frame_equal(X0, X0_demap, check_dtype=False)
    df_mean = X_space.mean()


@pytest.mark.fast
def test_param_encoding(X0, X_space):
    X0_encode = X_space.encode(X0)
    X0_decode = X_space.decode(X0_encode)
    assert_frame_equal(X0, X0_decode, check_dtype=False)

    # Also need to make sure that if certain parameters are excluded from the encoding, the returned df contains 0s for other categories
    designer = ExpDesigner(X_space, seed=0)
    X_zeros = designer.initialize(m_initial=10, method='Custom',
                                  sample_custom=np.zeros(shape=(10, len(X_space.params))))
    X_zeros_encode = X_space.encode(X_zeros)
    assert X_zeros_encode.shape[1] == X0_encode.shape[1]


@pytest.mark.fast
def test_param_transform_mapping(X_space):
    print(X_space)
    print(X_space.t_map)
    print(X_space.tinv_map)
    assert len(X_space.t_map) == X_space.n_dim
    assert len(X_space.tinv_map) == X_space.n_tdim


test_params = [
        Param_Continuous('Parameter 1', 0, 10),
        Param_Observational('Parameter 2', 0, 10),
        Param_Discrete_Numeric('Parameter 3', [-2, -1, 1, 2]),
        Param_Categorical('Parameter 4', ['A', 'B', 'C', 'D']),
        Param_Ordinal('Parameter 5', ['A', 'B', 'C', 'D']),
        Task('Parameter 6', ['A', 'B', 'C', 'D']),
    ]

test_type = [lambda x: list(x),
             lambda x: np.array(x),
             lambda x: list(x)[0][-2]]


@pytest.mark.fast
@pytest.mark.parametrize('param, check_type', zip(test_params, test_type))
def test_param_encoding_types(param, check_type):
    cont_vals = [0, 1, 2, 3, 4]
    cat_vals = ['A', 'B', 'C', 'D']
    num_disc_vals = [-2, -1, 1, 2]

    if isinstance(param, Param_Continuous):
        d_2 = [cont_vals] * 3
    elif isinstance(param, Param_Discrete) and not isinstance(param, Param_Discrete_Numeric):
        d_2 = [cat_vals] * 3
    elif isinstance(param, Param_Discrete_Numeric):
        d_2 = [num_disc_vals] * 3

    X = check_type(d_2)
    X_u = param.unit_map(X)
    X_u_inv = param.unit_demap(X_u)

    X_t = param.encode(X)
    X_t_inv = param.decode(X_t)

    if isinstance(X, np.ndarray):
        assert np.all(X_u_inv == X)
        if not isinstance(param, Param_Categorical):  # Categorical params don't encode to the same shape
            assert np.all(X_t_inv == X)
    else:
        assert X_u_inv == X
        if not isinstance(param, Param_Categorical):  # Categorical params don't decode to the same shape
            assert X_t_inv == X
    

if __name__ == '__main__':
    pytest.main([__file__, '-m', 'fast'])
