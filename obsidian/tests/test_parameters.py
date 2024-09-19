"""PyTests for obsidian.parameters"""

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
    ParamSpace,
    Target,
    Standard_Scaler,
    Logit_Scaler
)

from obsidian.exceptions import UnsupportedError, UnfitError
from obsidian.tests.utils import equal_state_dicts

import numpy as np
import pandas as pd
from pandas.testing import assert_frame_equal
import torch
import pytest


# Iterate over several preset parameter spaces
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
    obj_dict = X_space.save_state()
    X_space2 = ParamSpace.load_state(obj_dict)
    for param in X_space2:
        param.__repr__()
    X_space2.__repr__()
    obj_dict2 = X_space2.save_state()
    assert equal_state_dicts(obj_dict, obj_dict2), 'Error during serialization'


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
    assert len(X_space.t_map) == X_space.n_dim
    assert len(X_space.tinv_map) == X_space.n_tdim


# Set up a sampling of parameter types for testing encoding/decoding
test_params = [
        Param_Continuous('Parameter 1', 0, 10),
        Param_Observational('Parameter 2', 0, 10),
        Param_Discrete_Numeric('Parameter 3', [-2, -1, 1, 2]),
        Param_Categorical('Parameter 4', ['A', 'B', 'C', 'D']),
        Param_Ordinal('Parameter 5', 'A, B, C, D'),
        Task('Parameter 6', ['A', 'B', 'C', 'D']),
    ]

# Set up a numbe of different data types for testing encoding/decoding
test_type = [lambda x: list(x),
             lambda x: np.array(x),
             lambda x: list(x)[0][-2],  # Single value
             ]


@pytest.mark.fast
@pytest.mark.parametrize('param, type_i', zip(test_params, test_type))
def test_param_encoding_types(param, type_i):
    
    # Set up a variety of value types to test
    cont_vals = [0, 1, 2, 3, 4]
    cat_vals = ['A', 'B', 'C', 'D']
    num_disc_vals = [-2, -1, 1, 2]

    # Make sure that 2D arrays also work!
    if isinstance(param, Param_Continuous):
        d_2 = [cont_vals] * 3
    elif isinstance(param, Param_Discrete) and not isinstance(param, Param_Discrete_Numeric):
        d_2 = [cat_vals] * 3
    elif isinstance(param, Param_Discrete_Numeric):
        d_2 = [num_disc_vals] * 3

    X = type_i(d_2)
    
    # Unit map and demap
    X_u = param.unit_map(X)
    X_u_inv = param.unit_demap(X_u)

    # Encode and decode
    X_t = param.encode(X)
    X_t_inv = param.decode(X_t)

    # Check equivalence based on type
    if isinstance(X, np.ndarray):
        assert np.all(X_u_inv == X)
        if not isinstance(param, Param_Categorical):  # Categorical params don't encode to the same shape
            assert np.all(X_t_inv == X)
    else:
        assert X_u_inv == X
        if not isinstance(param, Param_Categorical):  # Categorical params don't decode to the same shape
            assert X_t_inv == X


# VALIDATION TESTS - Force errors to be raised in object usage

numeric_list = [1, 2, 3, 4]
number = 1
string = 'A'
string_list = ['A', 'B', 'C', 'D']


@pytest.mark.fast
def test_param_indexing():
    X_space = ParamSpace(test_params)
    p0 = X_space[0]
    p0 = X_space['Parameter 1']


@pytest.mark.fast
def test_numeric_param_validation():
    # Strings for numeric
    with pytest.raises(TypeError):
        param = Param_Continuous('test', min=string, max=string)
        
    # Value outside of range
    with pytest.raises(ValueError):
        param = Param_Continuous('test', min=1, max=0)
        param._validate_value(2)
        
    # Strings for numeric
    with pytest.raises(TypeError):
        Param_Observational('test', min=string, max=string)


@pytest.mark.fast
def test_discrete_param_validation():
    # Numeric for categorical
    with pytest.raises(TypeError):
        param = Param_Categorical('test', categories=numeric_list)

    # Too long of a string
    with pytest.raises(ValueError):
        param = Param_Categorical('test', categories=string_list
                                  + ['test'*64])

    # Value not in categories
    with pytest.raises(ValueError):
        param = Param_Categorical('test', categories=string_list)
        param._validate_value('E')
        
    # Numeric for ordinal
    with pytest.raises(TypeError):
        param = Param_Ordinal('test', categories=numeric_list)
        
    # Strings for discrete numeric
    with pytest.raises(TypeError):
        param = Param_Discrete_Numeric('test', categories=string_list)
    
    # Value outside of range
    with pytest.raises(ValueError):
        param = Param_Discrete_Numeric('test', categories=numeric_list)
        param._validate_value(5)


@pytest.mark.fast
def test_paramspace_validation():
    # Overlapping namespace
    with pytest.raises(ValueError):
        X_space = ParamSpace([test_params[0], test_params[0]])
    
    # Misuse of categorical separator
    cat_sep_param = Param_Continuous('Parameter^1', 0, 10)
    with pytest.raises(ValueError):
        X_space = ParamSpace([test_params[0], cat_sep_param])
    
    # >1 Task
    with pytest.raises(UnsupportedError):
        X_space = ParamSpace([Task('Parameter X', ['A', 'B', 'C', 'D']),
                              Task('Parameter Y', ['A', 'B', 'C', 'D'])])

    test_data = pd.DataFrame(np.random.uniform(0, 1, (10, 2)), columns=['Parameter X', 'Parameter Z'])
    X_space = ParamSpace([Param_Continuous('Parameter X', min=0, max=1),
                          Param_Continuous('Parameter Y', min=0, max=1)])
    
    # Missing X names
    with pytest.raises(KeyError):
        test_encoded = X_space.encode(test_data)
        
        
@pytest.mark.fast
def test_target_validation():
    
    # Invalid aim
    with pytest.raises(ValueError):
        Target('Response1', aim='maximize')
    
    # Invalid f_transform
    with pytest.raises(KeyError):
        Target('Response1', f_transform='quadratic')
    
    test_response = torch.rand(10)
    
    # Transform before fit
    with pytest.raises(UnfitError):
        Target('Response1').transform_f(test_response)
    
    # Transform non-arraylike
    with pytest.raises(TypeError):
        Target('Response1').transform_f('ABC')
        
    # Transform non-numerical arraylike
    with pytest.raises(TypeError):
        Target('Response1').transform_f(['A', 'B', 'C'])
    
    # Transform before fit
    with pytest.raises(UnfitError):
        transform_func = Standard_Scaler()
        transform_func.forward(test_response)
    
    test_neg_response = -0.5 - torch.rand(10)
    
    # Values outside of logit range, refitting
    with pytest.warns(UserWarning):
        transform_func = Logit_Scaler(range_response=100)
        transform_func.forward(test_neg_response, fit=False)
        
        
if __name__ == '__main__':
    pytest.main([__file__, '-m', 'fast'])
