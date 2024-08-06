from obsidian.parameters import (
    Param_Continuous,
    Param_Observational,
    Param_Categorical,
    Param_Ordinal,
    Param_Discrete_Numeric,
)

import pytest

numeric_list = [1, 2, 3, 4]
number = 1
string = 'A'
string_list = ['A', 'B', 'C', 'D']


@pytest.mark.fast
def test_numeric_param_validation():
    with pytest.raises(TypeError):
        param = Param_Continuous('test', min=string, max=string)
        
    with pytest.raises(ValueError):
        param = Param_Continuous('test', min=1, max=0)
        param._validate_value(2)
        
    with pytest.raises(TypeError):
        Param_Observational('test', min=string, max=string)


@pytest.mark.fast
def test_discrete_param_validation():
    with pytest.raises(TypeError):
        param = Param_Categorical('test', categories=numeric_list)
        
    with pytest.raises(ValueError):
        param = Param_Categorical('test', categories=string_list)
        param._validate_value('2')
        
    with pytest.raises(TypeError):
        param = Param_Ordinal('test', categories=numeric_list)
        
    with pytest.raises(TypeError):
        param = Param_Discrete_Numeric('test', categories=string_list)


if __name__ == '__main__':
    pytest.main([__file__, '-m', 'not slow'])
