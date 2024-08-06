from obsidian.parameters import Param_Continuous, Param_Ordinal, Param_Categorical, \
    Param_Observational, Param_Discrete_Numeric, ParamSpace

params = [
        Param_Continuous('Parameter 1', 0, 10),
        Param_Continuous('Parameter 2', -20, 0),
        Param_Continuous('Parameter 3', 5, 5),
        Param_Observational('Parameter 4', 0, 24),
        Param_Discrete_Numeric('Parameter 5', [-2, -1, 1, 2]),
        Param_Discrete_Numeric('Parameter 6', [7]),
        Param_Categorical('Parameter 7', ['A', 'B', 'C', 'D']),
        Param_Categorical('Parameter 8', ['E', 'F', 'G', 'H']),
        Param_Categorical('Parameter 9', ['I']),
        Param_Ordinal('Parameter 10', ['J', 'K', 'L', 'M']),
        Param_Ordinal('Parameter 11', ['N'])
    ]

default = [params[i] for i in [0, 1, 3, 6]]  # 2 continuous, 1 observational, 1 categorical
cont_small = [params[i] for i in [0, 1, 2]]  # continuous including edge cases
numeric = [params[i] for i in [0, 1, 2, 3, 4, 5]]  # numeric including edge cases

cat_small = [params[i] for i in [6, 7, 8]]  # categorical including edge cases
disc_small = [params[i] for i in [6, 9]]  # 1 categorical, 1 ordinal
disc_large = [params[i] for i in [6, 7, 8, 9, 10]]  # discrete including edge cases

params_cont_large = [
        Param_Continuous('Parameter 1', 0, 10),
        Param_Continuous('Parameter 2', 0, 10),
        Param_Continuous('Parameter 3', 0, 10),
        Param_Continuous('Parameter 4', 0, 10),
        Param_Continuous('Parameter 5', 0, 10),
        Param_Continuous('Parameter 6', 0, 10),
        Param_Continuous('Parameter 7', 0, 10),
        Param_Continuous('Parameter 8', 0, 10),
        Param_Continuous('Parameter 9', 0, 10),
        Param_Continuous('Parameter 10', 0, 10),
        Param_Continuous('Parameter 11', 0, 10),
        Param_Continuous('Parameter 12', 0, 10),
]

X_sp_default = ParamSpace(params=default)
X_sp_cont_small = ParamSpace(params=cont_small)
X_sp_cont_large = ParamSpace(params=params_cont_large)
X_sp_numeric = ParamSpace(params=numeric)
X_sp_cat_small = ParamSpace(params=cat_small)
X_sp_disc_small = ParamSpace(params=disc_small)
X_sp_disc_large = ParamSpace(params=disc_large)

test_X_space = [X_sp_default, X_sp_cont_small, X_sp_numeric, X_sp_cat_small, X_sp_disc_small, X_sp_disc_large]

X_sp_cont_ndims = [ParamSpace(params_cont_large[:i]) for i in range(len(params_cont_large))]
