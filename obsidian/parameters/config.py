"""Method pointers and config for parameters"""

from .continuous import (
    Param_Continuous,
    Param_Observational
)
from .discrete import (
    Param_Categorical,
    Param_Ordinal,
    Param_Discrete_Numeric,
    Task,
)

param_class_dict = {'Param_Continuous': Param_Continuous,
                    'Param_Categorical': Param_Categorical,
                    'Param_Ordinal': Param_Ordinal,
                    'Param_Discrete_Numeric': Param_Discrete_Numeric,
                    'Param_Observational': Param_Observational,
                    'Task': Task}
