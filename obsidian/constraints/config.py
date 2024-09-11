"""Method pointers and config for constraints"""

from .input import (
    Linear_Constraint,
    BatchVariance_Constraint
)

# from .output import (
# )

const_class_dict = {'Linear_Constraint': Linear_Constraint,
                    'BatchVariance_Constraint': BatchVariance_Constraint
                    }
