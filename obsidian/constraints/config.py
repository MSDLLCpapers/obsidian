"""Method pointers and config for constraints"""

from .input import (
    Linear_Constraint,
    BatchVariance_Constraint
)

from .output import (
    Blank_Constraint,
    L1_Constraint,
    ThresholdConstraint,
    UpperBoundConstraint,
    LowerBoundConstraint,
    InRangeConstraint,
)

const_class_dict = {'Linear_Constraint': Linear_Constraint,
                    'BatchVariance_Constraint': BatchVariance_Constraint,
                    'Blank_Constraint': Blank_Constraint,
                    'L1_Constraint': L1_Constraint,
                    'ThresholdConstraint': ThresholdConstraint,
                    'UpperBoundConstraint': UpperBoundConstraint,
                    'LowerBoundConstraint': LowerBoundConstraint,
                    'InRangeConstraint': InRangeConstraint,
                    }
