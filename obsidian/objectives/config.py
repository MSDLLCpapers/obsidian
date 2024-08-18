"""Method pointers and config for objective functions"""

from .custom import (
    Identity_Objective,
    Feature_Objective,
    Utopian_Distance,
    Bounded_Target,
    Index_Objective
)

from .scalarize import (
    Scalar_WeightedSum,
    Scalar_WeightedNorm,
    Scalar_Chebyshev,
)

obj_class_dict = {'Identity_Objective': Identity_Objective,
                  'Feature_Objective': Feature_Objective,
                  'Utopian_Distance': Utopian_Distance,
                  'Bounded_Target': Bounded_Target,
                  'Scalar_WeightedSum': Scalar_WeightedSum,
                  'Scalar_WeightedNorm': Scalar_WeightedNorm,
                  'Scalar_Chebyshev': Scalar_Chebyshev,
                  'Index_Objective': Index_Objective}
