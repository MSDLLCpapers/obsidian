"""Method pointers and config for acquisition functions"""

from .custom import qMean, qSpaceFill

from .botorch import (
    qProbabilityOfImprovement, qUpperConfidenceBound, qSimpleRegret,
    qLogExpectedImprovement, qLogNoisyExpectedImprovement,
    qLogExpectedHypervolumeImprovement, qLogNoisyExpectedHypervolumeImprovement,
    qLogNParEGO, qNegIntegratedPosteriorVariance
)


# Available acquisition functions and method pointers
aq_class_dict = {'EI': qLogExpectedImprovement,
                 'NEI': qLogNoisyExpectedImprovement,
                 'PI': qProbabilityOfImprovement,
                 'UCB': qUpperConfidenceBound,
                 'SR': qSimpleRegret,
                 'EHVI': qLogExpectedHypervolumeImprovement,
                 'NEHVI': qLogNoisyExpectedHypervolumeImprovement,
                 'NParEGO': qLogNParEGO,
                 'NIPV': qNegIntegratedPosteriorVariance,
                 'RS': None,
                 'Mean': qMean,
                 'SF': qSpaceFill}

# Acquisition functions separated by optimization modality
universal_aqs = ['RS', 'Mean', 'SF']
valid_aqs = {'single': ['EI', 'NEI', 'PI', 'UCB', 'SR', 'NIPV'] + universal_aqs,
             'multi': ['EHVI', 'NEHVI', 'NParEGO'] + universal_aqs}
aq_defaults = {'single': 'NEI', 'multi': 'NEHVI'}


# For default values, 'optional' indicates whether or not a default value (or None) 'val' can be used
aq_hp_defaults = {
    'EI': {'inflate': {'val':  0, 'optional': True}},
    'NEI': {},
    'PI': {'inflate': {'val':  0, 'optional': True}},
    'UCB': {'beta': {'val': 1, 'optional': True}},
    'SR': {},
    'RS': {},
    'Mean': {},
    'SF': {},
    'NIPV': {},
    'EHVI': {'ref_point': {'val': None, 'optional': True}},
    'NEHVI': {'ref_point': {'val': None, 'optional': True}},
    'NParEGO': {'scalarization_weights': {'val': [1], 'optional': True}},
}
