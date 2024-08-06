"""Method pointers and config for acquisition functions"""

from botorch.acquisition import qProbabilityOfImprovement, qUpperConfidenceBound, qSimpleRegret
    
from botorch.acquisition.logei import qLogExpectedImprovement, qLogNoisyExpectedImprovement
from botorch.acquisition.multi_objective.logei import qLogExpectedHypervolumeImprovement, qLogNoisyExpectedHypervolumeImprovement

from botorch.acquisition.multi_objective.parego import qLogNParEGO

from botorch.acquisition.active_learning import qNegIntegratedPosteriorVariance

from obsidian.acquisition import qMean, qSpaceFill

# Available acquisition functions and method pointers
aq_functions = {'EI': qLogExpectedImprovement,
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


# For default values, 'optional' indicates whether or not a default value (or None) 'val' can be used
aq_hp_defaults = {
    'EI': {'Xi_f': {'val':  0, 'optional': True}},
    'NEI': {},
    'PI': {'Xi_f': {'val':  0, 'optional': True}},
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
