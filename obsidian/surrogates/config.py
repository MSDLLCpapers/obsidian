"""Method pointers and config for surrogate models"""

from .custom_GP import DKLGP, FlatGP, PriorGP
from .custom_torch import DNN

from botorch.models import SingleTaskGP, MixedSingleTaskGP, MultiTaskGP

# Surrogate model method pointer
model_class_dict = {'GP': SingleTaskGP,
                    'MixedGP': MixedSingleTaskGP,
                    'DKL': DKLGP,
                    'GPflat': FlatGP,
                    'GPprior': PriorGP,
                    'MTGP': MultiTaskGP,
                    'DNN': DNN}
