"""Method pointers and config for surrogate models"""

from botorch.models import SingleTaskGP, MixedSingleTaskGP, MultiTaskGP
from .GP_custom import DKLGP, FlatGP, PriorGP
from .torch_custom import DNN

# Surrogate model method pointer
model_dict = {'GP': SingleTaskGP,
              'MixedGP': MixedSingleTaskGP,
              'DKL': DKLGP,
              'GPflat': FlatGP,
              'GPprior': PriorGP,
              'MTGP': MultiTaskGP,
              'DNN': DNN}
