"""obsidian global config"""

import torch


TORCH_DTYPE = torch.double
DEFAULT_DEVICE = torch.get_default_device()
CAT_SEP = '^'  # Separator for one-hot encoded categories
