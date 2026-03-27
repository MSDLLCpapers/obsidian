"""obsidian: Automated experiment design and black-box optimization"""

__version__ = "0.8.6"

# Set default torch dtype to double for numerical stability
import torch
from obsidian.config import TORCH_DTYPE
torch.set_default_dtype(TORCH_DTYPE)

# Import key objects
from obsidian.campaign import Campaign
from obsidian.optimizer import BayesianOptimizer
from obsidian.surrogates import SurrogateBoTorch
from obsidian.parameters import ParamSpace, Target
from obsidian.rng import create_rng_manager, USE_OLD_RNG_CONTROL

# Ensure that other subpackages are imported properly for documentation
from obsidian.objectives import Objective
from obsidian.experiment import ExpDesigner
import obsidian.constraints as constraints
import obsidian.exceptions as exceptions
import obsidian.acquisition as acquisition
import obsidian.plotting as plotting
import obsidian.rng as rng