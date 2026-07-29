"""obsidian: Automated experiment design and black-box optimization"""

__version__ = "1.0.0"

# Import key objects
from obsidian.campaign import Campaign
from obsidian.optimizer import BayesianOptimizer
from obsidian.surrogates import SurrogateBoTorch
from obsidian.parameters import ParamSpace, Target
from obsidian.rng import create_rng_manager, USE_OLD_RNG_CONTROL

# Ensure that other subpackages are imported properly for documentation
from obsidian.objectives import Objective
from obsidian.experiment import ExpDesigner
from obsidian.experiment import AdvExpDesigner
import obsidian.constraints as constraints
import obsidian.exceptions as exceptions
import obsidian.acquisition as acquisition
import obsidian.plotting as plotting
import obsidian.surrogates as surrogates
import obsidian.rng as rng
