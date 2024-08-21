"""obsidian: Automated experiment design and black-box optimization"""
__version__ = '0.8.3'

# Import key objects
from obsidian.campaign import Campaign
from obsidian.optimizer import BayesianOptimizer
from obsidian.surrogates import SurrogateBoTorch
from obsidian.parameters import ParamSpace, Target

# Ensure that other subpackages are imported properly for documentation
from obsidian.objectives import Objective
from obsidian.experiment import ExpDesigner
import obsidian.constraints as constraints
import obsidian.exceptions as exceptions
import obsidian.acquisition as acquisition
import obsidian.plotting as plotting
