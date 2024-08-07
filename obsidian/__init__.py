"""obsidian: Automated experiment design and black-box optimization"""
__version__ = '0.7.11'

from obsidian.campaign import Campaign
from obsidian.optimizer import BayesianOptimizer
from obsidian.surrogates import SurrogateBoTorch
from obsidian.parameters import ParamSpace, Target
