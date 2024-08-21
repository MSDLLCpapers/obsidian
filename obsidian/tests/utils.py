"""Utility functions for PyTest unit testing"""

from obsidian.tests.param_configs import X_sp_default, X_sp_cont_ndims

from obsidian import Campaign, ParamSpace, Target
from obsidian.experiment import Simulator
from obsidian.experiment.benchmark import shifted_parab, two_leaves
from obsidian.objectives import Identity_Objective

from numpy.typing import ArrayLike
from typing import Callable

import pandas as pd
import numpy as np
import json

# Default campaigns for testing without having to re-run optimization
DEFAULT_MOO_PATH = 'obsidian/tests/default_campaign_MOO.json'
DEFAULT_SOO_PATH = 'obsidian/tests/default_campaign_SOO.json'


def equal_state_dicts(e1: dict | str | float | int,
                      e2: dict | str | float | int) -> bool:
    """
    Recursively compare two dictionaries, and allow for floating-point error
    """
    # If the values are equal, skip
    if not e1 == e2:
        # First, make sure we are comparing the same type
        assert type(e1) == type(e2), f'Type mismatch at {e1} != {e2}'
        
        # If they are dictionaries, compare elements recursively
        if isinstance(e1, dict):
            assert e1.keys() == e2.keys(), f'Keys mismatch at {e1.keys()} != {e2.keys()}'
            for k, v in e1.items():
                equal_state_dicts(v, e2[k])
                
        # If they are lists, compare elements recursively
        elif isinstance(e1, list):
            assert len(e1) == len(e2), f'Length mismatch at {len(e1)} != {len(e2)}'
            for e1_i, e2_i in zip(e1, e2):
                equal_state_dicts(e1_i, e2_i)
                
        # Otherwise, if it is numerical, check for floating-point error
        elif isinstance(e1, (float, int)):
            if not (np.isnan(e1) and np.isnan(e2)):
                assert (e1-e2)/e1 < 1e-6, f'{e1} != {e2}'
        else:
            raise ValueError(f'{e1} != {e2}')
        
    return True


def approx_equal(x1: ArrayLike | float | int,
                 x2: ArrayLike | float | int,
                 tol: float = 1e-6):
    """
    Check if two numbers or arrays are approximately equal within a given tolerance.

    Args:
        x1 (ArrayLike or numeric): The first number.
        x2 (ArrayLike or numeric): The second number.
        tol (float, optional): The tolerance value. Defaults to 1e-6.

    Returns:
        bool: True if the numbers are approximately equal within the tolerance,
            False otherwise.
    """
    diff = abs((x1-x2)/x1)
    return diff.max() < tol


def save_default_campaign(X_space: ParamSpace,
                          response_func: Callable,
                          path: str,
                          m_initial: int = 6,
                          n_response: int = 1):  # pragma: no cover
    
    y_name = 'Response'
    y_names = [y_name+' '+str(i+1) for i in range(n_response)]
    target = [Target(name=n, f_transform='Standard', aim='max') for n in y_names]
    campaign = Campaign(X_space, target)
    simulator = Simulator(X_space, response_func, y_names)
    
    X0 = campaign.designer.initialize(m_initial=m_initial, method='LHS')
    y0 = simulator.simulate(X0)
    Z0 = pd.concat([X0, y0], axis=1)
    campaign.add_data(Z0)

    for i in range(3):
        print(f'Simulating iteration {i+1}')
        campaign.fit()
        
        X_suggest, eval_suggest = campaign.optimizer.suggest(
            m_batch=2, objective=Identity_Objective(mo=n_response > 1))
        y_i = simulator.simulate(X_suggest)
        Z_i = pd.concat([X_suggest, y_i, eval_suggest], axis=1)
        campaign.add_data(Z_i)
    
    obj_dict = campaign.save_state()
    with open(path, 'w') as outfile:
        json.dump(obj_dict, outfile)


if __name__ == '__main__':
    
    save_default_campaign(X_sp_default, shifted_parab, DEFAULT_SOO_PATH, n_response=1)
    save_default_campaign(X_sp_cont_ndims[2], two_leaves, DEFAULT_MOO_PATH, n_response=2)
