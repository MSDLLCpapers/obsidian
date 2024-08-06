from obsidian import Campaign, ParamSpace, Target
from obsidian.tests.param_configs import X_sp_default
from obsidian.experiment import Simulator
from obsidian.experiment.benchmark import shifted_parab
from numpy.typing import ArrayLike
from typing import Callable
import pandas as pd
import json

DEFAULT_PATH = 'obsidian/tests/default_campaign.json'


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
        bool: True if the numbers are approximately equal within the tolerance, False otherwise.
    """
    diff = abs((x1-x2)/x1)
    return diff.max() < tol


def save_default_campaign(X_space: ParamSpace,
                          response_func: Callable):  # pragma: no cover
    
    y_name = 'Response'
    target = Target(name=y_name, f_transform='Standard', aim='max')
    campaign = Campaign(X_space, target)
    simulator = Simulator(X_space, response_func, y_name)
    X0 = campaign.designer.initialize(m_initial=20, method='LHS')
    y0 = simulator.simulate(X0)
    Z0 = pd.concat([X0, y0], axis=1)
    campaign.add_data(Z0)
    campaign.fit()
    obj_dict = campaign.save_state()
    with open(DEFAULT_PATH, 'w') as outfile:
        json.dump(obj_dict, outfile)


if __name__ == '__main__':
    save_default_campaign(X_sp_default, shifted_parab)
