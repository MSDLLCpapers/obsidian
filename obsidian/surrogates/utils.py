"""Utility functions for surrogate model handling"""

import torch
from torch import Tensor


def tensordict_to_dict(state_dict: dict[Tensor]) -> dict[float | int | str]:
    """
    Converts a dictionary of tensors to a dictionary of numpy arrays or string representations.

    Args:
        state_dict (dict): A dictionary containing tensors.

    Returns:
        dict: A dictionary with the same keys as `state_dict`, but with values converted to numpy arrays or string representations.

    """
    dict = {}
    for param in state_dict:
        # Convert to data first, otherwise boolean comparisons will not work
        dict[param] = state_dict[param].cpu().data.numpy().tolist()
        if dict[param] == torch.inf:
            dict[param] = 'inf'
        elif dict[param] == -torch.inf:
            dict[param] = '-inf'
    return dict


def dict_to_tensordict(dict: dict[float | int | str]) -> dict[Tensor]:
    """
    Converts a dictionary of values to a dictionary of tensors.

    Args:
        dict (dict): The input dictionary.

    Returns:
        dict: A dictionary where the values are converted to tensors.
    """
    state_dict = {}
    # Make sure all parameters are tensors
    for param in dict:
        if dict[param] == 'inf':
            state_dict[param] = torch.tensor(torch.inf)
        elif dict[param] == '-inf':
            state_dict[param] = torch.tensor(-torch.inf)
        else:
            state_dict[param] = torch.tensor(dict[param])
    return state_dict
