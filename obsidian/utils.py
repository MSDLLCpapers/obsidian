"""Utility functions"""

from enum import Enum

import torch
from torch import Tensor

from .parameters import Target


def unscale_samples(samples: Tensor, target: list[Target]) -> Tensor:
    """
    Unscale the scaled samples based on the given target(s).

    Args:
        samples (Tensor): The scaled samples to be unscaled.
        target (Target | list[Target]): The target(s) used for unscaling.
            It can be a single Target object or a list of Target objects.

    Returns:
        Tensor: The unscaled samples.

    Raises:
        TypeError: If the target is not a Target object or a list of Target objects.
        ValueError: If the number of constraint targets does not match the number of output dimensions.

    """

    # samples = sample_shape x batch_shape (x q) x m
    shape = samples.shape
    if shape[-1] != len(target):
        raise ValueError("Number of constraint targets must match number of output dimensions")
    for i, t in enumerate(target):
        unscaled = t.transform_f(samples[..., i].flatten().detach(), inverse=True).values
        samples[..., i] = samples.new_tensor(unscaled).reshape(shape[:-1])
    return samples


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
            dict[param] = "inf"
        elif dict[param] == -torch.inf:
            dict[param] = "-inf"
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
        if dict[param] == "inf":
            state_dict[param] = torch.tensor(torch.inf)
        elif dict[param] == "-inf":
            state_dict[param] = torch.tensor(-torch.inf)
        else:
            state_dict[param] = torch.tensor(dict[param])
    return state_dict


class TaskType(Enum):
    """Task types for specifying an optimization or characterization task."""

    OPTIMIZATION = "optimization"
    CHARACTERIZATION = "characterization"

    @classmethod
    def allowed_tasks(cls) -> str:
        """Return a string listing all allowed task types."""
        return ", ".join(f"{t.value!r} (TaskType.{t.name})" for t in cls)

    @classmethod
    def from_value(cls, task: "TaskType | str") -> "TaskType":
        """Convert a string representation to a TaskType enum.

        Args:
            task: The string or TaskType to convert.

        Returns:
            TaskType: The corresponding TaskType enum.

        Raises:
            ValueError: If the task string does not match any TaskType.
        """
        if isinstance(task, str):
            try:
                return TaskType[task.upper()]
            except KeyError:
                raise ValueError(f"Invalid task type: {task!r}. Must be one of {cls.allowed_tasks()}")
        elif isinstance(task, cls):
            return task
        else:
            raise TypeError(f"task must be a string or TaskType, got {type(task).__name__}")
