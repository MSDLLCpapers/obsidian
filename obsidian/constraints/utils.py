"""Utility functions for constraint handling"""

from torch import Tensor
from obsidian.parameters import Target


def unscale_samples(samples: Tensor,
                    target: Target | list[Target]) -> Tensor:
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
    if not isinstance(target, (Target, list)):
        raise TypeError('Target must be a Target object or a list of Target objects')
    if isinstance(target, list):
        for t in target:
            if not isinstance(t, Target):
                raise TypeError('Target must be a Target object or a list of Target objects')
    
    # samples = sample_shape x batch_shape (x q) x m
    shape = samples.shape
    if isinstance(target, Target):
        target = [target]
    if shape[-1] != len(target):
        raise ValueError('Number of constraint targets must match number of output dimensions')
    for i, t in enumerate(target):
        samples[..., i] = Tensor(t.transform_f(samples[..., i].flatten().detach(),
                                               inverse=True).values).reshape(shape[:-1])
    return samples
