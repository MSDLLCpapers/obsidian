import random
import time
from contextlib import contextmanager
from functools import partial

import numpy as np
import torch


class GlobalRNG:
    """
    A class to manage global random number generation for reproducibility and statistical integrity.
    """

    def __init__(self, seed: int | None = None):
        # if seed is none, get a seed from current time and save it
        if seed is None:
            # time.time() * 1e6 gives microseconds level accuracy
            seed = int(time.time() * 1e6) % (2**32 - 1)
        self.seed = seed
        torch.use_deterministic_algorithms(True)
        self.np_rng = np.random.default_rng(seed)
        self.torch_rng = torch.Generator().manual_seed(seed)
        print(
            f"numpy and torch random number generators initialized with seed {seed}. "
            "Please reuse this seed to reproduce the results."
        )

    def tmp_seed_override(self, func):
        """Decorator that samples a seed from the global RNG and temporarily overrides the global random states (numpy, torch, random) for the duration of the decorated function call."""
        seed: int = get_new_seed(1, self.torch_rng)  # type: ignore

        def wrapper(*args, **kwargs):
            with with_tmp_seed(seed):
                return func(*args, **kwargs)

        return wrapper

    def save_state(self) -> dict:
        """Saves the objective and RNG states to a state dictionary"""
        obj_dict = {
            "name": self.__class__.__name__,
            "seed": self.seed,
            # numpy bit generator state is nested dictionary with only strings and integers
            "np_rng": self.np_rng.bit_generator.state,
            # torch generator state is a tensor so convert it to list for serialization
            "torch_rng": self.torch_rng.get_state().cpu().tolist(),
        }
        return obj_dict

    @classmethod
    def load_state(cls, obj_dict: dict):
        """Loads the RNG states from a state dictionary"""
        instance = cls(seed=obj_dict["seed"])
        instance.np_rng.bit_generator.state = obj_dict["np_rng"]
        # must be a ByteTensor
        instance.torch_rng.set_state(torch.ByteTensor(obj_dict["torch_rng"]))
        print(
            "numpy and torch random number generators, and their random states "
            f"restored with seed {instance.seed}. "
        )
        return instance


USE_OLD_RNG_CONTROL = False

_GLOBAL_RNG: GlobalRNG | None = None


def get_global_rng(seed: int | None = None, verbose: bool = False, reset: bool = False):
    global _GLOBAL_RNG
    if _GLOBAL_RNG is None or reset:
        if reset:
            print(f"Resetting global RNG with seed {seed}.")
        # when seed is not provided, a seed will be generated based on the current time
        _GLOBAL_RNG = GlobalRNG(seed=seed)
    elif seed and seed != _GLOBAL_RNG.seed:
        raise ValueError(
            f"Global RNG has already been initialized with seed {_GLOBAL_RNG.seed}, "
            f"while a different seed {seed} was provided. "
            "This is not allowed for statistical integrity considerations. "
            "To reset all random number generators, please restart the Python session "
            "or explicitly reinitialize the global RNG through `reset_global_rng`."
        )
    else:
        print(f"Retrieving global RNG initialized with seed {_GLOBAL_RNG.seed}.")
        if seed and verbose:
            print(
                "Global RNG has already been initialized with the same seed "
                f"{_GLOBAL_RNG.seed}. The current global RNG will be reused. "
                "If this operation is meant to reset the random state, please reset "
                "the global RNG explicitly through `reset_global_rng`."
            )
    return _GLOBAL_RNG


def is_global_rng_initialized() -> bool:
    """Checks if the global RNG has been initialized."""
    return _GLOBAL_RNG is not None


reset_global_rng = partial(get_global_rng, reset=True)


def get_new_seed(num: int, generator: torch.Generator | None = None) -> int | list[int]:
    """
    Generates a new random 32-bit integer seed or a list of seeds from a given torch generator.

    Args:
        num (int): The number of seeds to generate.
        generator (torch.Generator, optional): The generator to use.
                                              If None, a new default generator is used.

    Returns:
        int | list[int]: A new random 32-bit integer seed or a list of seeds.
    """
    # The default generator is used if none is provided
    if generator is None:
        generator = torch.Generator()

    # Generate a single 32-bit integer seed
    seed = torch.randint(2**32 - 1, (num,), generator=generator)
    if num > 1:
        return seed.tolist()
    else:
        return seed.item()  # type: ignore


@contextmanager
def with_tmp_seed(seed: int | None = None):
    """
    Context manager to temporarily set and restore global seeds for torch, numpy, and
    the built-in random module, in case of directly passing a generator to an external
    method is not possible.
    """
    if seed is None:
        yield
        return

    # Save original random states
    original_torch_state = torch.get_rng_state()
    original_numpy_state = np.random.get_state()
    original_random_state = random.getstate()

    try:
        # Set new seeds
        torch.manual_seed(seed)
        np.random.seed(seed)
        random.seed(seed)
        yield
    finally:
        # Restore original random states
        torch.set_rng_state(original_torch_state)
        np.random.set_state(original_numpy_state)
        random.setstate(original_random_state)


def dummy_decorator(func):
    """A dummy decorator for backward compatibility."""
    return func
