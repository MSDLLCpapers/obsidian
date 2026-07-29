import hashlib
import logging
import random
import time
from contextlib import contextmanager

import numpy as np
import torch

logger = logging.getLogger(__name__)


class RNGManager:
    """
    A class to manage random number generation for reproducibility and statistical integrity.
    Coordinates numpy, torch, and built-in random module states.

    Attributes:
        seed (int): The seed used to initialize the RNGs.
        np_rng (np.random.Generator): The numpy random number generator instance.
        torch_rng (torch.Generator): The torch random number generator instance.

    Methods:
        tmp_seed_override(func, manual_seed=None): Decorator to temporarily override global random states.
        save_state(): Saves the RNG states to a dictionary.
        load_state(obj_dict): Loads RNG states from a dictionary and returns a new RNGManager instance.
    """

    def __init__(self, seed: int | None = None):
        # if seed is none, get a seed from current time and save it
        if seed is None:
            # time.time() * 1e6 gives microseconds level accuracy
            seed = int(time.time() * 1e6) % (2**32 - 1)
        self.seed = seed
        self.np_rng = np.random.default_rng(seed)
        self.torch_rng = torch.Generator().manual_seed(seed)
        logger.info(
            "numpy and torch random number generators initialized with seed %d. "
            "Please reuse this seed to reproduce the results.",
            seed,
        )

    def tmp_seed_override(self, func, manual_seed: int | None = None):
        """
        Decorator that temporarily overrides global random states (numpy, torch, random)
        for the duration of the decorated function call.

        Args:
            func: The function to decorate
            manual_seed: Optional fixed seed. If provided, the same seed is used for every call.
                        If None, a new seed is sampled from the RNG for each call.

        Returns:
            Wrapped function with temporary seed override
        """

        if manual_seed is not None:
            # Use fixed seed (better numerical stability across multiple calls)
            get_seed = lambda: manual_seed
        else:
            # Sample new seed each call (provides stochastic variation)
            get_seed = lambda: get_new_seed(1, self.np_rng)

        def wrapper(*args, **kwargs):
            # Move this inside the wrapper to ensure a new seed is only generated when the function is called
            # not when the decorator is defined
            # Note that results from the previous version will not be reproducible
            seed = get_seed()
            with with_tmp_seed(seed):  # type: ignore
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
        logger.info(
            "numpy and torch random number generators, and their random states restored with seed %d.", instance.seed
        )
        return instance

    def spawn_child(self, seed: int | None = None) -> "RNGManager":
        """Create independent child RNG with derived or explicit seed"""
        if seed is None:
            seed = get_new_seed(1)  # type: ignore
        return RNGManager(seed=seed)


USE_OLD_RNG_CONTROL = False

# numpy's legacy RNG (np.random.seed) accepts seeds in [0, 2**32); torch/random are more
# permissive, so this is the binding constraint for a seed shared across all three.
MAX_SEED = 2**32


def validate_seed(seed: int | None, name: str = "manual_seed") -> int | None:
    """
    Validate a user-provided seed at the API boundary and fail fast with a clear error.

    A seed of ``None`` (auto) passes through unchanged. Otherwise the seed must be an
    integer in ``[0, 2**32)`` — the range accepted by ``np.random.seed``, which the seed is
    ultimately fed into via :func:`with_tmp_seed`. Rejecting here (rather than deep in the
    RNG stack) gives callers an actionable message naming the offending argument.

    Args:
        seed: The seed to validate, or ``None``.
        name: Name of the argument being validated, used in the error message.

    Returns:
        The seed unchanged (or ``None``).

    Raises:
        TypeError: If ``seed`` is not ``None`` and not an integer.
        ValueError: If ``seed`` is outside ``[0, 2**32)``.
    """
    if seed is None:
        return None
    # bool is a subclass of int; disallow it explicitly to avoid silent True/False -> 1/0
    if isinstance(seed, bool) or not isinstance(seed, (int, np.integer)):
        raise TypeError(f"{name} must be an integer or None, got {type(seed).__name__}")
    if not (0 <= seed < MAX_SEED):
        raise ValueError(f"{name} must be in [0, 2**32), got {seed}")
    return int(seed)


def create_rng_manager(seed: int | None = None) -> RNGManager:
    """
    Create a new isolated RNG manager instance (not global singleton).

    This is the recommended way to create independent RNGs for multiple campaigns
    or experiments that should not share random state.

    Args:
        seed: Explicit seed for the RNG manager, or None to generate from current time

    Returns:
        New independent RNGManager instance
    """
    return RNGManager(seed=seed)


def create_torch_rng(seed: int) -> torch.Generator:
    """
    Create a torch Generator with given seed.

    Args:
        seed (int): The seed for the generator.

    Returns:
        torch.Generator: A new generator initialized with the given seed.
    """
    return torch.Generator().manual_seed(seed)


def get_new_seed(num: int = 1, generator: np.random.Generator | None = None) -> int | list[int]:
    """
    Generates a new random 32-bit integer seed or a list of seeds.

    Args:
        num (int): The number of seeds to generate. Defaults to 1.
        generator (np.random.Generator, optional): The numpy random generator to use.
            If None, a new generator is created (seed controlled by RNG context manager).

    Returns:
        int | list[int]: A new random 32-bit integer seed or a list of seeds.
    """

    if generator is None:
        # Use Python's random module to generate an initial seed (controlled by context manager)
        seed = random.randint(0, 2**31 - 1)
        generator = np.random.default_rng(seed)
    seeds = generator.integers(0, 2**31 - 1, size=num).tolist()
    if num == 1:
        return seeds[0]
    else:
        return seeds


def derive_seed(base_seed: int, key=None) -> int:
    """
    Deterministically mix an integer base seed with an arbitrary key into a 32-bit seed.

    This is useful for deriving a reproducible sub-seed from a base seed and some
    additional state (e.g. a data-state fingerprint), so the derived seed is a pure
    function of ``(base_seed, key)`` rather than of call history.

    Args:
        base_seed (int): The base seed to mix from.
        key: Any value whose string representation keys the derivation. When ``None``,
            the base seed is returned unchanged (no mixing), so callers that provide no
            key get bit-identical behavior to using ``base_seed`` directly.

    Returns:
        int: A 32-bit integer seed in the same range as :func:`get_new_seed`.
    """
    base_seed = int(base_seed) % (2**31 - 1)
    if key is None:
        return base_seed
    h = int(hashlib.sha256(str(key).encode()).hexdigest(), 16)
    return (base_seed + h) % (2**31 - 1)


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
    original_deterministic = torch.are_deterministic_algorithms_enabled()

    try:
        # Set new seeds and enable deterministic algorithms
        torch.manual_seed(seed)
        np.random.seed(seed)
        random.seed(seed)
        torch.use_deterministic_algorithms(True)
        yield
    finally:
        # Restore original random states and deterministic mode
        torch.set_rng_state(original_torch_state)
        np.random.set_state(original_numpy_state)
        random.setstate(original_random_state)
        torch.use_deterministic_algorithms(original_deterministic)


def dummy_decorator(func, manual_seed: int | None = None):
    """A dummy decorator for backward compatibility."""
    return func
