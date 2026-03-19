"""Shared pytest fixtures for obsidian tests"""

import pytest
import obsidian


@pytest.fixture(params=[False, True], ids=["new_rng", "old_rng"])
def rng_mode(request):
    """
    Fixture to test both RNG control modes.

    Parametrizes tests to run with both:
    - new_rng: USE_OLD_RNG_CONTROL = False (default, RNGManager-based)
    - old_rng: USE_OLD_RNG_CONTROL = True (legacy, direct seeding)

    Automatically resets to default after each test.
    """
    original = obsidian.USE_OLD_RNG_CONTROL
    obsidian.USE_OLD_RNG_CONTROL = request.param
    yield request.param
    obsidian.USE_OLD_RNG_CONTROL = original
