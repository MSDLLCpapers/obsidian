
from obsidian.exceptions import UnsupportedError

import numpy as np
from numpy.random import Generator
from itertools import product
from types import ModuleType


def _fractional_factorial_2_level(d: int) -> np.ndarray:
    """
    Helper function to create Resolution IV+ fractional factorial designs for 2-level experiments.
    
    Args:
        d (int): Number of dimensions (2-12)
        
    Returns:
        ndarray: Fractional factorial design in (-1, 1) domain
    """
    steps = np.array([-1, 1])
    
    # Create a dictionary that allows us to use letters
    term_codes = {}
    alphabet = 'ABCDEFGHJKLMNOPQRSTUVWXYZ'  # DESIGN EXPERT SKIPS I!!!
    for i, letter in enumerate(alphabet):
        term_codes[letter] = i
    
    # Resolution IV+ alias structure
    res4_aliases = {
        2: {}, 3: {},
        4: {'D': 'ABC'},
        5: {'E': 'ABCD'}, 
        6: {'E': 'ABC', 'F': 'BCD'},
        7: {'F': 'ABCD', 'G': 'ABCE'}, 
        8: {'F': 'ABC', 'G': 'ABD', 'H': 'BCDE'},
        9: {'G': 'ABCD', 'H': 'ACEF', 'J': 'CDEF'},
        10: {'G': 'ABCD', 'H': 'ABCE', 'J': 'ADEF', 'K': 'BDEF'},
        11: {'G': 'ABCD', 'H': 'ABCE', 'J': 'ABDE', 'K': 'ACDEF', 'L': 'BCDEF'},
        12: {'H': 'ABC', 'J': 'ADEF', 'K': 'BDEG', 'L': 'CDFG', 'M': 'ABCEFG'}
    }
    
    # Determine the resolved dimensions (base design size)
    if res4_aliases[d] == {}:
        d_resolved = d
    else:
        decoded_keys = [term_codes[k] for k in res4_aliases[d].keys()]
        d_resolved = np.min(decoded_keys)
    
    # Create base design
    axes = np.tile(steps, (d_resolved, 1))
    X = np.array(list(product(*axes)))
    
    # Generate aliased columns
    for term, generator in res4_aliases[d].items():
        decoded_generator = [term_codes[g] for g in list(generator)]
        aliased_term = np.product(X[:, decoded_generator], axis=-1)[:, np.newaxis]
        X = np.hstack((X, aliased_term))
    
    return X


def factorial_DOE_n_level(d: int,
                          levels: int = 2,
                          n_CP: int | None = None,
                          shuffle: bool = True,
                          seed: Generator | int | None = None,
                          full: bool = False):
    """
    Creates a statistically designed factorial experiment (DOE).
    Supports n-level designs (2-level, 3-level, etc.).
    Uses the range (0,1) for low-high.

    Args:
        d (int): Number of dimensions/inputs in the design.
        levels (int, optional): Number of levels per factor (e.g., 2, 3, 4). Default is ``2``.
        n_CP (int | None, optional): Number of replicate centerpoints, for estimating pure error
            and testing curvature/lack-of-fit. Default (``None``) is ``3`` for all levels, since
            replication is what provides pure-error degrees of freedom. Use ``n_CP=0`` for a
            deterministic, noise-free grid comparison.
        shuffle (bool, optional): Whether or not to shuffle the design or leave them in the default run
            order. Default is ``True``.
        seed (Generator | int | None, optional): Controls the run-order shuffle. A ``Generator``
            is used directly; an int seeds an isolated ``default_rng`` (reproducible, without
            touching global RNG state); ``None`` (default) defers to the ambient global
            ``np.random`` stream (e.g. one set by ``with_tmp_seed``). Global state is never reseeded.
        full (bool, optional): Whether or not to run the full DOE. Default is ``False``, which
            will lead to an efficient Res4+ design (2-level only).

    Returns:
        ndarray: An (m)-by-(d) array of experiments in the (0,1) domain

    Raises:
        UnsupportedError: If the number of dimensions exceeds 12
        ValueError: If d < 1, levels < 2, n_CP < 0, or a fractional factorial is
            requested with levels != 2
    """
    if d < 1:
        raise ValueError('The number of dimensions must be at least 1')
    if d > 12:
        raise UnsupportedError('The number of dimensions must be 12 or fewer for DOE (currently)')
    if levels < 2:
        raise ValueError('The number of levels must be at least 2')

    if not full and levels != 2:
        raise ValueError('Fractional factorial designs only supported for 2-level. Use full=True for n-level designs.')

    # Replicate center runs; default to 3 for all levels.
    if n_CP is None:
        n_CP = 3
    if n_CP < 0:
        raise ValueError('The number of centerpoints must be non-negative')

    # Generate centerpoints (only if needed)
    if n_CP > 0:
        CP = np.zeros(shape=(n_CP, d))

    # Create design based on type. A single factor cannot be fractionated, so fall back
    # to the full factorial branch (which yields the same 2-run design) to avoid the
    # fractional-factorial helper, which is only defined for d >= 2.
    if levels == 2 and not full and d >= 2:
        # Use fractional factorial for efficiency
        X = _fractional_factorial_2_level(d)
    else:
        # Full factorial for any number of levels
        steps = np.linspace(-1, 1, levels)
        axes = np.tile(steps, (d, 1))
        X = np.array(list(product(*axes)))

    # Add centerpoints if any
    if n_CP > 0:
        X = np.vstack((X, CP))

    # Shuffle the run order. Only resolve an RNG when actually shuffling.
    if shuffle:
        if isinstance(seed, Generator):
            rng: Generator | ModuleType = seed  # use the provided generator directly
        elif seed is None:
            rng = np.random  # read the ambient global stream; never reseed it
        else:
            rng = np.random.default_rng(seed)  # isolated + reproducible; global untouched
        rng.shuffle(X)

    # Rescale from (-1,1) to [0,1], keeping interior points (incl. center) exact.
    # The upper edge is nudged just below 1 so discrete unit_demap (int(X*nc))
    # does not overflow the top category, without shifting the rest of the design.
    X = (X + 1) / 2
    X[X >= 1] = 1 - 1e-6

    return X


# Uniform-precision center-point counts for rotatable CCDs (Box & Hunter).
# Falls back to 3 for dimensions not tabulated.
_CCD_UNIFORM_PRECISION_CP = {2: 5, 3: 6, 4: 7, 5: 10, 6: 15, 7: 21}


def central_composite_DOE(d: int,
                          alpha: float | str = 'rotatable',
                          n_CP: int | None = None,
                          face_core_full: bool = True,
                          inscribe: bool = True,
                          shuffle: bool = True,
                          seed: Generator | int | None = None):
    """
    Creates a classical Central Composite Design (CCD) for response surface methodology (RSM).

    A CCD combines a two-level factorial (or fractional-factorial) core to estimate main
    effects and two-factor interactions, axial ("star") points to estimate quadratic
    curvature, and replicated center points to estimate pure error. It is the standard
    "old school" design for fitting a full second-order (quadratic) response surface.

    Because obsidian generates designs in the (0,1) unit cube (which is then mapped onto
    the real parameter bounds), the design is by default *inscribed* (``inscribe=True``):
    when the axial distance exceeds the cube (``alpha > 1``), the whole design is scaled by
    ``1/alpha`` so the axial points sit on the cube faces and the factorial core is pulled
    inward, guaranteeing every point falls within the parameter bounds (at the cost of
    running the core at a fraction of the full range). When ``alpha <= 1`` (e.g. ``'faced'``)
    the design already fits inside the cube, so no scaling is applied. Set ``inscribe=False``
    to instead keep the factorial core at the box corners and clip axials to the bounds,
    which breaks rotatability -- equivalent to a face-centered design.

    Args:
        d (int): Number of dimensions/inputs in the design.
        alpha (float | str, optional): Axial distance in coded (-1, 1) units. Either a
            positive float, or one of:
            - ``'rotatable'`` (default): ``alpha = n_factorial ** 0.25``, giving a
              rotatable design (constant prediction variance at fixed distance from center).
            - ``'faced'``: ``alpha = 1`` (face-centered, "CCF"); axials lie on the cube faces
              and only three levels per factor are used.
        n_CP (int | None, optional): Number of center points. If ``None`` (default), uses
            standard uniform-precision values (e.g. 5/6/7 for d=2/3/4), falling back to 3.
        face_core_full (bool, optional): Whether the factorial core is a full ``2**d``
            design (default) or an efficient Resolution IV+ fractional factorial. Full cores
            are recommended for RSM (Resolution V+) but grow quickly with d.
        inscribe (bool, optional): Whether to inscribe the design in the unit cube so all
            points respect the parameter bounds. Default is ``True``.
        shuffle (bool, optional): Whether to shuffle the run order. Default is ``True``.
        seed (Generator | int | None, optional): Controls the run-order shuffle. A ``Generator``
            is used directly; an int seeds an isolated ``default_rng`` (reproducible, without
            touching global RNG state); ``None`` (default) defers to the ambient global
            ``np.random`` stream (e.g. one set by ``with_tmp_seed``). Global state is never reseeded.

    Returns:
        ndarray: An (m)-by-(d) array of experiments in the (0,1) domain.

    Raises:
        UnsupportedError: If the number of dimensions exceeds 12.
        ValueError: If d < 1, n_CP < 0, or ``alpha`` is an unrecognized string or a
            non-positive value.
    """
    if d < 1:
        raise ValueError('The number of dimensions must be at least 1')
    if d > 12:
        raise UnsupportedError('The number of dimensions must be 12 or fewer for DOE (currently)')

    # Factorial core in coded (-1, 1) units, without center points or (0,1) rescaling.
    # A single factor cannot be fractionated, so use the full core for d == 1.
    if face_core_full or d < 2:
        steps = np.array([-1, 1])
        axes = np.tile(steps, (d, 1))
        core = np.array(list(product(*axes)), dtype=float)
    else:
        core = _fractional_factorial_2_level(d).astype(float)

    n_factorial = core.shape[0]

    # Resolve the axial distance
    if isinstance(alpha, str):
        if alpha == 'rotatable':
            alpha_val = n_factorial ** 0.25
        elif alpha == 'faced':
            alpha_val = 1.0
        else:
            raise ValueError("alpha must be a positive float, 'rotatable', or 'faced'")
    else:
        alpha_val = float(alpha)
        if alpha_val <= 0:
            raise ValueError('alpha must be a positive value')

    # Axial ("star") points: +/- alpha along each individual axis
    axial = np.vstack([np.eye(d) * alpha_val, -np.eye(d) * alpha_val])

    # Center points
    if n_CP is None:
        n_CP = _CCD_UNIFORM_PRECISION_CP.get(d, 3)
    if n_CP < 0:
        raise ValueError('The number of centerpoints must be non-negative')
    CP = np.zeros(shape=(n_CP, d)) if n_CP > 0 else np.empty((0, d))

    X = np.vstack((core, axial, CP))

    # Fit the design within the unit cube
    if inscribe:
        # Scale so the axial points sit on the cube faces (coded +/- 1)
        if alpha_val > 1:
            X = X / alpha_val
    else:
        # Keep the factorial core at the corners; clip axials to the box faces
        X = np.clip(X, -1, 1)

    # Shuffle the run order. Only resolve an RNG when actually shuffling.
    if shuffle:
        if isinstance(seed, Generator):
            rng: Generator | ModuleType = seed  # use the provided generator directly
        elif seed is None:
            rng = np.random  # read the ambient global stream; never reseed it
        else:
            rng = np.random.default_rng(seed)  # isolated + reproducible; global untouched
        rng.shuffle(X)

    # Rescale from (-1,1) to [0,1], keeping interior points (incl. center) exact.
    # The upper edge is nudged just below 1 so discrete unit_demap (int(X*nc))
    # does not overflow the top category, without shifting the rest of the design.
    X = (X + 1) / 2
    X[X >= 1] = 1 - 1e-6

    return X


def factorial_DOE(d: int,
                  n_CP: int = 3,
                  shuffle: bool = True,
                  seed: Generator | int | None = None,
                  full: bool = False):
    """
    Creates a statistically designed factorial experiment (DOE).
    Specifically for 2-level designs only.
    Uses the range (0,1) for low-high instead of the typical (-1,1),
    although (-1,1) is used for calculations during alias design.
    
    For n-level designs (3-level, 4-level, etc.), use `factorial_DOE_n_level`.

    Args:
        d (int): Number of dimensions/inputs in the design.
        n_CP (int, optional): The number of centerpoints to include in the design, for estimating
            uncertainty and curvature. Default is ``3``.
        shuffle (bool, optional): Whether or not to shuffle the design or leave them in the default run
            order. Default is ``True``.
        seed (Generator | int | None, optional): Controls the run-order shuffle. A ``Generator``
            is used directly; an int seeds an isolated ``default_rng`` (reproducible, without
            touching global RNG state); ``None`` (default) defers to the ambient global
            ``np.random`` stream (e.g. one set by ``with_tmp_seed``). Global state is never reseeded.
        full (bool, optional): Whether or not to run the full DOE. Default is ``False``, which
            will lead to an efficient Res4+ design.

    Returns:
        ndarray: An (m)-by-(d) array of experiments in the (0,1) domain

    Raises:
        UnsupportedError: If the number of dimensions exceeds 12
        ValueError: If d < 1 or n_CP < 0
    """
    return factorial_DOE_n_level(d=d, levels=2, n_CP=n_CP, shuffle=shuffle, seed=seed, full=full)
