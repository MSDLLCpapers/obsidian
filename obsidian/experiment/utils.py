
from obsidian.exceptions import UnsupportedError

import numpy as np
from itertools import product


def factorial_DOE(d: int,
                  n_CP: int = 3,
                  shuffle: bool = True,
                  seed: int | None = None,
                  full: bool = False):
    """
    Creates a statistically designed factorial experiment (DOE).
    Specifically for 2-level designs only.
    Uses the range (0,1) for low-high instead of the typical (-1,1),
    although (-1,1) is used for calculations during alias design

    Args:
        d (int): Number of dimensions/inputs in the design.
        n_CP (int, optional): The number of centerpoints to include in the design, for estimating
            uncertainty and curvature. Default is ``3``.
        shuffle (bool, optional): Whether or not to shuffle the design or leave them in the default run
            order. Default is ``True``.
        seed (int, optional): Randomization seed. Default is ``None``.
        full (bool, optional): Whether or not to run the full DOE. Default is ``False``, which
            will lead to an efficient Res4+ design.

    Returns:
        ndarray: An (m)-by-(d) array of experiments in the (0,1) domain

    Raises:
        UnsupportedError: If the number of dimensions exceeds 12
    """
    if d > 12:
        raise UnsupportedError('The number of dimensions must be 12 or fewer for DOE (currently)')
    
    steps = np.array([-1, 1])
    CP = np.zeros(shape=(n_CP, d))
    
    # Create a dictionary that allows us to use letters
    term_codes = {}
    alphabet = 'ABCDEFGHJKLMNOPQRSTUVWXYZ'  # DESIGN EXPERT SKIPS I!!!
    for i, letter in enumerate(alphabet):
        term_codes[letter] = i
    
    res4_aliases = {2: {}, 3: {},
                    4: {'D': 'ABC'},
                    5: {'E': 'ABCD'}, 6: {'E': 'ABC', 'F': 'BCD'},
                    7: {'F': 'ABCD', 'G': 'ABCE'}, 8: {'F': 'ABC', 'G': 'ABD', 'H': 'BCDE'},
                    9: {'G': 'ABCD', 'H': 'ACEF', 'J': 'CDEF'},
                    10: {'G': 'ABCD', 'H': 'ABCE', 'J': 'ADEF', 'K': 'BDEF'},
                    11: {'G': 'ABCD', 'H': 'ABCE', 'J': 'ABDE', 'K': 'ACDEF', 'L': 'BCDEF'},
                    12: {'H': 'ABC', 'J': 'ADEF', 'K': 'BDEG', 'L': 'CDFG', 'M': 'ABCEFG'}
                    }
    
    if full:
        axes = np.tile(steps, (d, 1))
        X = np.array(list(product(*axes)))
    else:
        if res4_aliases[d] == {}:
            d_resolved = d
        else:
            decoded_keys = [term_codes[k] for k in res4_aliases[d].keys()]
            d_resolved = np.min(decoded_keys)  # e.g. If the first aliased key is G, we are resolved up to F
        axes = np.tile(steps, (d_resolved, 1))
        X = np.array(list(product(*axes)))
        for term, generator in res4_aliases[d].items():
            decoded_generator = [term_codes[g] for g in list(generator)]
            aliased_term = np.product(X[:, decoded_generator], axis=-1)[:, np.newaxis]
            X = np.hstack((X, aliased_term))
    
    # Add centerpoints then shuffle
    X = np.vstack((X, CP))
    if seed is not None:
        np.random.seed(seed)
    if shuffle:
        np.random.shuffle(X)
    # Rescale from (-1,1) to (0,0.999999)
    X = (X+1)/2 - 1e-6
    
    return X
