"""Utility functions for parameter handling"""

import numpy as np


def transform_with_type(inner):
    """
    Wraps parameter transform functions (map, encode, etc)
    in a way that returns the same type that was given
    """
    def wrapper(self, X):
        X_arr = np.array(X)
        X_t = inner(self, X_arr)
        return X_t.tolist() if isinstance(X, (list, int, float, str)) else X_t
    return wrapper
