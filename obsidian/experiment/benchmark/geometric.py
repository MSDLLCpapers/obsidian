"""
Generic simulator functions based on simple geometric surfaces.
"""

import numpy as np
import pandas as pd
import torch
import math


def paraboloid(X):
    """
    Evaluates a simple n-dimensional paraboloid at the specified location.

    Args:
        X (ndarray): An (m)-observations by (d)-features array of data to be evaluated.

    Returns:
        ndarray: An (m)-sized array of responses.

    """

    if isinstance(X, pd.DataFrame):
        X = X.to_numpy()

    d = X.shape[1]
    a = np.ones(d) * 0.5
    z = 100 - (400 / d * (X - a) ** 2).sum(axis=1)

    return z


def shifted_parab(X):
    """
    Evaluates a simple n-dimensional parabaloid at the location specified.
    Target paraboloid contains a maximum of 100 at all x=0.2

    Args:
        X (ndarray): (m)-observations by (d)-features array of data to be evaluated

    Returns:
        ndarray: (m)-sized array of responses
    """
    if isinstance(X, pd.DataFrame):
        X = X.to_numpy()

    d = X.shape[1]
    a = np.ones(d) * 0.2
    z = 100 * np.exp(-(4 / d * (X - a) ** 2).sum(axis=1))
    return z


def cornered_parab(X):
    """
    Evaluates a simple n-dimensional parabaloid at the location specified.
    
    Args:
        X (ndarray): (m)-observations by (d)-features array of data to be evaluated
    
    Returns:
        ndarray: (m)-sized array of responses
    """
    if isinstance(X, pd.DataFrame):
        X = X.to_numpy()

    d = X.shape[1]
    a = np.ones(d) * 0
    z = 100 - (400 / d * (X - a) ** 2).sum(axis=1)
    
    return z


def ackley(X):
    """
    Evaluates the N-dimensional Ackley function.

    The Ackley function is a benchmark optimization problem that is commonly used to test optimization algorithms.
    It is a multimodal function with multiple local maxima and a global minimum at (0.8, 0.8, ..., 0.8).
    This implementation of the Ackley function has been transformed to have a maximum value of 10 at the global minimum.

    Args:
        X (ndarray or DataFrame):
            An (m x d) array or DataFrame of m observations with d features to be evaluated.

    Returns:
        ndarray:
            An (m)-sized array of responses.
    """

    if isinstance(X, pd.DataFrame):
        X = X.to_numpy()

    # Rescale problem from (0,1) to desired space (-1,3)
    X = X*4-3.2

    a = 10  # default = 20
    b = 0.2  # default = 0.2
    c = 1*math.pi  # default = 2*math.pi
    d = X.shape[1]
    
    z = -a*np.exp(-b*((1/d)*np.sum(X**2, axis=1))**0.5) - np.exp((1/d)*np.sum(np.cos(c*X), axis=1))+a+np.exp(1)
    
    z = -z + 10
    
    return z


def rosenbrock(X):
    """
    Evaluates the Rosenbrock function in log10 scale, and negative transform (for max).

    The global maximum of the Rosenbrock function is at X = (1, 1, ..., 1).
    The maximum value is 10, based on the standard function minimum of 0. We have -log10(0+1e-10).

    Args:
        X (ndarray): (m)-observations by (d)-features array of data to be evaluated.

    Returns:
        ndarray: (m)-sized array of responses.
    """

    if isinstance(X, pd.DataFrame):
        X = X.to_numpy()

    # Rescale problem from (0,1) to desired space (-1,3)
    X = X*4-2

    z = np.zeros(shape=(X.shape[0]))
    x_dim = X.shape[1]
    
    b = 100
    a = 1
    
    for i in range(0, x_dim-1):
        z += b*(X[:, i+1]-X[:, i]**2)**2+(a-X[:, i])**2
    
    z = -np.log10(z+1e-10)
    
    return z


def sixhump_camel(X):
    """
    Evaluates the 6-hump camel function in 2D
    Transformed to get a max of 10 at X = (0.0898,-0.7126) and (-0.0898, 0.7126)

    Args:
        X (ndarray): (m)-observations by (d)-features array of data to be evaluated.

    Returns:
        ndarray: (m)-sized array of responses.

    """

    if X.shape[1] != 2:
        raise ValueError('Camel function only accepts 2 dimensions')

    if isinstance(X, pd.DataFrame):
        X = X.to_numpy()

    # Rescale problem from (0,1) to desired space (-2,2),(-1,1)
    x1 = X[:, 0]*4-2
    x2 = X[:, 1]*2-1
    
    z = (4-2.1*x1**2+(x1**4)/3)*x1**2+x1*x2+(-4+4*x2**2)*x2**2
    
    z = -z+1.0316+10
    
    return z


def threehump_camel(X):
    """
    Evaluates the 3-hump camel function in 2D
    Negative transform, to get max
    Two global max exist at X = (0,0)
    Max is 1

    Args:
        X (ndarray): (m)-observations by (d)-features array of data to be evaluated.

    Returns:
        ndarray: (m)-sized array of responses.

    """

    if isinstance(X, pd.DataFrame):
        X = X.to_numpy()
    
    if X.shape[1] != 2:
        raise ValueError('Camel function only accepts 2 dimensions')

    # Rescale problem from (0,1) to desired space (-2,2)
    X = X*4-2

    x1 = X[:, 0]
    x2 = X[:, 1]
    
    z = 2*x1**2-1.05*x1**4+(x1**6)/6+x1*x2+x2**2
    
    z = -z+10

    return z


def perm(X):
    """
    Evaluates the perm function in N-dimensions
    Negative transform, to get max
    Max exists at X = (1,1/2,...,1/d)
    Max is 10
    
    Args:
        X (ndarray): (m)-observations by (d)-features array of data to be evaluated.

    Returns:
        ndarray: (m)-sized array of responses.

    """
    
    if isinstance(X, pd.DataFrame):
        X = X.to_numpy()

    # Rescale problem from (0,1) to desired space (0,1.5)
    X = X*2-0.5
        
    z = np.zeros(shape=(X.shape[0]))
    x_dim = X.shape[1]
    
    beta = 10
    
    for i in range(1, x_dim+1):
        inner_sum = 0
        for j in range(1, x_dim+1):
            xj = X[:, j-1]
            inner_sum += (j+beta)*((xj**i)-(1/j)**i)
        z += inner_sum**2
    
    z = -np.log(z+1)+10
    
    return z


def branin_currin(X):
    """
    Synthetic BraninCurrin function from BoTorch test_functions

    Two objective problem composed of the Branin and Currin functions:
    
        Branin (rescaled):
    
            f_1(x) = (
            15*x_1 - 5.1 * (15 * x_0 - 5) ** 2 / (4 * pi ** 2) + 5 * (15 * x_0 - 5)
            / pi - 5
            ) ** 2 + (10 - 10 / (8 * pi)) * cos(15 * x_0 - 5))
    
        Currin:
    
            f_2(x) = (1 - exp(-1 / (2 * x_1))) * (
            2300 * x_0 ** 3 + 1900 * x_0 ** 2 + 2092 * x_0 + 60
            ) / 100 * x_0 ** 3 + 500 * x_0 ** 2 + 4 * x_0 + 20
    
    Args:
        X (ndarray): (m)-observations by (d=2)-features array of data to be evaluated.

    Returns:
        ndarray: (k=2)-sized array of responses.
    """
    if isinstance(X, pd.DataFrame):
        X = torch.tensor(X.values)
    else:
        X = torch.tensor(X)
    
    if X.shape[1] != 2:
        raise ValueError('BraninCurrin function only accepts 2 dimensions')
    
    from botorch.test_functions.multi_objective import BraninCurrin
    problem = BraninCurrin(negate=True)
    z = problem(X).numpy()
    return z


def two_leaves(X):
    """
    Simulates the two-leaf multi-output function

    Args:
        X (ndarray): (m)-observations by (d=2)-features array of data to be evaluated.

    Returns:
        ndarray: (m=2)-sized array of responses.
    """
    if isinstance(X, pd.DataFrame):
        X = torch.tensor(X.values)
    if isinstance(X, np.ndarray) and X.ndim == 1:
        X = X[np.newaxis, :]
    if isinstance(X, np.ndarray):
        X = torch.tensor(X)
    x_0 = X[..., 0]
    x_1 = X[..., 1]
    t = torch.sqrt(torch.abs((x_0-0.4)*math.pi/2)+0.1)
    y_0 = t * torch.sin(x_0*math.pi/2) * torch.pow(x_1, 2)
    y_1 = t * torch.cos(x_0*math.pi/2)  # * torch.pow(x_1,3)
    return np.array(torch.stack([y_0, y_1], dim=-1))
