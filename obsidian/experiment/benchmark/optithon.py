"""
Simulation functions used for an optimization hackathon (OptiThon) in March 2024
"""

import numpy as np
import pandas as pd
from scipy.special import gamma


def Vm_func(X):
    # Maximum rate as a function of X
    rate = np.exp(-2*(X-0.75)**2)
    return rate


def Km_func(X):
    # Michaelis constant as a function of X
    rate = 2-np.exp(-1*(X-0.4)**2)
    return rate


def kI_func(X):
    # Inhibition constant as a function of X
    rate = 0.4/(1+np.exp(-1*(X-0.2)**2))
    return rate


def response_1(X):
    if X.shape[1] != 3:
        raise ValueError(f'Input to response_1 must have 3 columns, not {X.shape[1]}')
    X1, X2, X3 = X.T

    # Calculate kinetic parameters
    Vm = 100*Vm_func(X2)
    Km = Km_func(X3)
    kI = kI_func(X2)
    
    # Enzyme inhibition rate law
    rate = Vm*X1/(0.001+Km+X1+(X1/kI)**2)

    # Normalize
    rate = rate/10.5
    return rate


def response_2(X):
    if X.shape[1] != 2:
        raise ValueError(f'Input to response_2 must have 2 columns, not {X.shape[1]}')
    X1, X2 = X.T
    X2 = X2*1 + 2  # (ensure that there is at least one reactor)

    # Gamma function used in RTD calc
    gamma_value = gamma(X2)
    
    # RTD for a pulse into N CSTRs
    E_curve = (X1**(X2-1))*((2*X2)**X2)*np.exp(-2*X2*X1)/(gamma_value)
    
    # Normalize
    E_curve = E_curve/1.62
    
    return E_curve


def OT_simulator(X, addNoise=False):
    if isinstance(X, pd.DataFrame):
        X = X.to_numpy()
    if isinstance(X, np.ndarray) and X.ndim == 1:
        X = X[np.newaxis, :]

    if X.shape[1] != 6:
        raise ValueError(f'Input to simulate must have 6 columns, not {X.shape[1]}')
    
    # Break down the data into 3 parts for separate functions
    X123 = X[:, 0:3]
    X45 = X[:, 3:5]
    X6 = X[:, 5]
    
    # First three cols represented in an enzyme kinetics problem
    y1 = response_1(X123)
    
    # Next two cols represented in a flow problem
    y2 = response_2(X45)
    
    # Total performance is a weighted sum of two problems
    y = 0.6*y1 + 0.4*y2
    
    # Final column has a mean 0 effect, but determines the scale of noise
    # Note: The overall problem is slightly heteroskedastic
    if addNoise:
        sd = 0.01+0.02*X6
        rel_error = np.random.normal(loc=1, scale=sd, size=y.shape[0])
        y *= rel_error
    
    return y
