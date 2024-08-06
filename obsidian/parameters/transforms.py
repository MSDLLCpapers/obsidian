"""Transformation functions to normalize output responses"""

from torch import logit, sigmoid
from abc import ABC, abstractmethod
import warnings
from torch import Tensor

from obsidian.exceptions import UnfitError

# Method name pointers
f_transform_dict = {'Standard': lambda: Standard_Scaler(),
                    'Identity': lambda: Identity_Scaler(),
                    'Logit_MinMax': lambda: Logit_Scaler(),
                    'Logit_Percentage': lambda: Logit_Scaler(range_response=100, override_fit=True),
                    }


class Target_Transform(ABC):
    """
    Base class for obsidian Target transforms
    """
    def __init__(self):
        self.params = {}

    def _validate_fit(self):
        """
        Validates if all parameters have been fit before transforming.

        Raises:
            UnfitError: If any parameter value is None, indicating that the parameters have not been fit.
        """
        if not all([(v is not None) for v in self.params.values()]):
            raise UnfitError('Params must be fit before transforming.')

    @abstractmethod
    def forward(self,
                X: Tensor,
                fit: bool = False):
        pass  # pragma: no cover

    @abstractmethod
    def inverse(self,
                X: Tensor):
        pass  # pragma: no cover

    def __call__(self,
                 X: Tensor,
                 fit: bool = False):
        return self.forward(X, fit)
        

class Identity_Scaler(Target_Transform):
    """
    Dummy scaler class which simply returns the input
    """
    def forward(self,
                X: Tensor,
                fit: bool = False):
        return X
    
    def inverse(self,
                X: Tensor):
        return X


class Standard_Scaler(Target_Transform):
    """
    Scaler which normalizes based on zero mean and unit st-dev
    """
    def __init__(self):
        self.params = {'mu': None, 'sd': None}
    
    def forward(self,
                X: Tensor,
                fit: bool = False):
        
        if fit:
            X_v = X[~X.isnan()]
            self.params = {'mu': X_v.mean(), 'sd': X_v.std()}
        else:
            self._validate_fit()
        return (X-self.params['mu'])/self.params['sd']
    
    def inverse(self, X):
        self._validate_fit()
        return X*self.params['sd']+self.params['mu']


class Logit_Scaler(Target_Transform):
    """
    Scaler which normalizes based on a logit transform
    Can be fit to select an appropriate range for the logit
    """
    def __init__(self,
                 range_response: int | float = 1,
                 loc: int | float = 0,
                 override_fit: bool = False,
                 standardize: bool = True):
        self.params = {'scale': 1/range_response, 'loc': loc, 'mu': None, 'sd': None}
        # Override "fitting" when valid ranges are provided during init
        self.override_fit = override_fit
        self.standardize = standardize

    def _fit_minmax(self,
                    X: Tensor):

        # Scale X into a range from 0-1 with buffer/2 on either side
        self.override_fit = False
        range_response = X.max()-X.min()
        buffer = 0.2
        self.params['scale'] = (1-buffer)/range_response
        self.params['loc'] = X.min() - (buffer/2)*(1/self.params['scale'])

    def forward(self,
                X: Tensor,
                fit: bool = False):
        # If fit is not called, and the range is valid, transform
        # If the range is invalid, fit the range first and warn
        if not fit or self.override_fit:
            X_s = self.params['scale']*(X - self.params['loc'])
            valid_range = (X_s >= 0).all() and (X_s <= 1).all()
            if not valid_range:
                warnings.warn('Invalid range provided for logit scaler, proceeding with min-max fit')
                self._fit_minmax(X)
                return self.forward(X)
        else:
            X_v = X[~X.isnan()]
            self._fit_minmax(X_v)
            X_s = self.params['scale']*(X - self.params['loc'])
        X_st = logit(X_s)
        if self.standardize:
            self.params.update({'mu': X_st.mean(), 'sd': X_st.std()})
            return (X_st-self.params['mu'])/self.params['sd']
        else:
            return X_st
    
    def inverse(self,
                X: Tensor):
        if self.standardize:
            self._validate_fit()
            X = X*self.params['sd']+self.params['mu']
        return (1/self.params['scale'])*sigmoid(X)+self.params['loc']
