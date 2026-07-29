"""Transformation functions to normalize output responses"""

import warnings
from abc import ABC, abstractmethod

from torch import Tensor, logit, sigmoid, zeros_like

from obsidian.exceptions import UnfitError

# Method name pointers
f_transform_dict = {
    "Standard": lambda: Standard_Scaler(),
    "Identity": lambda: Identity_Scaler(),
    "Logit_MinMax": lambda: Logit_Scaler(),
    "Logit_Percentage": lambda: Logit_Scaler(range_response=100, override_fit=True),
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
            raise UnfitError("Params must be fit before transforming.")

    @abstractmethod
    def forward(self, X: Tensor, fit: bool = False):
        """Evaluate the forward transformation on input data X"""
        pass  # pragma: no cover

    @abstractmethod
    def inverse(self, X: Tensor):
        """Inverse transform the transformed data X_t"""
        pass  # pragma: no cover

    def __call__(self, X: Tensor, fit: bool = False):
        """Shortcut to forward method"""
        return self.forward(X, fit)


class Identity_Scaler(Target_Transform):
    """
    Dummy scaler class which simply returns the input
    """

    def forward(self, X: Tensor, fit: bool = False):
        """Evaluate the forward transformation on input data X"""
        return X

    def inverse(self, X: Tensor):
        """Inverse transform the transformed data X_t"""
        return X


class Standard_Scaler(Target_Transform):
    """
    Scaler which normalizes based on zero mean and unit st-dev
    """

    def __init__(self):
        self.params = {"mu": None, "sd": None}

    def forward(self, X: Tensor, fit: bool = False):
        """Evaluate the forward transformation on input data X"""
        if fit:
            X_v = X[~X.isnan()]
            if X_v.numel() == 0:
                warnings.warn("No non-NaN values to fit; returning zeros.", UserWarning)
                self.params = {"mu": X.new_tensor(0.0), "sd": X.new_tensor(1.0)}
                return zeros_like(X)
            mu = X_v.mean()
            sd = X_v.std()
            if sd.isnan():
                # Single element: unbiased std is undefined, treat as zero variance
                sd = X_v.new_tensor(0.0)
            self.params = {"mu": mu, "sd": sd}
        else:
            self._validate_fit()
        if self.params["sd"] == 0:
            # In the edge case where `X` is degenerate, avoid 0 divided by 0
            warnings.warn("Transform constant target values by mean subtraction", UserWarning)
            return zeros_like(X)
        else:
            return (X - self.params["mu"]) / self.params["sd"]

    def inverse(self, X):
        """Inverse transform the transformed data X_t"""
        self._validate_fit()
        return X * self.params["sd"] + self.params["mu"]


class Logit_Scaler(Target_Transform):
    """
    Scaler which normalizes based on a logit transform
    Can be fit to select an appropriate range for the logit
    """

    def __init__(
        self,
        range_response: int | float = 1,
        loc: int | float = 0,
        override_fit: bool = False,
        standardize: bool = True,
    ):
        self.params = {"scale": 1 / range_response, "loc": loc, "mu": None, "sd": None}
        # Override "fitting" when valid ranges are provided during init
        self.override_fit = override_fit
        self.standardize = standardize

    def _fit_minmax(self, X: Tensor):
        """Fits the min-max scale of the logit transform"""
        # Scale X into a range from 0-1 with buffer/2 on either side
        self.override_fit = False
        range_response = X.max() - X.min()

        # Handle constant or near-constant data
        if range_response < 1e-10:
            warnings.warn(
                "Cannot fit logit transform to constant or near-constant data. All values will be mapped to zero.",
                UserWarning,
            )
            # Set parameters that will result in zeros_like output
            self.params["scale"] = 1.0
            self.params["loc"] = X.mean().item()
            self.params["constant_data"] = True
            # Set mu and sd for inverse transform compatibility
            self.params["mu"] = 0.0
            self.params["sd"] = 1.0
            return

        self.params["constant_data"] = False
        buffer = 0.2
        self.params["scale"] = (1 - buffer) / range_response
        self.params["loc"] = X.min() - (buffer / 2) * (1 / self.params["scale"])

    def forward(self, X: Tensor, fit: bool = False):
        """Evaluate the forward transformation on input data X"""
        # Fit the range parameters if needed
        if fit and not self.override_fit:
            X_v = X[~X.isnan()]
            self._fit_minmax(X_v)

            # Early return for constant data
            if self.params.get("constant_data", False):
                return zeros_like(X)

        # Check if transform was fitted on constant data
        if self.params.get("constant_data", False):
            return zeros_like(X)

        # Scale the data
        X_s = self.params["scale"] * (X - self.params["loc"])

        # Validate range: always enforce for override_fit=True; for normal fit path
        # _fit_minmax guarantees [0.1, 0.9] so the check is a no-op, but still run
        # it for safety. For fit=False, also validate params are set first.
        if not fit:
            self._validate_fit()
        X_s_v = X_s[~X_s.isnan()]
        if X_s_v.numel() > 0:
            min_val = X_s_v.min().item()
            max_val = X_s_v.max().item()
            tol = 1e-6
            if min_val < -tol or max_val > 1 + tol:
                raise ValueError(
                    "Input data out of valid logit range [0, 1]. "
                    f"Got [{min_val:.6g}, {max_val:.6g}]. "
                    "Check that the correct transform is being applied to this data."
                )
            elif min_val < 0 or max_val > 1:
                warnings.warn(
                    "Input data slightly outside [0, 1] range (suspected numerical inaccuracy). "
                    f"Got [{min_val:.6g}, {max_val:.6g}]. Rescaling to [0, 1].",
                    UserWarning,
                )
                range_val = max_val - min_val
                if range_val == 0:
                    warnings.warn(
                        "Input data outside [0, 1] has zero range. Mapping non-NaN values to 0.5.",
                        UserWarning,
                    )
                    X_s = X_s.clone()
                    X_s[~X_s.isnan()] = 0.5
                else:
                    X_s = (X_s - min_val) / range_val

        # Apply logit transform
        X_st = logit(X_s)

        # Fit or apply standardization
        if self.standardize:
            if fit:
                # Fit standardization parameters using only finite values
                finite_mask = X_st.isfinite()
                X_st_finite = X_st[finite_mask]

                # Handle case where logit-transformed data has no finite values
                if X_st_finite.numel() == 0:
                    warnings.warn(
                        "Logit-transformed data contains no finite values. Using identity standardization parameters.",
                        UserWarning,
                    )
                    self.params.update({"mu": 0.0, "sd": 1.0})
                    return X_st

                mu = X_st_finite.mean()
                sd = X_st_finite.std()

                # Handle case where logit-transformed finite data has zero variance
                if sd == 0 or sd.isnan() or sd.isinf():
                    warnings.warn(
                        "Logit-transformed data has zero variance. Returning zero-centered values.", UserWarning
                    )
                    self.params.update({"mu": mu, "sd": 1.0})
                    return X_st - mu

                self.params.update({"mu": mu, "sd": sd})
                return (X_st - self.params["mu"]) / self.params["sd"]
            else:
                # Apply standardization using fitted parameters
                self._validate_fit()
                return (X_st - self.params["mu"]) / self.params["sd"]
        else:
            return X_st

    def inverse(self, X: Tensor):
        """Inverse transform the transformed data X_t"""
        # Handle constant data case
        if self.params.get("constant_data", False):
            # For constant data, forward returns zeros, so inverse should return the original constant
            return zeros_like(X) + self.params["loc"]

        if self.standardize:
            self._validate_fit()
            X = X * self.params["sd"] + self.params["mu"]
        return (1 / self.params["scale"]) * sigmoid(X) + self.params["loc"]
