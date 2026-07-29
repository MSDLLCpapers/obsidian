"""Output features to be optimized"""

import numbers
from typing import Any

import numpy as np
import pandas as pd
import torch
from numpy.typing import ArrayLike

from obsidian.config import TORCH_DTYPE
from obsidian.exceptions import UnfitError

from .transforms import f_transform_dict


def to_tensor(f: Any, dtype: torch.dtype = TORCH_DTYPE, device: torch.device | str | int | None = None) -> torch.Tensor:
    """
    Validate and convert f to a torch.Tensor. Accepts: torch.Tensor, numpy.ndarray,
    pandas Series/DataFrame, Python scalar (int/float), or list/tuple of numerics.

    Args:
        f: input to convert
        dtype: optional torch dtype for the resulting tensor
        device: optional torch device for the resulting tensor

    Returns:
        torch.Tensor
    """
    # If already a torch tensor, optionally cast device/dtype and return
    if torch.is_tensor(f):
        pass
    else:
        if isinstance(f, (pd.Series, pd.DataFrame)):
            arr = f.values
        elif isinstance(f, np.ndarray):
            arr = f
        elif isinstance(f, numbers.Number):
            arr = np.array([f])
        # list/tuple/iterable: convert to numpy array
        elif isinstance(f, (list, tuple)):
            arr = np.array(f)
        else:
            raise TypeError(
                "f being transformed must be numeric or array-like "
                "(torch.Tensor, numpy.ndarray, pandas Series/DataFrame, list/tuple, or numeric scalar)"
            )

        if not np.issubdtype(arr.dtype, np.number):
            raise TypeError("Each element of f being transformed must be numeric")

        # Convert numpy to torch
        f = torch.from_numpy(arr)
    if f.dtype != dtype or device is not None:
        return f.to(device=device, dtype=dtype)
    return f


def resolve_target_names(
    targets: list,
    target_names: list[str] | None,
    *,
    require_thresholds: bool = False,
    drop_tracking_only: bool = True,
) -> list:
    """Resolve a name filter against a list of ``Target``s.

    Centralizes the existence / tracking-only / threshold checks shared by
    plotting and characterization. ``None`` means "all targets that pass the
    filters"; an explicit list raises on unknown names so typos surface.

    Args:
        targets: All targets defined on the campaign.
        target_names: Optional subset to restrict to.
        require_thresholds: If True, every selected target must have a
            threshold set (used by characterization / passfail / confidence).
        drop_tracking_only: If True, ``tracking_only`` targets are dropped
            when ``target_names`` is None and rejected when listed explicitly.

    Returns:
        Selected targets in the order they appear in ``targets``.
    """
    by_name = {t.name: t for t in targets}

    def _eligible(t):
        if drop_tracking_only and t.tracking_only:
            return False
        if require_thresholds and t.threshold is None:
            return False
        return True

    if target_names is None:
        selected = [t for t in targets if _eligible(t)]
        if not selected:
            raise ValueError(
                "No eligible targets available "
                f"(require_thresholds={require_thresholds}, drop_tracking_only={drop_tracking_only})."
            )
        return selected

    unknown = [n for n in target_names if n not in by_name]
    if unknown:
        raise ValueError(f"Unknown target name(s) {unknown}. Valid: {sorted(by_name)}")

    if drop_tracking_only:
        rejected = [n for n in target_names if by_name[n].tracking_only]
        if rejected:
            raise ValueError(f"tracking_only targets cannot be selected: {rejected}")

    if require_thresholds:
        missing = [n for n in target_names if by_name[n].threshold is None]
        if missing:
            raise ValueError(f"Thresholds are required for the selected targets; missing: {missing}")

    requested = set(target_names)
    return [t for t in targets if t.name in requested]


class Target:
    """
    Base class for optimization response targets.

    Args:
        name: Name of the target/response variable
        f_transform: Transform function to apply (default: "Standard")
        aim: Optimization direction - "max" or "min" (default: "max")
        tracking_only: If True, target is tracked but not optimized (default: False)
        threshold: Optional threshold value for characterization tasks.
            - If aim="max": characterize regions where response >= threshold
            - If aim="min": characterize regions where response <= threshold
    """

    def __init__(
        self,
        name: str,
        f_transform: str | None = "Standard",
        aim: str = "max",
        tracking_only: bool = False,
        threshold: float | None = None,
    ):
        self.name = name
        if aim not in ["min", "max"]:
            raise ValueError('Aim must be either "min" or "max"')
        if aim == "min":
            self.multiplier = -1
        else:
            self.multiplier = 1
        self.aim = aim
        self.tracking_only = tracking_only
        if threshold is not None and not isinstance(threshold, numbers.Number):
            raise TypeError(f"threshold must be None or a numeric scalar, got {type(threshold).__name__}")
        self.threshold = threshold

        # Output scoring, used for transformation OR to create a cost function of multiple outputs/inputs
        if f_transform is not None:
            if f_transform not in f_transform_dict.keys():
                raise KeyError(f"Scoring function must be selected from one of: {f_transform_dict.keys()}")
        else:
            f_transform = "Identity"
        self.f_transform = f_transform

    def __repr__(self):
        """String representation of object"""
        threshold_str = f", threshold={self.threshold}" if self.threshold is not None else ""
        return f"{self.__class__.__name__}({self.name}, aim={self.aim}{threshold_str})"

    def get_threshold(self, transformed: bool = True) -> float | None:
        """
        Get the threshold value, optionally transformed.

        Args:
            transformed: If True, apply the target's transform to the threshold (default: True)

        Returns:
            The threshold value (transformed or raw), or None if no threshold is set

        Raises:
            UnfitError: If transformed=True but the transform function hasn't been fit yet
        """
        if self.threshold is None:
            return None

        if transformed:
            if not hasattr(self, "f_transform_func"):
                raise UnfitError("Cannot transform threshold: transform function hasn't been fit yet.")
            return self.transform_f(self.threshold, inverse=False).iloc[0]
        else:
            return self.threshold

    def transform_f(self, f: float | int | ArrayLike, inverse=False, fit=False):
        """
        Converts a raw response to an objective function value ("score").
        Cost-penalization and response transformation should be handled here.

        Args:
            f (array-like): The column(s) containing the response values (y)
            inverse (bool, optional): An indicator to perform the inverse transform. Defaults to ``False``.
            fit (bool, optional): An indicator to fit the properties of the transform function. Defaults to ``False``.

        Returns:
            pd.Series: An array of transformed f values matching the responses in Z

        Raises:
            TypeError: If f is not numeric or array-like
            UnfitError: If the transform function is called without being fit first
        """

        f = to_tensor(f)

        if not fit:
            if not hasattr(self, "f_transform_func"):
                raise UnfitError("Transform function is being called without being fit first.")

        if f.ndim == 1:
            f = f.reshape(-1, 1)

        if inverse:
            f_obj = self.f_transform_func.inverse(f * self.multiplier)
            return pd.Series(f_obj.flatten(), name=self.name)
        else:
            if fit:
                self.f_transform_func = f_transform_dict[self.f_transform]()
                f_obj = self.f_transform_func(f, fit=True)
                self.f_raw = f  # Save raw data for re-loading and re-fitting state as needed
            else:
                f_obj = self.f_transform_func(f)
            return pd.Series(f_obj.flatten(), name=self.name + " Trans") * self.multiplier

    def save_state(self) -> dict:
        """
        Saves the state of the object as a dictionary.

        Returns:
            dict: A dictionary containing the state of the object.
        """
        # Prepare a dictionary to describe the state
        obj_dict = {"init_attrs": {}}

        # Select some optimizer attributes to save directly
        init_attrs = ["name", "aim", "f_transform", "tracking_only", "threshold"]
        for attr in init_attrs:
            obj_dict["init_attrs"][attr] = getattr(self, attr)

        # If the transformer has been fit, store the raw data so it can be refit upon load
        if hasattr(self, "f_transform_func"):
            obj_dict["f_raw"] = self.f_raw.tolist()

        return obj_dict

    @classmethod
    def load_state(cls, obj_dict: dict):
        """
        Loads the state of the target object from a dictionary.

        Args:
            cls (class): The class of the target object.
            obj_dict (dict): A dictionary containing the state of the target object.

        Returns:
            The loaded target object.
        """
        new_target = cls(**obj_dict["init_attrs"])

        # If the transformer has been fit before saving, refit it
        if "f_raw" in obj_dict:
            f = torch.Tensor(obj_dict["f_raw"])
            new_target.transform_f(f, fit=True)

        return new_target
