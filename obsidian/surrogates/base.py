"""Surrogate model class definition"""

from obsidian.config import TORCH_DTYPE

from abc import ABC, abstractmethod

import numpy as np
import pandas as pd
import torch
import random


class SurrogateModel(ABC):
    """
    The model used for conducting optimization. Consumes data and produces a regressed representation
    of that system. Model can then be used to make predictions or evaluate uncertainty.

    Attributes:
        is_fit (bool): Flag for fitting (e.g. to prevent predicting from an unfit model).
        model_type (str): The type of the model.
        train_X (pd.DataFrame): The input data for the training data.
        train_Y (pd.Series): The target data for the training data.
        cat_dims (list): The categorical dimensions of the data.
        task_feature (str): The task feature of the data.
        X_order (list): The order of the columns in the input data.
        y_name (str): The name of the target column.
        seed (int): Randomization seed for stochastic surrogate models.
        verbose (bool): Flag for monitoring and debugging optimization
    """
    def __init__(self,
                 model_type: str = 'GP',
                 seed: int | None = None,
                 verbose: bool = False):
        
        # Set a flag for fitting (e.g. to prevent predicting from an unfit model)
        self.is_fit = False

        # Set up initial model state
        self.model_type = model_type
        self.train_X = None
        self.train_Y = None
        self.cat_dims = None
        self.task_feature = None
        self.X_order = None
        self.y_name = None

        # Handle randomization seed, considering all 3 sources (torch, random, numpy)
        self.seed = seed
        if self.seed is not None:
            torch.manual_seed(self.seed)
            torch.use_deterministic_algorithms(True)
            np.random.seed(self.seed)
            random.seed(self.seed)

        # Verbose output
        self.verbose = verbose

        return

    def _validate_data(self,
                       X: pd.DataFrame,
                       y: pd.Series | None = None):
        """
        Validate the data properties, e.g. column order
        
        Args:
            X (pd.DataFrame): Input parameters for the training data.
            y (pd.Series): Training data responses.
        
        Raises:
            ValueError: If the data properties do not match the training data.
        """

        if X.columns.tolist() != self.X_order:
            raise ValueError('X columns do not match training data')
        if y is not None:
            if y.name != self.y_name:
                raise ValueError('y column does not match training data')
        return
    
    def _prepare(self,
                 X: pd.DataFrame,
                 y: pd.Series | None = None) -> tuple:
        """
        Converts X, Y into appropriate Tensor dtypes and shapes for torch_model
        
        Args:
            X (pd.DataFrame): The input data as a pandas DataFrame.
            Y (pd.Series): The target data as a pandas Series.
        
        Returns:
            tuple: A tuple containing the converted input data (X_torch) and target data (Y_torch) as torch Tensors.
        """
        self._validate_data(X, y)
        X_torch = torch.tensor(X.to_numpy(), dtype=TORCH_DTYPE)
        if y is not None:
            y_torch = torch.tensor(y.to_numpy(), dtype=TORCH_DTYPE).unsqueeze(-1)
            return (X_torch, y_torch)
        else:
            return X_torch

    @abstractmethod
    def fit(self,
            X: pd.DataFrame,
            y: pd.Series):
        """Fit the surrogate model to data"""
        pass  # pragma: no cover

    @abstractmethod
    def predict(self,
                X: pd.DataFrame):
        """Predict outputs based on candidates X"""
        pass  # pragma: no cover

    @abstractmethod
    def score(self,
              X: pd.DataFrame,
              y: pd.Series):
        """Score the model based on the given test data"""
        pass  # pragma: no cover
    
    @abstractmethod
    def save_state(self):
        """Save the model to a state dictionary"""
        pass  # pragma: no cover
    
    @classmethod
    @abstractmethod
    def load_state(cls,
                   obj_dict: dict):
        """Load the model from a state dictionary"""
        pass  # pragma: no cover
