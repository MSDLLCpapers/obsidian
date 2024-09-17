"""Custom implementations of PyTorch surrogate models using BoTorch API"""

from .utils import fit_pytorch

from obsidian.config import TORCH_DTYPE

from botorch.models.model import FantasizeMixin
from botorch.models.ensemble import EnsembleModel, Model
from botorch.posteriors.ensemble import Posterior, EnsemblePosterior

import torch
import torch.nn as nn
from torch.nn import Module
from torch import Tensor

from typing import TypeVar
TFantasizeMixin = TypeVar("TFantasizeMixin", bound="FantasizeMixin")


class DNNPosterior(EnsemblePosterior):
    
    def __init__(self, values: Tensor):
        super().__init__(values)
         
    def quantile(self, value: Tensor) -> Tensor:
        """Quantile of the ensemble posterior"""
        return self.values.quantile(q=value.to(self.values), dim=-3, interpolation='linear')


class DNN(EnsembleModel, FantasizeMixin):
    def __init__(self,
                 train_X: Tensor,
                 train_Y: Tensor,
                 p_dropout: float = 0.2,
                 h_width: int = 16,
                 h_layers: int = 2,
                 num_outputs: int = 1):
        
        super().__init__()
        
        if h_layers < 1:
            raise ValueError("h_layers must be at least 1")
        if p_dropout < 0 or p_dropout > 1:
            raise ValueError("p_dropout must be in [0, 1]")
        
        self.register_buffer('train_X', train_X)
        self.register_buffer('train_Y', train_Y)
        self.register_buffer('p_dropout', torch.tensor(p_dropout, dtype=TORCH_DTYPE))
        self.register_buffer('h_width', torch.tensor(h_width, dtype=torch.int))
        self.register_buffer('h_layers', torch.tensor(h_layers, dtype=torch.int))
        self.register_buffer('num_outputs', torch.tensor(num_outputs, dtype=torch.int))

        self.input_layer = nn.Sequential(
            nn.Linear(train_X.shape[-1], h_width),
            nn.PReLU(),
            nn.Dropout(p=p_dropout)
        )

        self.middle_layers = nn.Sequential(
            *[nn.Sequential(
                nn.Linear(h_width, h_width),
                nn.PReLU(),
                nn.Dropout(p=p_dropout)
            ) for _ in range(h_layers)]
        )
        
        self.outer_layer = nn.Linear(h_width, num_outputs)
        self._num_outputs = num_outputs
        self.to(TORCH_DTYPE)
        
    def forward(self,
                x: Tensor) -> Tensor:
        x = self.input_layer(x)
        x = self.middle_layers(x)
        x = self.outer_layer(x)
        """Evaluate the forward pass of the model on inputs X"""
        return x

    def posterior(self,
                  X: Tensor,
                  n_sample: int = 512,
                  output_indices: list[int] = None,
                  observation_noise: bool | Tensor = False) -> Posterior:
        """Calculates the posterior distribution of the model at X"""
        if not output_indices:
            output_indices = list(range(self._num_outputs))
        elif not all(0 <= i < self._num_outputs for i in output_indices):
            raise ValueError("Invalid output index")
        
        if X.ndim == 2:
            X_sample = X.unsqueeze(0).repeat_interleave(n_sample, 0)
        else:  # Account for possible batch dimension
            X_sample = X.unsqueeze(1).repeat_interleave(n_sample, 1)
            
        self.train()
        y_out = self.forward(X_sample)
        self.eval()
        
        post = DNNPosterior(y_out[..., output_indices])
        
        return post
    
    @property
    def num_outputs(self) -> int:
        """Number of outputs of the model"""
        return self._num_outputs
    
    def transform_inputs(self,
                         X: Tensor,
                         input_transform: Module = None) -> Tensor:
        """
        Transform inputs.

        Args:
            X: A tensor of inputs
            input_transform: A Module that performs the input transformation.

        Returns:
            A tensor of transformed inputs
        """
        if input_transform is not None:
            input_transform.to(X)
            return input_transform(X)
        try:
            return self.input_transform(X)
        except AttributeError:
            return X

    def condition_on_observations(self,
                                  X: Tensor,
                                  Y: Tensor) -> TFantasizeMixin:
        """
        Condition the model to new observations, returning a fantasy model
        """

        X_c = torch.concat((self.train_X, X), axis=0)
        Y_c = torch.concat((self.train_Y, Y), axis=0)

        # Create a new model based on the current one
        fantasy = self.__class__(train_X=X_c, train_Y=Y_c,
                                 p_dropout=float(self.p_dropout),
                                 h_width=int(self.h_width), h_layers=int(self.h_layers),
                                 num_outputs=int(self.num_outputs))

        # Fit to the new data
        fit_pytorch(fantasy, X_c, Y_c)

        return fantasy
    
    def fantasize(self,
                  X: Tensor) -> Model:
        
        Y_f = self.forward(X).detach()
        fantasy = self.condition_on_observations(X, Y_f)

        return fantasy
