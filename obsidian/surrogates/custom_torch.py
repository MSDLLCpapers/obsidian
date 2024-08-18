"""Custom implementations of PyTorch surrogate models using BoTorch API"""

from botorch.models.model import Model
from botorch.posteriors.ensemble import Posterior, EnsemblePosterior

import torch.nn as nn
from torch import Tensor


class DNNPosterior(EnsemblePosterior):
    
    def __init__(self, values: Tensor):
        super().__init__(values)
         
    def quantile(self, value: Tensor) -> Tensor:
        """Quantile of the ensemble posterior"""
        return self.values.quantile(q=value.to(self.values), dim=-3, interpolation='linear')


class DNN(Model):
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
        
    def forward(self,
                x: Tensor) -> Tensor:
        x = self.input_layer(x)
        x = self.middle_layers(x)
        x = self.outer_layer(x)
        """Evaluate the forward pass of the model on inputs X"""
        return x

    def posterior(self,
                  X: Tensor,
                  n_sample: int = 16384,
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
