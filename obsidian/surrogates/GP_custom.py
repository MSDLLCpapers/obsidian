"""Custom implementations of Gaussian Process models using BoTorch API"""

import torch
from torch import nn
import gpytorch

from obsidian.utils import TORCH_DTYPE

from botorch.models.gpytorch import GPyTorchModel
from gpytorch.distributions import MultivariateNormal
from gpytorch.likelihoods import GaussianLikelihood

from gpytorch.means import ConstantMean
from gpytorch.models import ExactGP

from gpytorch.kernels import ScaleKernel, MaternKernel
from gpytorch.priors import GammaPrior
from gpytorch.constraints import GreaterThan


class PriorGP(ExactGP, GPyTorchModel):
    """
    Class which builds a GP with custom prior distributions;
    by default set to the values of BoTorch SingleTaskGP
    """
    _num_outputs = 1

    def __init__(self, train_X, train_Y):

        # Prior on the noise (custom likelihood)
        noise_prior = GammaPrior(1.1, 0.05)
        noise_prior_mode = (noise_prior.concentration - 1) / noise_prior.rate
        likelihood = GaussianLikelihood(
                noise_prior=noise_prior,
                noise_constraint=GreaterThan(
                    1e-4,
                    transform=None,
                    initial_value=noise_prior_mode,
                ),
            )
        
        super().__init__(train_X, train_Y.squeeze(-1), likelihood)
        
        n_dims = train_X.shape[-1]
        
        self.mean_module = ConstantMean()
        self.covar_module = ScaleKernel(
            base_kernel=MaternKernel(nu=2.5, ard_num_dims=n_dims,
                                     lengthscale_prior=GammaPrior(3.0, 6.0)),
            outputscale_prior=GammaPrior(2.0, 0.15),
        )
        
    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return MultivariateNormal(mean_x, covar_x)


class FlatGP(ExactGP, GPyTorchModel):
    """
    GP Surrogate with non-informative or no prior distributions
    """
    _num_outputs = 1

    def __init__(self, train_X, train_Y, nu=2.5):

        super().__init__(train_X, train_Y.squeeze(-1), GaussianLikelihood())
        
        n_dims = train_X.shape[-1]
        
        self.mean_module = ConstantMean()
        self.covar_module = ScaleKernel(
            base_kernel=MaternKernel(nu=nu, ard_num_dims=n_dims),
        )
        
    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return MultivariateNormal(mean_x, covar_x)


class DKLGP(ExactGP, GPyTorchModel):
    """
    GP surrogate with a FF NN feature extractor
    """
    _num_outputs = 1

    def __init__(self, train_X, train_Y):

        super().__init__(train_X, train_Y.squeeze(-1), GaussianLikelihood())
        
        n_dims = train_X.shape[-1]
        
        self.mean_module = ConstantMean()
        self.covar_module = ScaleKernel(
            base_kernel=MaternKernel(nu=2.5, ard_num_dims=n_dims),
        )
        
        # Set up the NN feature extractor
        torch.set_default_dtype(TORCH_DTYPE)
        self.feature_extractor = nn.Sequential(
            # Hiden layer 1
            nn.Linear(n_dims, n_dims),
            nn.PReLU(),
            # Hidden layer 2
            nn.Linear(n_dims, n_dims),
            nn.PReLU(),
            # #Output layer
            nn.Linear(n_dims, n_dims)
            )
        
        # Use a scaler to make sure the GP only sees well-conditioned values
        self.scale_to_bounds = gpytorch.utils.grid.ScaleToBounds(0, 1)

    def forward(self, x):

        # Pass the data through the feature extractor and scaler
        projected_x = self.feature_extractor(x)
        projected_x = self.scale_to_bounds(projected_x)
        
        mean_x = self.mean_module(projected_x)
        covar_x = self.covar_module(projected_x)
        return MultivariateNormal(mean_x, covar_x)
