"""Surrogate models built using BoTorch API and torch_model objects"""

from .base import SurrogateModel
from .config import model_class_dict

from obsidian.utils import tensordict_to_dict, dict_to_tensordict
from obsidian.exceptions import SurrogateFitError
from obsidian.config import TORCH_DTYPE

from botorch.fit import fit_gpytorch_mll
from botorch.optim.fit import fit_gpytorch_mll_torch, fit_gpytorch_mll_scipy
from botorch.models.gpytorch import GPyTorchModel
from gpytorch.mlls import ExactMarginalLogLikelihood

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pandas as pd
import warnings


class SurrogateBoTorch(SurrogateModel):
    """
    BoTorch GP model, subclass of the obsidian SurrogateModel

    Attributes:
        model_type (str): The type of model to be used.
        
            Defaults to ``'GP'``. Options are as follows:
        
            - ``'GP'``: Gaussian Process with default settings (Matern Kernel, Gamma covariance priors)
            - ``'MixedGP'``: GP with mixed parameter types (continuous, categorical). Will be re-selected
                by default if 'GP' is selected and input space is mixed.
            - ``'DKL'``: GP with a NN feature-extractor (deep kernel learning)
            - ``'GPflat'``: GP without priors. May result in optimization instability, but removes bias
                for special situations.
            - ``'GPprior'``: GP with custom priors on the mean, likelihood, and covariance
            - ``'MTGP'``: Multi-task GP for multi-output optimization. Will be re-selected by default
                if 'GP' is selected and the input space contains Task parameters.
            - ``'DNN'``: Dropout neural network. Uses MC sampling to mask neurons during training and
                to estimate uncertainty.
              
        device (str): The device on which the model is run.
        hps (dict): Optional surrogate function hyperparameters.
        mll (ExactMarginalLogLikelihood): The marginal log likelihood of the model.
        torch_model (torch.nn.Module): The torch model for the surrogate.
        loss (float): The loss of the model.
        score (float): The R2 score of the model.
    """
    def __init__(self,
                 model_type: str = 'GP',
                 seed: int | None = None,
                 verbose: int | None = False,
                 hps: dict = {}):
        
        super().__init__(model_type=model_type, seed=seed, verbose=verbose)
        
        if torch.cuda.is_available():
            self.device = 'cuda'
        else:
            self.device = 'cpu'

        # Optional surrogate function hyperparameters
        self.hps = hps
                
        return
    
    def init_model(self,
                   X: pd.DataFrame,
                   y: pd.Series,
                   cat_dims: list[int],
                   task_feature: int):
        """
        Instantiates the torch model for the surrogate.
        Cannot be called during __init__ normally as X,y are required
        and may not be available until fit methods are called

        Args:
            X (pd.DataFrame): Input parameters for the training data.
            y (pd.Series): Training data responses.
            cat_dims (list): A list of indices for categorical dimensions in the input data.

        Returns:
            None. Updates surrogate attributes, including self.torch_model.
        
        Raises:
            TypeError: If cat_dims is not a list of integers.
            
        """

        # Once the model is created, steward the order of data columns
        self.X_order = X.columns.tolist()
        self.y_name = y.name

        X_p, y_p = self._prepare(X, y)
    
        if not isinstance(cat_dims, list):
            raise TypeError('cat_dims must be a list')
        if not all([isinstance(c, int) for c in cat_dims]):
            raise TypeError('cat_dims must be a list of integers')

        if issubclass(model_class_dict[self.model_type], GPyTorchModel):
            if self.model_type == 'GP' and cat_dims:  # If cat_dims is not an empty list, returns True
                self.torch_model = model_class_dict['MixedGP'](train_X=X_p, train_Y=y_p, cat_dims=cat_dims).to(self.device)
            else:
                if self.model_type == 'MTGP':
                    self.torch_model = model_class_dict[self.model_type](
                        train_X=X_p, train_Y=y_p, task_feature=task_feature, **self.hps).to(self.device)
                else:
                    # Note: Doesn't matter if input empty dictionary as self.hps for model without those additional args
                    self.torch_model = model_class_dict[self.model_type](train_X=X_p, train_Y=y_p, **self.hps).to(self.device)
        else:
            self.torch_model = model_class_dict[self.model_type](train_X=X_p, train_Y=y_p, **self.hps).to(self.device).to(TORCH_DTYPE)

        return

    def fit(self,
            X: pd.DataFrame,
            y: pd.Series,
            cat_dims=None,
            task_feature=None):
        """
        Fits the surrogate model to data

        Args:
            X (pd.DataFrame): Input parameters for the training data
            y (pd.Series): Training data responses
            cat_dims (list, optional): A list of indices for categorical dimensions in the input data. Default is ``None``.

        Returns:
            None. Updates the surrogate model attributes, including regressed parameters.
        """
     
        # Instantiate self.torch_model
        self.init_model(X, y, cat_dims, task_feature)
        X_p, y_p = self._prepare(X, y)

        # Save the raw data inputs into the model metadata
        self.train_X = X
        self.train_Y = y
        self.cat_dims = cat_dims
        self.task_feature = task_feature
        
        # Train
        if isinstance(self.torch_model, GPyTorchModel):
            self.loss_fcn = ExactMarginalLogLikelihood(self.torch_model.likelihood, self.torch_model)
            if self.model_type == 'DKL':
                optimizer = fit_gpytorch_mll_torch
            else:
                optimizer = fit_gpytorch_mll_scipy

            try:
                fit_gpytorch_mll(self.loss_fcn, optimizer=optimizer)
            except Exception:
                try:
                    fit_gpytorch_mll(self.loss_fcn, optimizer=fit_gpytorch_mll_torch)
                except Exception:
                    raise SurrogateFitError('BoTorch model failed to fit')
        else:
            self.loss_fcn = nn.MSELoss()
            self.optimizer = optim.Adam(self.torch_model.parameters(), lr=1e-2)

            self.torch_model.train()
            for epoch in range(200):
                self.optimizer.zero_grad()
                output = self.torch_model(X_p)
                loss = self.loss_fcn(output, y_p)
                loss.backward()
                self.optimizer.step()

            self.torch_model.eval()

        self.is_fit = True

        # Quick evaluation
        self.score(X, y)
        
        return
    
    def score(self,
              X: pd.DataFrame,
              y: pd.Series) -> tuple:
        """
        Computes simple model statistics on a dataset.

        Args:
            X (pd.DataFrame): Input parameters for the evaluation data.
            y (pd.DataFrame): Evaluation data responses.

        Returns:
            tuple: A tuple containing the loss and R2 score of the evaluation.
        """

        X_p, y_p = self._prepare(X, y)
        
        if self.model_type == 'DKL':
            self.torch_model(X_p)
        
        # Quick evaluation
        # Suppress an error that will be generated when evaluating R2_training
        warnings.filterwarnings('ignore', message=r'[.\n]*The input matches the stored training data*')
        mu_pred, _ = self.predict(X)
        corr_matrix = np.corrcoef(mu_pred.detach().cpu().flatten(),
                                  y_p.detach().cpu().flatten())
        
        # Calculate a final loss and R2 train score
        loss = self.loss = self.loss_fcn(self.torch_model(X_p), y_p).sum().detach().cpu().data.numpy().tolist()
        score = self.score = corr_matrix[0][1]**2
        
        return loss, score
    
    def predict(self,
                X: pd.DataFrame,
                q: float | None = None):
        """
        Computes model predictions over new experimental space.

        Args:
            X (pd.DataFrame): Input parameters for the prediction space.
            q (float, op)

        Returns:
            tuple: A tuple containing:
                - mu (ndarray): Mean responses for each experiment in the prediction space.
                - q_pred | sd (ndarray): Quantile or standard deviation of the predicted response for each experiment.
        """
        
        X_p = self._prepare(X)
        
        pred_posterior = self.torch_model.posterior(X_p)
        mu = pred_posterior.mean.detach().cpu().squeeze(-1)
        if q is not None:
            if (q < 0) or (q > 1):
                raise ValueError('Quantile must be between 0 and 1')
            q_pred = pred_posterior.quantile(torch.tensor(q)).detach().cpu().squeeze(-1)
            return mu, q_pred
        else:
            sd = pred_posterior.variance.detach().cpu().squeeze(-1)**0.5
            return mu, sd

    def save_state(self) -> dict:
        """
        Saves the state of the SurrogateBoTorch model.

        Returns:
            dict: A dictionary containing the state of the SurrogateBoTorch model.
        """
        obj_dict = {'model_attrs': {}}

        model_attrs = ['model_type', 'seed', 'hps',
                       'is_fit',
                       'cat_dims', 'task_feature',
                       'X_order', 'y_name']
        for attr in model_attrs:
            obj_dict['model_attrs'][attr] = getattr(self, attr)
 
        obj_dict['train_X'] = self.train_X.to_dict()
        obj_dict['train_Y'] = self.train_Y.to_dict()

        # Use torch's built in state_dict function
        obj_dict['torch_params'] = tensordict_to_dict(self.torch_model.state_dict())

        return obj_dict
    
    @classmethod
    def load_state(cls,
                   obj_dict: dict):
        """
        Generates a BoTorch surrogate model from a state dictionary (saved after fitting).
        Note that the training data is necessary to instantiate a matching model,
        but no fitting occurs here.

        Args:
            obj_dict (OrderedDict): The state dictionary of a previously fit BoTorch model.

        Returns:
            None. Updates the parameters of the model.
        """

        new_model = cls(model_type=obj_dict['model_attrs']['model_type'],
                        seed=obj_dict['model_attrs']['seed'], verbose=False,
                        hps=obj_dict['model_attrs']['hps'])

        # Directly unpack all of the entries in model_attrs
        for k, v in obj_dict['model_attrs'].items():
            setattr(new_model, k, v)

        # Load data objects
        new_model.train_X = pd.DataFrame(obj_dict['train_X'], columns=new_model.X_order)
        new_model.train_Y = pd.Series(obj_dict['train_Y'], name=new_model.y_name)
 
        # Re-initialize torch model
        new_model.init_model(new_model.train_X, new_model.train_Y, new_model.cat_dims, new_model.task_feature)

        # Load saved parameters into the torch model
        new_model.torch_model.load_state_dict(dict_to_tensordict(obj_dict['torch_params']))

        # Rebuild MLL which would have been done during fitting
        new_model.torch_model.training = False
        if isinstance(new_model.torch_model, GPyTorchModel):
            new_model.loss_fcn = ExactMarginalLogLikelihood(new_model.torch_model.likelihood, new_model.torch_model)
        else:
            new_model.loss_fcn = nn.MSELoss()
        new_model.score(new_model.train_X, new_model.train_Y)
        
        return new_model
