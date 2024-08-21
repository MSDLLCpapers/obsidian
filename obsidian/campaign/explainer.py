"""Explainer Class: Surrogate Model Interpretation Methods"""

from obsidian.parameters import Param_Continuous, ParamSpace
from obsidian.optimizer import Optimizer
from obsidian.exceptions import UnfitError

import shap
from shap import KernelExplainer, Explanation
from obsidian.plotting.shap import partial_dependence, one_shap_value

import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt
from matplotlib.figure import Figure


class Explainer():
    """
    Base class for surrogate model interpretation and post-hoc analysis.

    Properties:
        optimizer (Optimizer): obsidian Optimizer object with fitted surrogate model(s).

    Attributes:
        X_space (ParamSpace): obsidian ParamSpace object representing the allowable space for model explanation,
            could be different from the optimizer.X_space used for optimization.
        responseid (int): the index of a single outcome whose surrogate model will be explained by shap
        shap (dict): A dictionary of values containing shapley analysis results
    
    Raises:
        ValueError: If the optimizer is not fit
        
    """
    def __init__(self,
                 optimizer: Optimizer,
                 X_space: ParamSpace | None = None) -> None:
        
        if not optimizer.is_fit:
            raise UnfitError('Surrogate model in optimizer is not fit to data. ')
        
        self.set_optimizer(optimizer)
        self.X_space = optimizer.X_space if X_space is None else X_space

        self.shap = {}
    
    def __repr__(self) -> str:
        return f"Explainer(optimizer={self.optimizer})"
    
    @property
    def optimizer(self) -> Optimizer:
        """Explainer Optimizer object"""
        return self._optimizer
    
    def set_optimizer(self, optimizer: Optimizer) -> None:
        """Sets the explainer optimizer"""
        self._optimizer = optimizer
    
    def shap_explain(self,
                     responseid: int = 0,
                     n: int = 100,
                     X_ref: pd.DataFrame | None = None,
                     seed: int | None = None) -> None:
        """
        Explain the parameter sensitivities using shap values.

        Args:
            responseid (int): Index of the target response variable.
            n (int): Number of samples to generate for shap values.
            X_ref (pd.DataFrame | None): Reference DataFrame for shap values. If None,
                the mean of self.X_space will be used.
            seed (int | None): Seed value for random number generation.

        Returns:
            None: This function fits a Kernel shap explainer and save results as class attributes.

        Raises:
            ValueError: If X_ref does not contain all parameters in self.X_space or if X_ref is not a single row DataFrame.
        """
        
        self.shap['responseid'] = responseid
        
        if X_ref is None:
            X_ref = self.X_space.mean()
        else:
            if not all(x in X_ref.columns for x in self.X_space.X_names):
                raise ValueError('X_ref must contain all parameters in X_space')
            if X_ref.shape[0] != 1:
                raise ValueError('X_ref must be a single row DataFrame')
        
        self.shap['X_ref'] = X_ref
        
        y_name = self.optimizer.target[responseid].name
        
        def pred_func(X):
            # helper function for shap_explain
            if isinstance(X, np.ndarray):
                X = pd.DataFrame(X, columns=self.X_space.X_names)
                for p, x in zip(self.X_space, X.columns):
                    if isinstance(p, Param_Continuous):
                        X[x] = pd.to_numeric(X[x])
            y_pred = self.optimizer.predict(X, return_f_inv=True)
            mu = y_pred[y_name+' (pred)'].values
            return mu
        
        self.shap['pred_func'] = pred_func
        self.shap['explainer'] = KernelExplainer(pred_func,  X_ref)
        self.shap['X_sample'] = self.X_space.unit_demap(
            pd.DataFrame(torch.rand(size=(n, self.X_space.n_dim)).numpy(),
                         columns=X_ref.columns)
            )
        self.shap['values'] = self.shap['explainer'].shap_values(self.shap['X_sample'],
                                                                 seed=seed, l1_reg='aic')
        self.shap['explanation'] = Explanation(self.shap['values'], feature_names=X_ref.columns)

        return

    def shap_summary(self) -> Figure:
        """SHAP Summary Plot (Beeswarm)"""
        if not self.shap:
            raise UnfitError('SHAP explainer is not fit.')
        
        fig = plt.figure()
        shap.summary_plot(self.shap['values'], self.shap['X_sample'],
                          show=False)
        plt.close(fig)
        
        return fig
    
    def shap_summary_bar(self) -> Figure:
        """SHAP Summary Plot (Bar Plot / Importance)"""
        if not self.shap:
            raise UnfitError('SHAP explainer is not fit.')
         
        fig = plt.figure()
        shap.plots.bar(self.shap['explanation'],
                       ax=fig.gca(), show=False)
        plt.close(fig)
        
        return fig

    def shap_pdp_ice(self,
                     ind: int | tuple[int] = 0,
                     ice_color_var: int | None = None,
                     ace_opacity: float = 0.5,
                     npoints: int | None = None,
                     ) -> Figure:
        """
        SHAP Partial Dependence Plot with ICE
        
        Args:
            ind (int): Index of the parameter to plot
            ice_color_var (int): Index of the parameter to color the ICE lines
            ace_opacity (float): Opacity of the ACE line
            npoints (int, optional): Number of points for PDP x-axis. By default
                will use ``100`` for 1D PDP and ``20`` for 2D PDP.
        
        Returns:
            Matplotlib Figure of 1D or 2D PDP with ICE lines
                    
        """
        if not self.shap:
            raise UnfitError('SHAP explainer is not fit.')
        
        fig, ax = partial_dependence(
                ind=ind,
                model=self.shap['pred_func'],
                data=self.shap['X_sample'],
                ice_color_var=ice_color_var,
                hist=False,
                ace_opacity=ace_opacity,
                show=False,
                npoints=npoints
                )
        plt.close(fig)
        
        return fig

    def shap_single_point(self,
                          X_new: pd.DataFrame | pd.Series,
                          X_ref=None) -> tuple[pd.DataFrame, Figure, Figure]:
        """
        SHAP Pair-wise Marginal Explanations
        
        Args:
            X_new (pd.DataFrame | pd.Series): New data point to explain
            X_ref (pd.DataFrame | pd.Series, optional): Reference data point
                for shap values. Default uses ``X_space.mean()``
        
        Returns:
            pd.DataFrame: DataFrame containing SHAP values for the new data point
            Figure: Matplotlib Figure for SHAP values
            Figure: Matplotlib Figure for SHAP summary plot
        
        """
        if not self.shap:
            raise UnfitError('SHAP explainer is not fit.')

        if isinstance(X_new, pd.Series):
            X_new = X_new.copy().to_frame().T
        if isinstance(X_ref, pd.Series):
            X_ref = X_ref.copy().to_frame().T
            
        if not list(X_new.columns) == list(self.optimizer.X_space.X_names):
            raise ValueError('X_new must contain all parameters in X_space')
        
        if X_ref is None:
            shap_value_new = self.shap['explainer'].shap_values(X_new).squeeze()
            expected_value = self.shap['explainer'].expected_value
        else:
            if not list(X_ref.columns) == list(self.optimizer.X_space.X_names):
                raise ValueError('X_ref must contain all parameters in X_space')
            
            # if another reference point is input, need to re-fit another explainer
            explainer = shap.KernelExplainer(self.shap['pred_func'],  X_ref)
            shap_value_new = explainer.shap_values(X_new).squeeze()
            expected_value = explainer.expected_value
        
        df_shap_value_new = pd.DataFrame([shap_value_new], columns=self.X_space.X_names)
        
        fig_bar, fig_line = one_shap_value(shap_value_new, expected_value, self.X_space.X_names)
        
        return df_shap_value_new, fig_bar, fig_line

    def sensitivity(self,
                    dx: float = 1e-6,
                    X_ref: pd.DataFrame | pd.Series | None = None) -> pd.DataFrame:
        """
        Calculates the local sensitivity of the surrogate model predictions with
        respect to each parameter in the X_space.

        Args:
            optimizer (BayesianOptimizer): The optimizer object which contains a surrogate
                that has been fit to data
            and can be used to make predictions.
            dx (float, optional): The perturbation size for calculating the sensitivity.
                Defaults to ``1e-6``.
            X_ref (pd.DataFrame | pd.Series | None, optional): The reference input values for
                calculating the sensitivity.  If None, the mean of X_space will be used as the
                reference. Defaults to ``None``.

        Returns:
            pd.DataFrame: A DataFrame containing the sensitivity values for each parameter
                in X_space.

        Raises:
            ValueError: If X_ref does not contain all parameters in optimizer.X_space or if
                X_ref is not a single row DataFrame.
                
        """
        
        if isinstance(X_ref, pd.Series):
            X_ref = X_ref.copy().to_frame().T
        
        if X_ref is None:
            X_ref = self.optimizer.X_space.mean()
        else:
            if not all(x in X_ref.columns for x in self.optimizer.X_space.X_names):
                raise ValueError('X_ref must contain all parameters in X_space')
            if X_ref.shape[0] != 1:
                raise ValueError('X_ref must be a single row DataFrame')
        
        y_ref = self.optimizer.predict(X_ref)
        
        sens = {}
        
        # Only do positive perturbation, for simplicity
        for param in self.optimizer.X_space:
            base = param.unit_map(X_ref[param.name].values)[0]
            # Space already mapped to (0,1), use absolute perturbation
            dx_pos = np.array(base+dx).reshape(-1, 1)
            X_sim = X_ref.copy()
            X_sim[param.name] = param.unit_demap(dx_pos)[0]
            y_sim = self.optimizer.predict(X_sim)
            dydx = (y_sim - y_ref)/dx
            sens[param.name] = dydx.to_dict('records')[0]
        
        df_sens = pd.DataFrame(sens).T[[y+' (pred)' for y in self.optimizer.y_names]]
        
        return df_sens
