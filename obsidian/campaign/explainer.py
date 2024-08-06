"""Explainer Class: Surrogate Model Interpretation Methods"""

import numpy as np
import pandas as pd
import torch

import matplotlib.pyplot as plt
from matplotlib.figure import Figure

from obsidian.parameters import Param_Continuous, ParamSpace
from obsidian.optimizer import Optimizer

import shap
from shap import KernelExplainer, Explanation
from obsidian.plotting.shap_helper import partial_dependence, one_shap_value

from obsidian.campaign.analysis import sensitivity


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
                 X_space: ParamSpace | None = None):
        
        if not optimizer.is_fit:
            raise ValueError('Surrogate model in optimizer is not fit to data. ')
        
        self.set_optimizer(optimizer)
        self.X_space = optimizer.X_space if X_space is None else X_space

        self.shap = {}
    
    def __repr__(self):
        return f"Explainer(optimizer={self.optimizer})"
    
    @property
    def optimizer(self):
        return self._optimizer
    
    def set_optimizer(self, optimizer: Optimizer):
        self._optimizer = optimizer
    
    def shap_explain(self,
                     responseid: int = 0,
                     n: int = 100,
                     X_ref: pd.DataFrame | None = None,
                     seed: int | None = None):
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
        
        def f_preds(X):
            # helper function for shap_explain
            if isinstance(X, np.ndarray):
                X = pd.DataFrame(X, columns=self.X_space.X_names)
                for p, x in zip(self.X_space, X.columns):
                    if isinstance(p, Param_Continuous):
                        X[x] = pd.to_numeric(X[x])
            y_pred = self.optimizer.predict(X, return_f_inv=True)
            mu = y_pred[y_name+' (pred)'].values
            return mu
        
        self.shap['f_preds'] = f_preds
        self.shap['explainer'] = KernelExplainer(f_preds,  X_ref)
        self.shap['X_sample'] = self.X_space.unit_demap(
            pd.DataFrame(torch.rand(size=(n, self.X_space.n_dim)).numpy(),
                         columns=X_ref.columns)
            )
        self.shap['values'] = self.shap['explainer'].shap_values(self.shap['X_sample'], seed=seed, l1_reg='aic')
        self.shap['explanation'] = Explanation(self.shap['values'], feature_names=X_ref.columns)

        return

    def shap_summary(self) -> Figure:
        if not self.shap:
            raise ValueError('shap explainer is not fit.')
        
        fig = shap.summary_plot(self.shap['values'], self.shap['X_sample'])
        return fig
    
    def shap_summary_bar(self) -> Figure:
        if not self.shap:
            raise ValueError('shap explainer is not fit.')
         
        fig = plt.figure()
        shap.plots.bar(self.shap['explanation'], ax=fig.gca(), show=False)
        plt.close(fig)
        
        return fig

    def shap_pdp_ice(self,
                     ind=0,  # var name or index
                     ice_color_var=0,  # var name or index
                     ace_opacity: float = 0.5,
                     ace_linewidth="auto"
                     ) -> Figure:
        
        if not self.shap:
            raise ValueError('shap explainer is not fit.')
        
        fig = partial_dependence(
                ind=ind,
                model=self.f_preds,
                data=self.X_sample,
                ice_color_var=ice_color_var,
                hist=False,
                ace_opacity=ace_opacity,
                ace_linewidth=ace_linewidth,
                show=False
            )
        
        return fig

    def shap_single_point(self,
                          X_new,
                          X_ref=None):
        if not self.shap:
            raise ValueError('shap explainer is not fit.')
        
        if X_ref is None:
            if self.shap_explainer is None:
                raise ValueError('shap explainer is not fit. ')
                return
            shap_value_new = self.shap_explainer.shap_values(X_new)
            expected_value = self.shap_explainer.expected_value
        else:
            # if another reference point is input, need to re-fit another explainer
            explainer = shap.KernelExplainer(self.f_preds,  X_ref)
            shap_value_new = explainer.shap_values(X_new)
            expected_value = explainer.expected_value
        
        shap_value_new = np.squeeze(shap_value_new)
        df_shap_value_new = pd.DataFrame([shap_value_new], columns=self.X_space.X_names)
        
        fig1, fig2 = one_shap_value(shap_value_new, expected_value, self.X_space.X_names)
        
        return df_shap_value_new, fig1, fig2

    def cal_sensitivity(self,
                        dx: float = 1e-6,
                        X_ref: pd.DataFrame | None = None) -> pd.DataFrame:
        
        df_sens = sensitivity(self.optimizer, dx=dx, X_ref=X_ref)
        
        return df_sens
