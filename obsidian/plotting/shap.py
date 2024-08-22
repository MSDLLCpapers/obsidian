"""Custom plots for SHAP analysis visualization"""

from .branding import obsidian_cm, obsidian_colors
from obsidian.exceptions import UnsupportedError

from shap.plots._partial_dependence import compute_bounds

import matplotlib.pyplot as plt
from matplotlib.axes import Axes
from matplotlib.figure import Figure

import numpy as np
import pandas as pd
from pandas.api.types import is_numeric_dtype

from typing import Callable


def one_shap_value(shap_value_new: np.ndarray,
                   expected_value: float,
                   X_names: list[str]) -> tuple[Figure, Figure]:
    """
    Visualize the shap values of one data point
    
    Args:
        shap_value_new (np.ndarray): The SHAP values of a single data point
            to be compared to a reference point.
        expected_value (float): The expected value at the reference point.
        X_names (list[str]): The names of the features.
        
    Returns:
        Figure: The bar plot of SHAP values for the single data point.
        Figure: The line plot of cumulative SHAP values for the data point in
            comparison to the reference point.
            
    """
    
    # First figure = Bar plot, just SHAP values of the new point
    pred_new = expected_value + np.sum(shap_value_new)
    
    fig_bar = plt.figure(figsize=(8, 6))
    plt.bar(range(len(shap_value_new)), shap_value_new, color=obsidian_colors.primary.teal)
    
    plt.xticks(range(len(shap_value_new)), X_names)
    plt.ylabel('SHAP Value')
    plt.xlabel('Feature')
    plt.title('SHAP Values for Single Datapoint')
    plt.axhline(0, color='grey', linestyle='--')
    plt.close(fig_bar)

    # Second figure = Line plot, cumulative SHAP values from new point to reference
    fig_line = plt.figure(figsize=(8, 6))
    
    cumulative_shap = np.cumsum(shap_value_new)
    y = cumulative_shap+expected_value
    n = len(cumulative_shap)
    
    plt.plot(range(n), y, color=obsidian_colors.primary.teal)
    plt.scatter(range(n), y, color=obsidian_colors.secondary.light_teal)
    
    plt.vlines(0, expected_value, y[0], colors='steelblue', linestyles='solid')
    plt.scatter(0, expected_value, color=obsidian_colors.accent.vista_blue)
    plt.scatter(n-1, y[-1], color=obsidian_colors.secondary.blue)
    
    plt.xticks(range(n), X_names)
    plt.xlabel('Feature')
    plt.ylabel('Cumulative SHAP Value')
    plt.title('Cumulative SHAP Values for Single Datapoint')
    plt.axhline(expected_value, color=obsidian_colors.accent.vista_blue,
                linestyle='--', label='Reference')
    plt.axhline(pred_new, color=obsidian_colors.secondary.blue,
                linestyle='--', label='New Data')
    plt.ylim([min(min(y), expected_value)*0.98, max(max(y), expected_value)*1.02])
    plt.legend()
    plt.close(fig_line)
    
    return fig_bar, fig_line


def partial_dependence(ind: int | tuple[int],
                       model: Callable,
                       data: pd.DataFrame,
                       ice_color_var: int | None = None,
                       xmin: str | tuple[float] | float = "percentile(0)",
                       xmax: str | tuple[float] | float = "percentile(100)",
                       npoints: int | None = None,
                       hist: bool = False,
                       ylabel: str | None = None,
                       ice: bool = True,
                       ace_opacity: float = 1,
                       pd_opacity: float = 1,
                       pd_linewidth: float = 2,
                       ace_linewidth: str | float = 'auto',
                       ax: Axes | None = None,
                       show: bool = True) -> Figure:
    """
    Calculates and plots the partial dependence of a feature or a pair of features on the model's output.

    This function is revised from the partial_dependence_plot function in shap package,
    in order to color the ICE curves by certain feature for checking interaction between features.
    Ref: https://github.com/shap/shap/blob/master/shap/plots/_partial_dependence.py

    Args:
        ind (int | tuple): The index or indices of the feature(s) to calculate
            the partial dependence for.
        model (Callable): The model used for prediction.
        data (pd.DataFrame): The input data used for prediction.
        ice_color_var (int, optional): The index of the feature used for coloring
            the ICE lines (for 1D partial dependence plot). Default is ``0``.
        xmin (str | tuple | float, optional): The minimum value(s) for the feature(s) range.
            Default is ``"percentile(0)"``.
        xmax (str | tuple | float): The maximum value(s) for the feature(s) range.
            Default is ``"percentile(100)"``.
        npoints (int, optional): The number of points to sample within the feature(s) range.
            By default, will use ``100`` points for 1D PDP and ``20`` points for 2D PDP.
        hist (bool, optional): Whether to plot the histogram of the feature(s). Default
            is ``False``.
        ylabel (str, optional): The label for the y-axis. Default is ``None``.
        ice (bool, optional): Whether to plot the Individual Conditional Expectation (ICE) lines.
            Default is ``True``.
        ace_opacity (float, optional): The opacity of the ACE lines. Default is ``1``.
        pd_opacity (float, optional): The opacity of the PDP line. Default is ``1``.
        pd_linewidth (float, optional): The linewidth of the PDP line. Default is ``2``.
        ace_linewidth (float | str, optional): The linewidth of the ACE lines. Default is ``'auto'``
            for automatic calculation.
        ax (Axes, optional): The matplotlib axis to plot on. By default will attach to Figure.gca().
        show (bool, optional): Whether to show the plot. Default is ``True``.

    Returns:
        tuple: A tuple containing the matplotlib figure and axis objects if `show` is False, otherwise None.
    """

    def cast_to_float(x: np.ndarray) -> np.ndarray:
        """
        Attempts to cast array to float type
        """
        try:
            return x.astype(float)
        except ValueError:
            return x

    # Extract vals, names from data
    df_features = data
    feature_names = df_features.columns

    # 1D PDP
    if not isinstance(ind, tuple):
        # xv = values, independent variable
        # xs = scaled values
        xv = df_features.iloc[:, ind]
        df_test = df_features.copy()
        if not is_numeric_dtype(xv):
            xs = np.array(sorted(set(xv)))
            npoints = len(xs)
            xmin = xs[0]
            xmax = xs[-1]
        else:
            xmin, xmax = compute_bounds(xmin, xmax, xv)
            npoints = 100 if npoints is None else npoints
            xs = np.linspace(xmin, xmax, npoints).astype(xv.dtype)
        if ice:
            df_test = df_features.copy()
            ice_vals = []
            for i in range(npoints):
                df_test.iloc[:, ind] = xs[i]
                ice_vals.append(model(df_test))
            ice_vals = np.array(ice_vals)
        
        vals = []
        for i in range(npoints):
            df_test.iloc[:, ind] = xs[i]
            vals.append(model(df_test).mean())
        vals = np.array(vals)
        
        if ax is None:
            fig = plt.figure()
            ax1 = plt.gca()
        else:
            fig = plt.gcf()
            ax1 = plt.gca()

        ax2 = ax1.twinx()

        # Histogram
        if hist:
            ax2.hist(sorted(xv), 50, density=False, facecolor='black', alpha=0.1)

        # ICE line plot
        if ice:
            if ace_linewidth == "auto":
                ace_linewidth = min(1, 50/ice_vals.shape[1])
            if ice_color_var is None:
                ax1.plot(xs, ice_vals, color=obsidian_colors.secondary.light_teal,
                         linewidth=ace_linewidth, alpha=ace_opacity)
            else:
                if ice_color_var == ind:
                    raise UnsupportedError("Coloring by the feature(s) used in the PDP is not supported.")
                colormap = obsidian_cm.obsidian_viridis
                xc = df_features.iloc[:, ice_color_var]
                
                if not is_numeric_dtype(xc):
                    xc_cat_vals = sorted(set(xc))
                    colorbar_min = 0
                    colorbar_max = len(xc_cat_vals) - 1
                    color_vals = [xc_cat_vals.index(c)/colorbar_max for c in xc]
                else:
                    color_vals = xc
                    colorbar_min = color_vals.min()
                    colorbar_max = color_vals.max()
                    color_vals = ((color_vals - colorbar_min)/(colorbar_max-colorbar_min))
                    
                for i in range(ice_vals.shape[1]):
                    ax1.plot(xs, ice_vals[:, i], color=colormap(color_vals[i]),
                             linewidth=ace_linewidth, alpha=ace_opacity)
                cbar = plt.colorbar(plt.cm.ScalarMappable(cmap=colormap,
                                                          norm=plt.Normalize(
                                                              vmin=colorbar_min, vmax=colorbar_max)),
                                    ax=ax1)
                if not is_numeric_dtype(xc):
                    cbar.set_ticks(np.linspace(0, colorbar_max, len(xc_cat_vals)))
                    cbar.set_ticklabels(xc_cat_vals)
                cbar.set_label('Color by ' + feature_names[ice_color_var])

        ax1.plot(xs, vals, color="black", linewidth=pd_linewidth, alpha=pd_opacity)

        ax2.set_ylim(0, df_features.shape[0])
        ax1.set_xlabel(feature_names[ind], fontsize=13)
        if ylabel is None:
            if not ice:
                ylabel = "E[f(x) | " + str(feature_names[ind]) + "]"
            else:
                ylabel = "f(x) | " + str(feature_names[ind])

        ax1.set_ylabel(ylabel, fontsize=13)
        ax1.xaxis.set_ticks_position('bottom')
        ax1.yaxis.set_ticks_position('left')
        ax1.spines['right'].set_visible(False)
        ax1.spines['top'].set_visible(False)
        ax1.tick_params(labelsize=11)

        ax2.xaxis.set_ticks_position('bottom')
        ax2.yaxis.set_ticks_position('left')
        ax2.yaxis.set_ticks([])
        ax2.spines['right'].set_visible(False)
        ax2.spines['top'].set_visible(False)
        ax2.spines['left'].set_visible(False)
        ax2.spines['bottom'].set_visible(False)

        if show:
            plt.show()
        else:
            return fig, ax1

    # 2D PDP
    else:
        # xv = values, independent variable
        # xs = scaled values
        # x = scaled values, ordered, then cast to original type
        if ind[0] == ind[1]:
            raise UnsupportedError("The two features must be different for 2D PDP.")
        xv = []
        for i in ind:
            xv.append(df_features.iloc[:, i])

        xmin = list(xmin) if isinstance(xmin, tuple) else list([xmin, xmin])
        xmax = list(xmax) if isinstance(xmax, tuple) else list([xmax, xmax])
        xs = []
        x = []
        cat_list = []
        npoints = [20, 20] if npoints is None else [npoints, npoints]

        for i, xv_i in enumerate(xv):
            if not is_numeric_dtype(xv_i):
                cat_list.append(sorted(set(xv_i)))
                xmin[i] = 0
                xmax[i] = len(cat_list[i])
                xs.append(np.linspace(xmin[i], xmax[i]-1e-6, npoints[i]))
                x.append([cat_list[i][int(xi)] for xi in xs[i]])
                npoints[i] = len(xs[i])

            else:
                xmin[i], xmax[i] = compute_bounds(xmin[i], xmax[i], xv_i)
                xs.append(np.linspace(xmin[i], xmax[i], npoints[i]))
                x.append(np.linspace(xmin[i], xmax[i], npoints[i]).astype(xv_i.dtype))
                cat_list.append(None)

        df_test = df_features.copy()
        vals = []
        for i in range(npoints[0]):
            for j in range(npoints[1]):
                df_test.iloc[:, ind[0]] = x[0][i]
                df_test.iloc[:, ind[1]] = x[1][j]
                vals.append(model(df_test).mean())
        vals = np.array(vals).reshape(npoints[0], npoints[1])

        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')

        x0, x1 = np.meshgrid(xs[0], xs[1], indexing='ij')
        ax.plot_surface(x0, x1, vals, cmap=obsidian_cm.obsidian_viridis)

        ax.set_xlabel(feature_names[ind[0]], fontsize=13)
        if not is_numeric_dtype(xv[0]):
            ax.set_xticks(np.arange(len(cat_list[0])))
            ax.set_xticklabels(cat_list[0])
        
        ax.set_ylabel(feature_names[ind[1]], fontsize=13)
        if not is_numeric_dtype(xv[1]):
            ax.set_yticks(np.arange(len(cat_list[1])))
            ax.set_yticklabels(cat_list[1])
            
        ax.set_zlabel("E[f(x) | " + str(feature_names[ind[0]]) + ", " + str(feature_names[ind[1]]) + "]", fontsize=13)

        if show:
            plt.show()
        else:
            return fig, ax
