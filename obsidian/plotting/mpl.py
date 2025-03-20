"""Matplotlib figure-generating functions"""

from obsidian.optimizer import Optimizer

import matplotlib.pyplot as plt
from matplotlib.pyplot import Figure

import numpy as np
import pandas as pd


def plot_ofat_ranges(optimizer: Optimizer,
                     ofat_ranges: pd.DataFrame) -> Figure:
    """
    Plots each parameter's 1D OFAT acceptable range

    Args:
        optimizer (Optimizer): The optimizer object which contains a surrogate
            that has been fit to data and can be used to make predictions.
        ofat_ranges (pd.DataFrame): A DataFrame containing the acceptable range
            values for each parameter, at the low bound, average, and high bound.

    Returns:
        Figure: The parameter OFAT acceptable-range plot
    """

    fig = plt.figure(figsize=(2*len(ofat_ranges), 4))
    colors = plt.rcParams['axes.prop_cycle'].by_key()['color']
    
    # Iterate over the parameteres
    for i, (p_name, row) in enumerate(ofat_ranges.iterrows()):
        color = colors[i]

        # Plot as a bar chart; x-axis is the parameter name, y-axis is the scaled value
        plt.plot([p_name, p_name], [row['Min_LB'], row['Max_LB']],
                 linewidth=6, linestyle='solid', color=color, label='High Confidence' if i == 0 else None)
        
        # If the edges of LB are too close to mean, only annotate LB (higher conf)
        if row['Min_LB'] > row['Min_Mu']:
            plt.annotate(
                f'{(optimizer.X_space[i].unit_demap(row["Min_LB"])):.2f}',
                xy=(i, row['Min_LB']), xytext=(i + 0.25, row['Min_LB']),
                fontsize=8, ha='left', va='center', rotation=0, arrowprops=dict(arrowstyle='-', color=color, lw=1))
        if row['Max_LB'] < row['Max_Mu']:
            plt.annotate(
                f'{(optimizer.X_space[i].unit_demap(row["Max_LB"])):.2f}',
                xy=(i, row['Max_LB']), xytext=(i + 0.25, row['Max_LB']),
                fontsize=8, ha='left', va='center', rotation=0, arrowprops=dict(arrowstyle='-', color=color, lw=1))

        plt.plot([p_name, p_name], [row['Min_Mu'], row['Max_Mu']], linewidth=3,
                 linestyle='solid', color=color, label='Average' if i == 0 else None)
        
        # If the edges of the mean are too close to the UB, only annotate mean (higher conf)
        plt.annotate(
            f'{(optimizer.X_space[i].unit_demap(row["Min_Mu"])):.2f}',
            xy=(i, row['Min_Mu']), xytext=(i + 0.25, row['Min_Mu']),
            fontsize=8, ha='left', va='center', rotation=0, arrowprops=dict(arrowstyle='-', color=color, lw=1))
        plt.annotate(
            f'{(optimizer.X_space[i].unit_demap(row["Max_Mu"])):.2f}',
            xy=(i, row['Max_Mu']), xytext=(i + 0.25, row['Max_Mu']),
            fontsize=8, ha='left', va='center', rotation=0, arrowprops=dict(arrowstyle='-', color=color, lw=1))
        
        # Only plot UB if it isn't already encompassed by higher-confidence ranges
        if row['Min_UB'] < row['Min_Mu']:
            plt.plot([p_name, p_name], [row['Min_UB'], row['Min_Mu']],  linewidth=1, linestyle=':', color=color)
        if row['Max_UB'] > row['Max_Mu']:
            plt.plot([p_name, p_name], [row['Max_UB'], row['Max_Mu']],  linewidth=1, linestyle=':', color=color)
        plt.plot([0], [0], linewidth=1, linestyle=':', color=color, label='Low Confidence' if i == 0 else None)
        
        # Never annotate UB (low confidence)

    alpha = ofat_ranges['PI Range'].mode().iloc[0]
    LCL = (1 - alpha) / 2
    UCL = 1 - LCL
    comparator = ">" if row['Aim'] == 'max' else "<"
    
    plt.xticks(rotation=90)
    plt.ylabel('Parameter Value (Scaled)')
    plt.ylim([-0.15, 1.15])
    plt.xlim([-1, len(ofat_ranges)])
    plt.title('Univariate Range (OFAT) Estimates from APO Model \n'
              + f'Ranges Satisfying {row["Response"]} ' + comparator + f' {row["Threshold"]} \n'
              + f'Confidence Range: {LCL*100:.1f} - {UCL*100:.1f}%',
              fontsize=10)
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.close(fig)

    return fig


def plot_interactions(optimizer: Optimizer,
                      cor: np.ndarray,
                      clamp: bool = False):
    """
    Plots the parameter interaction matrix

    Args:
        optimizer (ptimizer): The optimizer object which contains a surrogate
            that has been fit to data and can be used to make predictions.
        cor (np.ndarray): The correlation matrix representing the parameter interactions.
        clamp (bool, optional): Whether to clamp the colorbar range to (0, 1).
            Defaults to ``False``.

    Returns:
        Figure: The parameter interaction plot
    """
    
    fig = plt.figure(figsize=(4, 4))
    ax = fig.gca()
    
    # Use matrix imshow to plot correlation matrix
    cax = ax.matshow(cor)
    if clamp:
        cax.set_clim(0, 1)
    
    # Set axis labels and ticks
    axis = np.arange(len(optimizer.X_space.X_names))
    names = optimizer.X_space.X_names
    ax.set_xticks(axis)
    ax.set_xticklabels(names, rotation=90)
    ax.set_yticks(axis)
    ax.set_yticklabels(names, rotation=0)
    cbar = fig.colorbar(cax)
    ax.set_title('Parameter Interactions')
    cbar.ax.set_ylabel('Range Shrinkage')
    
    # Add text annotations if correlation is greater than 0.05
    for (i, j), z in np.ndenumerate(cor):
        if z > 0.05:
            ax.text(j, i, '{:0.2f}'.format(z), ha='center', va='center', fontsize=8)
    plt.close(fig)
    
    return fig
