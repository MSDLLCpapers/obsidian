"Analysis utility functions for examining metrics over the context of an optimization campaign"

import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
from obsidian.parameters import Param_Continuous


def plot_ofat_ranges(optimizer, ofat_ranges):
    """
    Plots each parameter's 1D OFAT acceptable range

    Args:
        optimizer (BayesianOptimizer): The optimizer object which contains a surrogate that has been fit to data
            and can be used to make predictions.
        ofat_ranges (pd.DataFrame): A DataFrame containing the acceptable range values for each parameter.

    Returns:
        fig (matplotlib.figure.Figure): The parameter OFAT acceptable-range plot
    """

    fig = plt.figure(figsize=(8, 4))
    colors = plt.rcParams['axes.prop_cycle'].by_key()['color']
    for i, (index, row) in enumerate(ofat_ranges.iloc[0:10, :].iterrows()):
        color = colors[i]

        plt.plot([index, index], [row['Min_LB'], row['Max_LB']],
                 linewidth=6, linestyle='solid', color=color, label='High Confidence' if i == 0 else None)
        if row['Min_LB'] > row['Min_Mu']:
            plt.annotate(
                f'{(row["Min_LB"]*optimizer.X_space.X_range[index].iloc[0]+optimizer.X_space.X_min[index].iloc[0]):.2f}',
                xy=(i, row['Min_LB']), xytext=(i + 0.25, row['Min_LB']),
                fontsize=8, ha='left', va='center', rotation=0, arrowprops=dict(arrowstyle='-', color=color, lw=1))
        if row['Max_LB'] < row['Max_Mu']:
            plt.annotate(
                f'{(row["Max_LB"]*optimizer.X_space.X_range[index].iloc[0]+optimizer.X_space.X_min[index].iloc[0]):.2f}',
                xy=(i, row['Max_LB']), xytext=(i + 0.25, row['Max_LB']),
                fontsize=8, ha='left', va='center', rotation=0, arrowprops=dict(arrowstyle='-', color=color, lw=1))

        plt.plot([index, index], [row['Min_Mu'], row['Max_Mu']], linewidth=3,
                 linestyle='solid', color=color, label='Average' if i == 0 else None)
        plt.annotate(
            f'{(row["Min_Mu"]*optimizer.X_space.X_range[index].iloc[0]+optimizer.X_space.X_min[index].iloc[0]):.2f}',
            xy=(i, row['Min_Mu']), xytext=(i + 0.25, row['Min_Mu']),
            fontsize=8, ha='left', va='center', rotation=0, arrowprops=dict(arrowstyle='-', color=color, lw=1))
        plt.annotate(
            f'{(row["Max_Mu"]*optimizer.X_space.X_range[index].iloc[0]+optimizer.X_space.X_min[index].iloc[0]):.2f}',
            xy=(i, row['Max_Mu']), xytext=(i + 0.25, row['Max_Mu']),
            fontsize=8, ha='left', va='center', rotation=0, arrowprops=dict(arrowstyle='-', color=color, lw=1))

        if row['Min_UB'] < row['Min_Mu']:
            plt.plot([index, index], [row['Min_UB'], row['Min_Mu']],  linewidth=1, linestyle=':', color=color)
        if row['Max_UB'] > row['Max_Mu']:
            plt.plot([index, index], [row['Max_UB'], row['Max_Mu']],  linewidth=1, linestyle=':', color=color)
        plt.plot([0], [0], linewidth=1, linestyle=':', color=color, label='Low Confidence' if i == 0 else None)

    plt.xticks(rotation=90)
    plt.ylabel('Parameter Value (Scaled)')
    plt.ylim([-0.15, 1.15])
    plt.xlim([-1, i+1])
    plt.title(f'Univeriate Range (OFAT) Estimates from APO Model \n Ranges exceeding {row["Threshold"]} {row["Response"]}',
              fontsize=10)
    plt.legend(bbox_to_anchor=(1.1, 1.05))

    return fig


def plot_interactions(optimizer, cor, clamp=False):
    """
    Plots the parameter interaction matrix

    Args:
        optimizer (BayesianOptimizer): The optimizer object which contains a surrogate that has been fit to data
            and can be used to make predictions.
        cor (np.ndarray): The correlation matrix representing the parameter interactions.
        clamp (bool, optional): Whether to clamp the colorbar range to (0, 1). Defaults to False.

    Returns:
        matplotlib.figure.Figure: The parameter interaction plot
    """
    fig = plt.figure(figsize=(5, 5))
    ax = fig.gca()
    cax = ax.matshow(cor)
    if clamp:
        cax.set_clim(0, 1)
    axis = np.arange(len(optimizer.X_space.X_names))
    names = optimizer.X_space.X_names
    ax.set_xticks(axis)
    ax.set_xticklabels(names, rotation=90)
    ax.set_yticks(axis)
    ax.set_yticklabels(names, rotation=0)
    cbar = fig.colorbar(cax)
    ax.set_title('Parameter Interactions')
    cbar.ax.set_ylabel('Range Shrinkage')
    for (i, j), z in np.ndenumerate(cor):
        if z > 0.05:
            ax.text(j, i, '{:0.2f}'.format(z), ha='center', va='center', fontsize=8)
    return fig


def calc_ofat_ranges(optimizer, threshold, X_ref, PI_range=0.7,
                     steps=100, response_id=0, calc_interacts=True):
    """
    Calculates an OFAT design space using confidence bounds around the optimizer prediction. Also
    includes a matrix of interaction scores.

    Args:
        optimizer (BayesianOptimizer): The optimizer object which contains a surrogate that has been fit to data
            and can be used to make predictions.
        X_ref (pd.DataFrame): The reference data point from which the OFAT variations are calculated.
        threshold (float): The response value threshold (minimum value) which would be considered passing for OFAT variations.
        PI_range (float, optional): The prediction interval coverage (fraction of density)
        steps (int, optional): The number of steps to use in the search for the OFAT boundaries.
            The default value is 100.
        response_id (int, optional): The index of the relevant response within the fitted optimizer object.
            The default value is 0.
        calc_interacts (bool, optional): Whether or not to return the interaction matrix; default is True.

    Returns:
        ofat_ranges (pd.DataFrame): A dataframe describing the min/max OFAT values using each LB, UB, and average prediction.
            Values are scaled in the (0,1) space based on optimizer.X_space.
        cor (np.array): A matrix of interaction values between every combination of two parameters.
            Each value is the fractional reduction in size for the acceptable range envelope created by a 2-factor variation,
            in comparison to the corresponding two independent 1-factor variations. As such, diagonal elements are 0.
    """

    threshold = 0.4
    ofat_ranges = []
    response_name = optimizer.target[response_id].name

    for p in optimizer.X_space:
        if isinstance(p, Param_Continuous):
            X_min = p.min
            X_max = p.max
            X_range = p.range
            X_span = np.linspace(X_min, X_max, steps)

            X_sim = pd.DataFrame(np.repeat(X_ref.values, repeats=steps, axis=0), columns=X_ref.columns)
            X_sim[p.name] = X_span
            df_pred = optimizer.predict(X_sim, PI_range=PI_range)
            lb = df_pred[response_name + ' lb']
            ub = df_pred[response_name + ' ub']
            pred_mu = df_pred[response_name + ' (pred)']
        
            row = {'Name': p.name, 'PI Range': PI_range, 'Threshold': threshold, 'Response': response_name}
            labels = ['Mu', 'LB', 'UB']

            for label, y in zip(labels, [pred_mu, lb, ub]):
                pass_ids = np.where(pred_mu > threshold)
                pass_vals = X_sim[p.name].iloc[pass_ids]

                row['Min_'+label] = (pass_vals.min()-X_min)/X_range
                row['Max_'+label] = (pass_vals.max()-X_min)/X_range
            ofat_ranges.append(row)

    ofat_ranges = pd.DataFrame(ofat_ranges).set_index('Name')

    if calc_interacts:
        cor = []

        for i, pi in enumerate(optimizer.X_space.X_names):
            cor_j = []

            Xi_pass_min = optimizer.X_space.X_min[pi] + optimizer.X_space.X_range[pi]*ofat_ranges['Min_Mu'][pi]
            Xi_pass_max = optimizer.X_space.X_min[pi] + optimizer.X_space.X_range[pi]*ofat_ranges['Max_Mu'][pi]
            Xi_pass_span = np.linspace(Xi_pass_min, Xi_pass_max, steps)

            for j, pj in enumerate(optimizer.X_space.X_names):
                Xj_pass_min = optimizer.X_space.X_min[pj] + optimizer.X_space.X_range[pj]*ofat_ranges['Min_Mu'][pj]
                Xj_pass_max = optimizer.X_space.X_min[pj] + optimizer.X_space.X_range[pj]*ofat_ranges['Max_Mu'][pj]
                Xj_pass_span = np.linspace(Xj_pass_min, Xj_pass_max, steps)

                X_sim_cor = pd.DataFrame(np.repeat(X_ref.values, repeats=steps, axis=0), columns=X_ref.columns)
                
                X_sim_cor[pj] = Xj_pass_span
                if not pi == pj:
                    X_sim_cor[pi] = Xi_pass_span

                pred_mu_cor_all, _ = optimizer.predict(X_sim_cor)
                pred_mu_cor = pred_mu_cor_all.iloc[:, response_id]
                cor_passing = np.where(pred_mu_cor > threshold)[0]

                if len(cor_passing) > 0:
                    start = cor_passing[0]
                    stop = cor_passing[-1]
                    cor_ij = 1-(stop-start)/(steps-1)
                    cor_j.append(cor_ij)
                else:
                    cor_j.append(0)

            cor.append(cor_j)
        cor = np.array(cor)
    else:
        cor = None

    return ofat_ranges, cor


def sensitivity(optimizer,
                dx: float = 1e-6,
                X_ref: pd.DataFrame | None = None) -> pd.DataFrame:
    """
    Calculates the sensitivity of the surrogate model predictions with respect to each parameter in the X_space.

    Args:
        optimizer (BayesianOptimizer): The optimizer object which contains a surrogate that has been fit to data
        and can be used to make predictions.
        dx (float, optional): The perturbation size for calculating the sensitivity. Defaults to 1e-6.
        X_ref (pd.DataFrame | None, optional): The reference input values for calculating the sensitivity.
            If None, the mean of X_space will be used as the reference. Defaults to None.

    Returns:
        pd.DataFrame: A DataFrame containing the sensitivity values for each parameter in X_space.

    Raises:
        ValueError: If X_ref does not contain all parameters in optimizer.X_space or if X_ref is not a single row DataFrame.
    """
    if X_ref is None:
        X_ref = optimizer.X_space.mean()
    else:
        if not all(x in X_ref.columns for x in optimizer.X_space.X_names):
            raise ValueError('X_ref must contain all parameters in X_space')
        if X_ref.shape[0] != 1:
            raise ValueError('X_ref must be a single row DataFrame')
    
    y_ref = optimizer.predict(X_ref)
    
    sens = {}
    
    # Only do positive perturbation, for simplicity
    for param in optimizer.X_space:
        base = param.unit_map(X_ref[param.name].values)[0]
        # Space already mapped to (0,1), use absolute perturbation
        dx_pos = np.array(base+dx).reshape(-1, 1)
        X_sim = X_ref.copy()
        X_sim[param.name] = param.unit_demap(dx_pos)[0]
        y_sim = optimizer.predict(X_sim)
        dydx = (y_sim - y_ref)/dx
        sens[param.name] = dydx.to_dict('records')[0]
    
    df_sens = pd.DataFrame(sens).T[[y+' (pred)' for y in optimizer.y_names]]
    
    return df_sens
