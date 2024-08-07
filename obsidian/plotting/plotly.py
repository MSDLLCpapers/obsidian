"""Plotly figure-generating functions"""

from obsidian.campaign import Campaign
from obsidian.optimizer import Optimizer
from obsidian.exceptions import UnfitError, UnsupportedError
from .branding import obsidian_colors

import plotly.graph_objects as go
from plotly.graph_objects import Figure

import pandas as pd
import numpy as np


def parity_plot(optimizer: Optimizer,
                f_transform: bool = False,
                response_id: int = 0) -> Figure:
    """
    Produces a plot of surrogate model predictions at an OFAT range in one variable.

    Args:
        optimizer (Optimizer): The optimizer object which contains a surrogate that has been fit to data
            and can be used to make predictions.
        f_transform (bool, optional): An indicator for whether or not to plot the response value in the "objective
            function" form which is directly used by the optimizer, else using the "measured response" form which the
            optimizer preproceses. Default value is ``False`` which plots the raw measured response scale.
        response_id (int, optional): Index of the response for potential multi-response models.
            Default value is ``0`` (single-response).

    Returns:
        fig (Figure): The optimizer fit parity plot

    Raises:
        TypeError: If the optimizer is not an instance of obsidian Optimizer
        UnfitError: If the optimizer is not fit
        ValueError: If the response_id is not a valid response index
    """

    if not isinstance(optimizer, Optimizer):
        raise TypeError('Optimizer must be an instance of obsidian Optimizer')
    if not optimizer.is_fit:
        raise UnfitError('Optimizer must be fit before plotting predictions')
    if response_id < 0 or response_id >= len(optimizer.target):
        raise ValueError('response_id must be a valid response index')

    X = optimizer.X_train
    y_pred = optimizer.predict(X, return_f_inv=not f_transform)
    y_true = optimizer.f_train if f_transform else optimizer.y_train

    y_name = optimizer.y_names[response_id]

    y_lb = y_pred[y_name+' lb'].values
    y_ub = y_pred[y_name+' ub'].values
    y_pred = y_pred[y_name+' (pred)'].values
    y_true = y_true[y_name].values
    
    RMSE = ((y_true-y_pred)/y_true)**2
    
    y_min = np.min([y_true.min(), y_pred.min()])
    y_max = np.max([y_true.max(), y_pred.max()])
    abs_margin = 0.1
    y_abs = [y_min/(1+abs_margin), y_max*(1+abs_margin)]
    
    fig = go.Figure()

    fig.add_trace(go.Scatter(x=y_true, y=y_pred,
                             error_y={'array': y_ub - y_pred,
                                      'arrayminus': y_pred - y_lb,
                                      'color': 'gray', 'thickness': 0.5},
                             mode='markers',
                             name='Observations',
                             marker={'color': RMSE, 'colorscale': 'Viridis', 'size': 15},
                             ))
    
    fig.add_trace(go.Scatter(x=y_abs, y=y_abs,
                             mode='lines',
                             name='Parity',
                             showlegend=False,
                             line={'color': 'black', 'dash': 'dot'}))

    fig.update_xaxes(title_text=f'Actual Response ({y_name})')
    fig.update_yaxes(title_text=f'Predicted Response ({y_name})')
    fig.update_layout(template='ggplot2', title='Parity Plot',
                      autosize=False, height=400, width=600)
    
    return fig


def factor_plot(optimizer: Optimizer,
                feature_id: int = 0,
                response_id: int = 0,
                f_transform: bool = False,
                X_ref: pd.DataFrame | None = None, plotRef: bool = True,
                ylim: tuple[float, float] | None = None) -> Figure:
    """
    Produces a plot of surrogate model predictions at an OFAT range in one variable.

    Args:
        optimizer (Optimizer): The optimizer object which contains a surrogate that has been fit to data
            and can be used to make predictions.
        feature_id (int, optional): The index of the desired variable to plot from the last data used
            to fit the surrogate model. The default value is ``0``.
        response_id (int, optional): Index of the response for potential multi-response models.
            Default value is ``0`` (single-response).
        f_transform (bool, optional): An indicator for whether or not to plot the response value in the "objective
            function" form which is directly used by the optimizer, else using the
            "measured response" form which the optimizer preproceses. Default value is
            ``False`` which plots the raw measured response scale.
        plotRef (bool, optional): An indicator for whether or not to plot the reference data points.
            Default value is ``True``.
        ylim (tuple, optional): The y-axis limits for the plot. Default value is ``None``.

    Returns:
        fig (Figure): The matplotlib plot of response value versus 1 predictor variable.

    Raises:
        TypeError: If the optimizer is not an instance of obsidian Optimizer
        UnfitError: If the optimizer is not fit
        ValueError: If the feature_id is not a valid feature index
        ValueError: If the response_id is not a valid response index
        ValueError: If X_ref is provided and not a pd.Series
    """
    if not isinstance(optimizer, Optimizer):
        raise TypeError('Optimizer must be an instance of obsidian Optimizer')
    if not optimizer.is_fit:
        raise UnfitError('Optimizer must be fit before plotting predictions')
    if feature_id >= len(optimizer.X_space):
        raise ValueError('feature_id must be a valid feature index')
    if response_id < 0 or response_id >= len(optimizer.target):
        raise ValueError('response_id must be a valid response index')

    # Create a dataframe of test samples for plotting
    n_samples = 100
    if X_ref is None:
        df_mean = optimizer.X_space.mean()
        X_test = pd.concat([df_mean]*n_samples, axis=0).reset_index(drop=True)
    else:
        if not isinstance(X_ref, pd.DataFrame):
            raise TypeError('X_ref must be a DataFrame')
        X_test = pd.concat([X_ref]*n_samples, axis=0).reset_index(drop=True)

    # Vary the indicated column
    X_name = X_test.columns[feature_id]
    param_i = optimizer.X_space.params[feature_id]
    unit_span = np.linspace(0, 1-1e-15, n_samples)  # Prevent issues with cat[1.0] = undefined
    X_test[X_name] = param_i.unit_demap(unit_span)
    X = X_test[X_name].values
    
    Y_pred = optimizer.predict(X_test, return_f_inv=not f_transform, PI_range=0.95)
    y_name = optimizer.y_names[response_id]

    Y_mu = Y_pred[y_name+('_t (pred)' if f_transform else ' (pred)')].values
    LCB = Y_pred[y_name+('_t lb' if f_transform else ' lb')].values
    UCB = Y_pred[y_name+('_t ub' if f_transform else ' ub')].values
        
    fig = go.Figure()

    fig.add_trace(go.Scatter(x=np.append(X, X[::-1]), y=np.append(UCB, LCB[::-1]),
                             fill='toself',
                             opacity=0.3,
                             line={'color': obsidian_colors.teal},
                             showlegend=True,
                             name='95% Pred Band'),
                  )
    
    fig.add_trace(go.Scatter(x=X, y=Y_mu,
                             mode='lines',
                             line={'color': obsidian_colors.teal},
                             name='Mean'),
                  )
    if (X_ref is not None) and plotRef:
        Y_pred_ref = optimizer.predict(X_ref, return_f_inv=not f_transform)
        Y_mu_ref = Y_pred_ref[y_name+('_t (pred)' if f_transform else ' (pred)')].values
        fig.add_trace(go.Scatter(x=X_ref.iloc[:, feature_id].values, y=Y_mu_ref,
                                 mode='markers',
                                 line={'color': obsidian_colors.teal},
                                 name='Ref'),
                      )
    fig.update_xaxes(title_text=X_name)
    fig.update_yaxes(title_text=y_name)
    fig.update_layout(template='ggplot2', title=f'Factor Effect Plot for {X_name}')
    fig.update_layout(autosize=False, width=600, height=400)
    if ylim is not None:
        fig.update_layout(yaxis_range=ylim)
    
    return fig


def surface_plot(optimizer: Optimizer,
                 feature_ids: list[int, int] = [0, 1],
                 response_id: int = 0,
                 f_transform: bool = False,
                 plot_bands: bool = True,
                 plot_data: bool = False) -> Figure:
    """
    Produces a surface plot of surrogate model predictions over a 2-parameter grid range.

    Args:
        optimizer (Optimizer): The optimizer object which contains a surrogate that has been fit to data
            and can be used to make predictions.
        feature_ids (list, optional): A list of integers containing the indices of the desired variables
            to plot from the last data used to fit the surrogate model. Default value is ``[0,1]``.
        f_transform (bool, optional): An indicator for whether or not to plot the response value in the "objective
            function" form which is directly used by the optimizer, else using the
            "measured response" form which the optimizer preprocesses. Default value is
            ``False`` which plots the raw measured response scale.
        plot_bands (bool, optional): An indicator for whether or not to plot the confidence bands as a wire
            frame around the surface plot. Default is ``True``.
        plot_data (bool, optional): An indicator for whether or not to plot the raw data locations.
            Default is ``False``, as the data z-height can be misleading for >2D data on a 3D plot.
        response_id (int, optional): Index of the response for potential multi-response models.
            Default value is ``0`` (single-response).

    Returns:
        fig (Figure): The matplotlib plot of surfaces over a 2-parameter grid.

    Raises:
        TypeError: If the optimizer is not an instance of obsidian Optimizer
        UnfitError: If the optimizer is not fit
        ValueError: If the feature_ids are not valid feature indices
        ValueError: If the response_id is not a valid response index
    """
    if not isinstance(optimizer, Optimizer):
        raise TypeError('Optimizer must be an instance of obsidian Optimizer')
    for feature_id in feature_ids:
        if feature_id >= len(optimizer.X_space):
            raise ValueError('feature_id must be a valid feature index')
    if not optimizer.is_fit:
        raise UnfitError('Optimizer must be fit before plotting predictions')
    if response_id < 0 or response_id >= len(optimizer.target):
        raise ValueError('response_id must be a valid response index')
    if plot_data and f_transform:
        raise UnsupportedError('Plotting data is not supported for transformed responses')

    # Create a dataframe of test samples for plotting
    n_grid = 100
    df_mean = optimizer.X_space.mean()
    X_test = pd.concat([df_mean]*(n_grid**2), axis=0).reset_index(drop=True)
    
    # Create a mesh grid which is necessary for the 3D plot
    X0_name = X_test.columns[feature_ids[0]]
    X1_name = X_test.columns[feature_ids[1]]
    
    # Vary the indicated column
    x_axes = []
    unit_span = np.linspace(0, 1-1e-15, n_grid)

    params = [optimizer.X_space.params[i] for i in feature_ids]
    for param_i in params:
        x_axes.append(np.array(param_i.unit_demap(unit_span)))

    x0_ax, x1_ax = x_axes
    X0, X1 = np.meshgrid(x0_ax, x1_ax)
    X_test[X0_name] = X0.flatten().reshape(-1, 1)
    X_test[X1_name] = X1.flatten().reshape(-1, 1)

    Y_pred = optimizer.predict(X_test, return_f_inv=not f_transform, PI_range=0.95)
    y_name = optimizer.y_names[response_id]

    Y_mu = Y_pred[y_name+('_t (pred)' if f_transform else ' (pred)')].values
    LCB = Y_pred[y_name+('_t lb' if f_transform else ' lb')].values
    UCB = Y_pred[y_name+('_t ub' if f_transform else ' ub')].values
    
    Z_mu = Y_mu.reshape(n_grid, n_grid)
    Z_LCB = LCB.reshape(n_grid, n_grid)
    Z_UCB = UCB.reshape(n_grid, n_grid)
    
    fig = go.Figure()
    fig.add_trace(go.Surface(z=Z_mu, x=x0_ax, y=x1_ax,
                             name='Mean',
                             opacity=0.85))
    
    # Plot the confidence bands only if desired
    if plot_bands:
        for i, (x0, x1, ucb, lcb) in enumerate(zip(X0, X1, Z_UCB, Z_LCB)):
            fig.add_trace(go.Scatter3d(z=ucb, x=x0, y=x1,
                                       mode='lines',
                                       line={'color': '#b0b0b0', 'width': 2},
                                       opacity=0.5,
                                       name='UCB',
                                       legendgroup='UCB',
                                       showlegend=True if i == 0 else False),
                          )
            fig.add_trace(go.Scatter3d(z=lcb, x=x0, y=x1,
                                       mode='lines',
                                       line={'color': '#b0b0b0', 'width': 2},
                                       opacity=0.5,
                                       name='LCB',
                                       legendgroup='LCB',
                                       showlegend=True if i == 0 else False),
                          )
            
    fig.update_layout(legend_orientation='h', template='ggplot2', title=f'Surface Plot for {X0_name} vs {X1_name}',
                      scene=dict(xaxis_title=X0_name, yaxis_title=X1_name, zaxis_title=y_name))
    
    Z_train = optimizer.f_train if f_transform else optimizer.y_train
    
    # Plot the locations of raw data only if desired
    if plot_data:
        # Use a marker size which is larger if closer to the "other" values of the surface
        # Smaller markers mean there are other variables pulling it away from the surface plot
        X_t_train_ex = optimizer.X_t_train.copy().drop(columns=[X0_name, X1_name]).values
        X_t_test_ex = optimizer.X_space.encode(X_test.copy().drop(columns=[X0_name, X1_name]).drop_duplicates()).values

        if X_t_train_ex.shape[0] > 1:
            X_t_dist = np.linalg.norm(X_t_train_ex-X_t_test_ex, ord=2, axis=1)
            X_t_dist_scaled = (X_t_dist - X_t_dist.min())/(X_t_dist.max()-X_t_dist.min())
            dist_scaled = 15-10*X_t_dist_scaled
        else:
            dist_scaled = 10
        
        fig.add_trace(go.Scatter3d(x=optimizer.X_train[X0_name], y=optimizer.X_train[X1_name],
                                   z=Z_train[y_name],
                                   mode='markers',
                                   marker={'color': '#000000', 'size': dist_scaled},
                                   name='Observations'))

    return fig


def MOO_results(campaign: Campaign,
                response_ids: list[int] = [0, 1],
                color_feature_id: int | None = None,
                y_suggest: pd.DataFrame | None = None) -> Figure:
    """
    Generates a plotly figure to visualize multi-objective optimization (MOO) results.

    Args:
        campaign (Campaign): The campaign object containing the data.
        response_ids (list[int], optional): The indices of the responses to plot. Defaults to ``[0, 1]``.
        color_feature_id (int | None, optional): The index of the feature to use for coloring the markers.
            Defaults to ``None``.
        y_suggest (pd.DataFrame | None, optional): The suggested data for the responses.
            Defaults to ``None``.

    Returns:
        Figure: The plotly figure.

    Raises:
        ValueError: If the campaign has less than two responses.
        ValueError: If the response ID is out of range.
        ValueError: If the color feature ID is out of range.
        ValueError: If the suggested data does not contain all responses.
    """
    fig = go.Figure()

    if not campaign._is_moo:
        raise ValueError('Campaign must have at least two responses for MOO results')
    for id in response_ids:
        if id >= campaign.n_response:
            raise ValueError(f'Response ID {id} is out of range')
    if color_feature_id is not None:
        if color_feature_id >= len(campaign.X_space):
            raise ValueError(f'Color feature ID {color_feature_id} is out of range')

    response_0 = campaign.y_names[response_ids[0]]
    response_1 = campaign.y_names[response_ids[1]]
    X_names = list(campaign.X_space.X_names)

    yexp_0 = campaign.data[response_0]
    yexp_1 = campaign.data[response_1]

    if color_feature_id is not None:
        x_color_name = campaign.X_space.X_names[color_feature_id]
        x_color = campaign.data[x_color_name]
        marker_dict = dict(color=x_color,
                           colorscale=[[0, obsidian_colors.rich_blue],
                                       [0.5, obsidian_colors.teal],
                                       [1, obsidian_colors.lemon]],
                           showscale=True,
                           colorbar=dict(title=x_color_name))

    else:
        x_color = None
        marker_dict = dict(color=obsidian_colors.primary.teal)

    fig.add_trace(go.Scatter(
        x=yexp_0, y=yexp_1,
        mode='markers',
        marker=marker_dict,
        customdata=campaign.data[X_names],
        name='Data'))

    template = ["<b>"+str(name)+": "+" %{customdata["+str(i)+"]:.3G}</b><br>"
                for i, name in enumerate(X_names)]
    fig.update_traces(hovertemplate=''.join(template))

    if y_suggest is not None:
        if not all(y+' (pred)' in y_suggest.columns for y in campaign.y_names):
            raise ValueError('Suggested data must contain all responses')
        
        y_0 = y_suggest[response_0 + ' (pred)']
        y_1 = y_suggest[response_1 + ' (pred)']
        lb_0 = y_suggest[response_0 + ' lb']
        ub_0 = y_suggest[response_0 + ' ub']
        lb_1 = y_suggest[response_1 + ' lb']
        ub_1 = y_suggest[response_1 + ' ub']

        fig.add_trace(go.Scatter(
            x=y_0,
            y=y_1,
            mode='markers',
            marker=dict(color=obsidian_colors.accent.pastel_blue,
                        symbol='diamond-open',
                        size=7,
                        line=dict(width=2)),
            name='Suggested',
            error_y={'array': ub_1 - y_1,
                     'arrayminus': y_1 - lb_1,
                     'color': 'gray', 'thickness': 1},
            error_x={'array': ub_0 - y_0,
                     'arrayminus': y_0 - lb_0,
                     'color': 'gray', 'thickness': 1},
            hovertemplate=response_0 + ' :%{x:.3G} +%{error_x.array:.2G}/- \
                        %{error_x.arrayminus:.2G}<br>' + response_1 + ': %{y:.3G} \
                            +%{error_y.array:.2G}/-%{error_y.arrayminus:.2G}'))
        
    fig.update_layout(
        xaxis_title=response_0,
        yaxis_title=response_1,
        title='Optimization Results'
    )

    fig.update_layout(coloraxis_colorbar=dict(yanchor="top", y=1, x=0, ticks="outside"))

    fig.update_layout(legend=dict(
        yanchor="bottom",
        y=0.05,
        xanchor="right",
        x=0.95
    ))

    fig.update_layout(coloraxis_colorbar_title_text='your title')
    fig.update_layout(width=500, height=400, template='ggplot2')
    return fig
