"""Plotly figure-generating functions"""

from obsidian.campaign import Campaign
from obsidian.optimizer import Optimizer
from obsidian.exceptions import UnfitError, UnsupportedError
from obsidian.parameters import Param_Continuous
from .branding import obsidian_colors
from obsidian.plotting.branding import obsidian_color_list as colors

import plotly.graph_objects as go
from plotly.graph_objects import Figure
from plotly.subplots import make_subplots
from sklearn.manifold import MDS

import pandas as pd
import numpy as np
import math


def visualize_inputs(campaign: Campaign) -> Figure:
    """
    Visualizes the input variables of a campaign.

    Args:
        campaign (Campaign): The campaign object containing the input data.

    Returns:
        Figure: The plotly Figure object containing the visualization.
    """
    n_dim = campaign.X_space.n_dim
    X = campaign.X
    
    # Enforce that there are 2 rows
    # Determine the number of columns based on the number of dimensions
    rows = 2
    cols = math.ceil(n_dim / rows)
    
    height = 200 * rows
    width = 300 * cols
    fontsize = 8

    color_list = colors * 10
    
    # Add an extra 2 cols for the correlation matrix
    fig = make_subplots(
        rows=rows, cols=cols + 2,
        vertical_spacing=0.2,
        horizontal_spacing=0.1,
        specs=[[{}]*cols + [{"colspan": 2, "rowspan": 2}, None],
               [{}]*cols + [None, None]],
        subplot_titles=[X.columns[i] for i in range(cols)]
        + ['Correlation Matrix']
        + [X.columns[i] for i in range(cols, n_dim)]
    )
    
    for i, param in enumerate(X.columns):
        row_i = i // cols + 1
        col_i = i % cols + 1
        fig.add_trace(go.Scatter(x=X.index, y=X[param],
                                 mode='markers', name=param,
                                 marker=dict(color=color_list[i]),
                                 showlegend=False),
                      row=row_i, col=col_i)
        fig.update_xaxes(tickvals=np.around(np.linspace(0, campaign.m_exp, 5)),
                         row=row_i, col=col_i)
    
    # Calculate the correlation matrix
    X_u = campaign.X_space.unit_map(X)
    corr_matrix = X_u.corr()
    fig.add_trace(go.Heatmap(z=corr_matrix.values,
                             x=corr_matrix.columns,
                             y=corr_matrix.columns,
                             colorscale=[[0, obsidian_colors.rich_blue],
                                         [0.5, obsidian_colors.teal],
                                         [1, obsidian_colors.lemon]],
                             name='Correlation'),
                  row=1, col=cols+1)
    
    fig.update_yaxes(showticklabels=False, row=1, col=cols+1)
    fig.update_xaxes(tickangle=-90, row=1, col=cols+1)
    
    fig.update_layout(width=width, height=height, template='ggplot2',
                      font_size=fontsize, title_text='Campaign Data Visualization')
    fig.update_annotations(font_size=fontsize)
    
    return fig


def MDS_plot(campaign: Campaign) -> Figure:
    """
    Creates a Multi-Dimensional Scaling (MDS) plot of the campaign data,
    colored by iteration.
    
    This plot is helpful to visualize the convergence of the optimizer on a 2D plane.
    
    Args:
        campaign (Campaign): The campaign object containing the data.
        
    Returns:
        fig (Figure): The MDS plot
    """
    mds = MDS(n_components=2)
    X_mds = mds.fit_transform(campaign.X_space.encode(campaign.X))

    iter_max = campaign.data['Iteration'].max()
    iter_vals = campaign.data['Iteration'].values
    
    if campaign.data['Iteration'].nunique() == 1:
        iter_vals = np.zeros_like(iter_vals)
        iter_max = 0
        cbar = None
    else:
        cbar = dict(title=dict(text='Iteration', font=dict(size=10)))
    
    fig = go.Figure()

    fig.add_trace(go.Scatter(x=X_mds[:, 0], y=X_mds[:, 1],
                             mode='markers',
                             name='',
                             marker={'color': iter_vals, 'size': 10,
                                     'cmax': iter_max, 'cmin': 0,
                                     'colorscale': [[0, obsidian_colors.rich_blue],
                                                    [0.5, obsidian_colors.teal],
                                                    [1, obsidian_colors.lemon]],
                                     'colorbar': cbar
                                     },
                             customdata=campaign.data[
                                 list(campaign.X_space.X_names) + ['Iteration']],
                             showlegend=False
                             ))

    template = ["<b>"+str(param.name)+"</b>: "+" %{customdata["+str(i)+"]"
                + (":.3G}"if isinstance(param, Param_Continuous) else "}") + "<br>"
                for i, param in enumerate(campaign.X_space)]
    
    fig.update_traces(hovertemplate=''.join(template)
                      + '<b>Iteration</b>'
                      + ": %{customdata["+str(len(campaign.X_space))+"]}<br>"
                      + '<b>MDS C1</b>' + ": %{x:.3G}<br>"
                      + '<b>MDS C2</b>' + ": %{y:.3G}<br>")
    
    fig.update_xaxes(title_text='Component 1')
    fig.update_yaxes(title_text='Component 2')
    fig.update_layout(template='ggplot2', title='Multi-Dimensional Scaling (MDS) Plot',
                      autosize=False, height=400, width=500)
    
    return fig


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
    NRMSE = RMSE/(y_true.max()-y_true.min())
    
    y_min = np.min([y_true.min(), y_pred.min()])
    y_max = np.max([y_true.max(), y_pred.max()])
    abs_margin = 0.1
    y_abs = [y_min/(1+abs_margin), y_max*(1+abs_margin)]
    
    error_y = y_ub - y_pred
    error_y_minus = y_pred - y_lb
    
    fig = go.Figure()

    fig.add_trace(go.Scatter(x=y_true, y=y_pred,
                             error_y={'array': [f'{y:.3G}' for y in error_y],
                                      'arrayminus': [f'{y:.3G}' for y in error_y_minus],
                                      'color': 'gray', 'thickness': 0.5},
                             mode='markers',
                             name='Observations',
                             marker={'color': NRMSE, 'size': 15,
                                     'cmax': 0.5, 'cmin': 0,
                                     'colorscale': [[0, obsidian_colors.rich_blue],
                                                    [0.5, obsidian_colors.teal],
                                                    [1, obsidian_colors.lemon]],
                                     'colorbar': dict(title=dict(text='NRMSE', font=dict(size=10)))
                                     },
                             showlegend=False
                             ))
    
    fig.update_traces(hovertemplate="(%{x:.3G}, %{y:.3G}) +%{error_y.array:.3G}/-%{error_y.arrayminus:.3G}")
    
    fig.add_trace(go.Scatter(x=y_abs, y=y_abs,
                             mode='lines',
                             name='Parity',
                             showlegend=False,
                             line={'color': 'black', 'dash': 'dot'}))

    fig.update_xaxes(title_text=f'Actual Response ({y_name})')
    fig.update_yaxes(title_text=f'Predicted Response ({y_name})')
    fig.update_layout(template='ggplot2', title='Parity Plot',
                      autosize=False, height=400, width=500)
    
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
        df_mean = optimizer.X_best_f
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
    df_mean = optimizer.X_best_f
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
        if len(optimizer.X_space) != 2:
            X_t_train_ex = optimizer.X_t_train.copy().drop(columns=[X0_name, X1_name]).values
            X_t_test_ex = optimizer.X_space.encode(X_test.copy().drop(columns=[X0_name, X1_name]).drop_duplicates()).values

            if X_t_train_ex.shape[0] > 1:
                X_t_dist = np.linalg.norm(X_t_train_ex-X_t_test_ex, ord=2, axis=1)
                X_t_dist_scaled = (X_t_dist - X_t_dist.min())/(X_t_dist.max()-X_t_dist.min())
                dist_scaled = 15-10*X_t_dist_scaled
            else:
                dist_scaled = 10
        else:
            dist_scaled = 10
        
        fig.add_trace(go.Scatter3d(x=optimizer.X_train[X0_name], y=optimizer.X_train[X1_name],
                                   z=Z_train[y_name],
                                   mode='markers',
                                   marker={'color': '#000000', 'size': dist_scaled},
                                   name='Observations'))

    return fig


def optim_progress(campaign: Campaign,
                   response_ids: int | tuple[int] | None = None,
                   color_feature_id: int | None | str = 'Iteration',
                   X_suggest: pd.DataFrame | None = None) -> Figure:
    """
    Generates a plotly figure to visualize optimization progress

    Args:
        campaign (Campaign): The campaign object containing the data.
        response_ids (list[int], optional): The indices of the responses to plot. Defaults to ``[0, 1]``.
        color_feature_id (int | None, optional): The index of the feature to use for coloring the markers.
            Defaults to ``None``, which will color by iteration.
        X_suggest (pd.DataFrame | None, optional): The suggested next experiments to evaluate.
            Defaults to ``None``.

    Returns:
        Figure: The plotly figure.

    """
    fig = go.Figure()

    if response_ids is None:
        if campaign._is_mo:
            response_ids = (0, 1)
        else:
            response_ids = (0)
    if isinstance(response_ids, int):
        response_ids = (response_ids,)

    # Extract input and output names
    out_names = []
    for id in response_ids:
        out_names.append(campaign.out.columns[id])
    X_names = list(campaign.X.columns)

    for id in response_ids:
        if id >= len(out_names):
            raise ValueError(f'Response ID {id} is out of range')
    if isinstance(color_feature_id, int):
        if color_feature_id >= len(campaign.X_space):
            raise ValueError(f'Color feature ID {color_feature_id} is out of range')
        x_color_name = X_names[color_feature_id]
    if isinstance(color_feature_id, str):
        if color_feature_id not in campaign.data.columns:
            raise ValueError(f'Color feature {color_feature_id} is not in the data')
        x_color_name = color_feature_id

    # Unpack experimental data to plot progress
    out_exp = campaign.out[out_names]
    if not campaign._is_mo:
        # In this case, we only have 1 response to plot, so use the index on x-axis
        out_exp = out_exp.reset_index(drop=False).rename(columns={'index': 'Experiment'})
        out_names.insert(0, 'Experiment')

    if color_feature_id is not None:
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
        x=out_exp.iloc[:, 0], y=out_exp.iloc[:, 1],
        mode='markers',
        marker=marker_dict,
        customdata=campaign.data[X_names],
        name='Data'))

    template = ["<b>"+str(param.name)+"</b>: "+" %{customdata["+str(i)+"]"
                + (":.3G}"if isinstance(param, Param_Continuous) else "}") + "<br>"
                for i, param in enumerate(campaign.X_space)]
    fig.update_traces(hovertemplate=''.join(template)
                      + '<b>' + out_names[0] + '</b>' + ": %{x:.3G}<br>"
                      + '<b>' + out_names[1] + '</b>' + ": %{y:.3G}<br>")

    if X_suggest is not None:
        if not all(x in X_suggest.columns for x in campaign.X.columns):
            raise ValueError('Suggested data must contain all responses')
        
        eval_suggest = campaign.evaluate(X_suggest)
        
        if campaign.objective is None:
            y_mu = []
            lb = []
            ub = []
            for response in out_names:
                y_mu.append(eval_suggest[response + ' (pred)'])
                lb.append(eval_suggest[response + ' lb'])
                ub.append(eval_suggest[response + ' ub'])
            error_y_plus = ub[1] - y_mu[1]
            error_y_minus = y_mu[1] - lb[1]
            error_x_plus = ub[0] - y_mu[0]
            error_x_minus = y_mu[0] - lb[0]
            
            y_mu = pd.concat(y_mu, axis=1)
            
            hovertext = out_names[0] + ' :%{x:.3G} +%{error_x.array:.2G}/- \
                        %{error_x.arrayminus:.2G}<br>' + out_names[1] + ': %{y:.3G} \
                            +%{error_y.array:.2G}/-%{error_y.arrayminus:.2G}'
            
        else:
            if campaign._is_mo:
                y_mu = eval_suggest[out_names]
            else:
                y_mu = eval_suggest[out_names[-1]]
                m_data = len(out_exp)
                m_suggest = len(X_suggest)
                y_mu = pd.concat([pd.DataFrame(np.arange(m_data, m_data+m_suggest), columns=['Experiment']),
                                  y_mu], axis=1)
            error_y_plus = error_y_minus = error_x_minus = error_x_plus = None

            hovertext = out_names[0] + ' :%{x:.3G}' + '<br>' \
                + out_names[1] + ' :%{y:.3G}'

        fig.add_trace(go.Scatter(
            x=y_mu.iloc[:, 0],
            y=y_mu.iloc[:, 1],
            mode='markers',
            marker=dict(color=obsidian_colors.accent.pastel_blue,
                        symbol='diamond-open',
                        size=7,
                        line=dict(width=2)),
            name='Suggested',
            error_y={'array': error_y_plus,
                     'arrayminus': error_y_minus,
                     'color': 'gray', 'thickness': 1},
            error_x={'array': error_x_plus,
                     'arrayminus': error_x_minus,
                     'color': 'gray', 'thickness': 1},
            hovertemplate=hovertext))
        
    fig.update_layout(
        xaxis_title=out_names[0],
        yaxis_title=out_names[1],
        title='Optimization Results'
    )

    fig.update_layout(coloraxis_colorbar=dict(yanchor="top", y=1, x=0, ticks="outside"))

    fig.update_layout(legend=dict(
        yanchor="bottom",
        y=0.05,
        xanchor="right",
        x=0.95
    ))

    fig.update_layout(width=500, height=400, template='ggplot2')
    
    return fig
