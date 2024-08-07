from .utils import add_tab, make_input, make_dropdown, make_switch, make_slider, make_knob, make_table, make_collapse
import dash_bootstrap_components as dbc
from dash import dcc, html, Dash, dash_table, callback, Output, Input, State, ALL, MATCH

import pandas as pd

from obsidian.plotting.plotly import surface_plot, factor_plot
from .utils import load_optimizer, center


def setup_plots(app, app_tabs):
    
    surface_plot = dbc.Container(children=[
        html.H5('Surface Plot'),
        html.Br(),
        dbc.Row([
            dbc.Col(make_dropdown('X Axis', 'Select the first parameter for the surface plot',
                                  options=[], id='input-surfaceplot-x')),
            dbc.Col(make_dropdown('Y Axis', 'Select the second parameter for the surface plot',
                                  options=[], id='input-surfaceplot-y'))
            ]),
        dbc.Spinner(html.Div(id='graph-surface'), color='primary')
        ])
    factor_plot = dbc.Container(children=[
        html.H5('1D Factor Effects'),
        html.Br(),
        make_dropdown('Factor', 'Select the parameter for the factor effect plot',
                      options=[], id='input-factorplot-x',),
        dbc.Spinner(html.Div(id='graph-factor'), color='primary')
        ])
    
    # Add all of these elements to the app
    elements = [html.Br(), surface_plot, html.Hr(), factor_plot]
    add_tab(app_tabs, elements, 'tab-plots', 'Explore')
    setup_plots_callbacks(app)
    
    return


def setup_plots_callbacks(app):
    
    @app.callback(
        Output('input-surfaceplot-x', 'options'),
        Output('input-surfaceplot-x', 'value'),
        Input('store-fit', 'data'),
        State('store-X0', 'data'),
        State('input-response_name', 'value')
    )
    def select_x_surface_plot(opt_save, X0, response):
        df_X0 = pd.DataFrame(X0)
        if opt_save is None:
            return [], None
        x_options = df_X0.columns.to_list()
        x_options.remove(response)
        return x_options, x_options[0]
    
    @app.callback(
        Output('input-surfaceplot-y', 'options'),
        Output('input-surfaceplot-y', 'value'),
        Input('store-fit', 'data'),
        Input('input-surfaceplot-x', 'value'),
        State('input-surfaceplot-x', 'options')
    )
    def select_y_surface_plot(opt_save, xval, x_options):
        if opt_save is None:
            return [], None
        y_options = x_options
        y_options.remove(xval)
        return y_options, y_options[0]
    
    @app.callback(
        Output('input-factorplot-x', 'options'),
        Output('input-factorplot-x', 'value'),
        Input('store-fit', 'data'),
        State('store-X0', 'data'),
        State('input-response_name', 'value')
    )
    def select_x_factor_plot(opt_save, X0, response):
        df_X0 = pd.DataFrame(X0)
        if opt_save is None:
            return [], None
        x_options = df_X0.columns.to_list()
        x_options.remove(response)
        return x_options, x_options[0]
    
    @app.callback(
        Output('graph-surface', 'children'),
        Input('store-fit', 'data'),
        Input('input-surfaceplot-x', 'value'),
        Input('input-surfaceplot-y', 'value'),
        State('store-config', 'data')
    )
    def graph_surface_plot(opt_save, p_x, p_y, config):
        if opt_save is None:
            return dbc.Alert('Model must be fit first', color='info')
        
        optimizer = load_optimizer(config, opt_save)
        i_x = list(optimizer.X_space.X_names).index(p_x)
        i_y = list(optimizer.X_space.X_names).index(p_y)
        
        splot = surface_plot(optimizer, [i_x, i_y], plot_data=True)
        splot.update_layout(height=600)
        
        graph = dcc.Graph(figure=splot)
        
        return center(graph)
 
    @app.callback(
        Output('graph-factor', 'children'),
        Input('store-fit', 'data'),
        Input('input-factorplot-x', 'value'),
        State('store-config', 'data')
    )
    def graph_factor_plot(opt_save, p_x, config):
        if opt_save is None:
            return dbc.Alert('Model must be fit first', color='info')
        
        optimizer = load_optimizer(config, opt_save)
        i_x = list(optimizer.X_space.X_names).index(p_x)
        
        fplot = factor_plot(optimizer, feature_id=i_x)
        fplot.update_layout(height=600, width=800)
        
        graph = dcc.Graph(figure=fplot)
        
        return center(graph)
 
    return
