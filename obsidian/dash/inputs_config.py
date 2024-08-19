from .utils import add_tab, make_input, make_dropdown, make_switch, make_slider, make_knob, make_table, make_collapse
from .utils import center
import dash_bootstrap_components as dbc
from dash import dcc, html, Dash, dash_table, callback, Output, Input, State, ALL, MATCH
import pandas as pd
import base64
import io
from dash.exceptions import PreventUpdate


def make_acquisition(index, delete=True):
    
    aq_functions = [{'label': 'Expected Improvement', 'value': 'EI'},
                    {'label': 'Probability of Improvement', 'value': 'PI'},
                    {'label': 'Upper Confidence Band', 'value': 'UCB'},
                    {'label': 'Space Filling', 'value': 'SF'}]
    columns = [dbc.Col(make_dropdown('', 'Select an acquisition function', aq_functions,
                                     id={'type': 'input-acquisition', 'index': index},
                                     kwargs={'value': aq_functions[0]['value']}),
                       width=7 if delete else 9)]
    
    hyper_input = [dbc.Input(id={'type': 'input-alpha', 'index': index})]
    if delete:
        hyper_input.append(dbc.Button('Delete', id={'type': 'aq_delete', 'index': index},
                                      className='me-2', color='danger', n_clicks=0))

    columns.append(dbc.Col([dbc.InputGroup(hyper_input),
                            dbc.FormText('Input a hyperparameter (optional)', style={'font-size': '0.7em'})]))

    return dbc.Row(id={'type': 'div-acquisition', 'index': index}, children=columns, align='center')


def setup_config(app, app_tabs):

    input_optimizer_seed = make_input('Optimizer Seed', 'Randomization seed for Bayesian optimizer', required=False,
                                      id='input-optimizer_seed',
                                      kwargs={'type': 'number', 'min': 1, 'max': 99999,
                                              'step': 1, 'placeholder': 'None'})
    
    options_f_transform = ['Standard', 'Power']
    
    input_f_transform = make_dropdown('Response Transform', 'Select transformation for response', options_f_transform,
                                      id='input-f_transform', kwargs={'value': options_f_transform[0]})
    
    options_surrogate = [{'label': 'Default Gaussian Process', 'value': 'GP'},
                         {'label': 'Uninformed Gaussian Process', 'value': 'GPflat'},
                         {'label': 'Gaussian Process with Custom Prior', 'value': 'GPprior'},
                         {'label': 'Deep Kernel Learning Gaussian Process', 'value': 'DKL'}]
    
    input_surrogate = make_dropdown('Surrogate Model', 'Select a surrogate model', options_surrogate,
                                    id='input-surrogate', kwargs={'value': options_surrogate[0]['value']})
    
    input_m_batch = make_input('Number of Suggestions', 'The number of suggestions for the \
                               optimizer to provide (per acquisition function)', 1,
                               id='input-m_batch', kwargs={'type': 'number', 'min': 1, 'max': 16, 'step': 1})
    input_optim_sequential = make_switch('Optimize Sequentially', 'Switch for optimizing batch \
                                         suggestions sequentially or simultaneously', id='toggle-optim_sequential')
    input_optim_restarts = make_slider('Optimizer Restarts', 'Number of automatic restarts for \
                                        acquisition optimization', 1, 100,
                                       id='input-optim_restarts', kwargs={'value': 10})

    acquisitions = dbc.Container(dbc.Card(id='aq_inputs', children=[
        dbc.CardHeader(['Acquisition Functions', html.Div(dbc.FormText('Objective function of the experiment\
                                                                      search/selection',
                                                          style={'font-size': '0.7em'}))]),
        dbc.CardBody(html.Div(id='div-acquisition_all', children=[make_acquisition(index=0, delete=False)])),
        dbc.CardFooter(dbc.Button('Add', id='button-acquisition_add', className='me-2', color='secondary', n_clicks=0))
        ], style={'margin-top': '15px'}))
    
    general_options = dbc.Container(dbc.Card(dbc.CardBody(
        children=[input_surrogate, input_m_batch, input_optim_sequential])),
                                    style={'margin-top': '15px'})
    
    advanced_options = dbc.Container(make_collapse('adv_options',
                                                   [input_optimizer_seed, input_f_transform, input_optim_restarts],
                                                   'Advanced Options'),
                                     style={'margin-top': '15px'})

    # Config store
    config = dcc.Store(id='store-config')
    
    # Extra div for printing outputs, for troubleshooting
    troubleshoot = html.Div(id='debug-print-config')
    
    elements = [html.Br(), general_options, acquisitions, advanced_options, config, html.Hr(), troubleshoot]
    add_tab(app_tabs, elements, 'tab-config', 'Config')
    setup_config_callbacks(app)
    
    return


def setup_config_callbacks(app):
    
    @app.callback(
        Output('collapse-adv_options', 'is_open'),
        Input('button-collapse-adv_options', 'n_clicks'),
        State('collapse-adv_options', 'is_open'),
    )
    def toggle_collapse(n, is_open):
        if n:
            return not is_open
        return is_open
    
    # Add acquisition function
    @app.callback(
        Output('div-acquisition_all', 'children'),
        Output('button-acquisition_add', 'n_clicks'),
        Input('button-acquisition_add', 'n_clicks'),
        State('div-acquisition_all', 'children'),
    )
    def add_aq(add_clicked, aq_funs):
        if add_clicked:
            aq_funs.append(make_acquisition(index=len(aq_funs)+1))
        return aq_funs, 0
    
    # Delete acquisition function
    @app.callback(
        Output({'type': 'div-acquisition', 'index': MATCH}, 'children'),
        Output({'type': 'aq_delete', 'index': MATCH}, 'n_clicks'),
        Input({'type': 'aq_delete', 'index': MATCH}, 'n_clicks'),
        State({'type': 'div-acquisition', 'index': MATCH}, 'children'),
    )
    def func(del_clicked, aq_match):
        if del_clicked:
            aq_match = None
        return aq_match, 0
    
    # Store all of the input and settings into config
    @app.callback(
        Output('store-config', 'data'),
        Input('input-response_name', 'value'),
        Input('input-optimizer_seed', 'value'),
        Input('input-f_transform', 'value'),
        Input('input-surrogate', 'value'),
        Input('input-m_batch', 'value'),
        Input('toggle-optim_sequential', 'value'),
        Input('input-optim_restarts', 'value'),
        Input({'type': 'input-acquisition', 'index': ALL}, 'value'),
        Input({'type': 'input-alpha', 'index': ALL}, 'value'),
        prevent_initial_call=True
    )
    def compile_config(response_name, optimizer_seed, f_transform, surrogate,
                       m_batch, optim_sequential, optim_restarts, aq, alpha):
        
        config = {}
        
        config['response_name'] = response_name
        config['optimizer_seed'] = int(optimizer_seed) if optimizer_seed is not None else None
        config['surrogate_params'] = {'f_transform': f_transform, 'surrogate': surrogate}
        # TODO: Re-implement hyperparmeters based on selection in alpha
        config['aq_params'] = {'optim_sequential': optim_sequential, 'optim_restarts': optim_restarts,
                               'm_batch': m_batch, 'acquisition': aq}
        config['verbose'] = 0
        
        return config
    
    @app.callback(
        Output('debug-print-config', 'children'),
        Input('store-config', 'data'),
    )
    def troubleshoot_config(config):
        return
    
    return
