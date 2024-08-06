from .utils import add_tab, make_input, make_dropdown, make_switch, make_slider, make_knob, make_table, make_collapse
import dash_bootstrap_components as dbc
from dash import dcc, html, Dash, dash_table, callback, Output, Input, State, ALL, MATCH

import pandas as pd

from obsidian.experiment import ParamSpace
from obsidian.optimizer import BayesianOptimizer
from obsidian.plotting.plotly_plotting import parity_plot
from .utils import load_optimizer, center


def setup_help(app, app_infobar):
    
    app_infobar.children = dbc.Container([
        dbc.Row([
            dbc.Col(dbc.Button('Help', outline='True', color='warning', className='me-1', id='button-help'),
                    style={'textAlign': 'left'}, width={'size': 2}),
            dbc.Col(html.Div(dbc.Badge('v0.1.1', color='primary', className='me-1'), style={'textAlign': 'center'}),
                    width={'size': 2}),
            dbc.Col(dbc.Button('Contact', outline='True', color='secondary', className='me-1', id='button-contact'),
                    style={'textAlign': 'right'}, width={'size': 2}),
            ], justify='center'),
        dbc.Modal([
            dbc.ModalHeader(dbc.ModalTitle('Contact Us')),
            dbc.ModalBody([
                dbc.Button('Email Support', href='', external_link=True,
                           color='primary', className='me-1'),
                dbc.Button('Visit Our Site', href='', target='_blank', external_link=True,
                           color='primary', className='me-1')
                ], style={'textAlign': 'center'}),
            dbc.ModalFooter([])
            ], id='modal-contact', is_open=False, size='xl', centered=True),
        dbc.Modal([
            dbc.ModalHeader(dbc.ModalTitle('Help: APO with the obsidian Web App')),
            dbc.ModalBody('Coming soon...'),
            dbc.ModalFooter([])
            ], id='modal-help', is_open=False, size='xl', centered=True)
        ], fluid=True)
    
    setup_help_callbacks(app)
    
    return


def setup_help_callbacks(app):
    
    @app.callback(
        Output('modal-help', 'is_open'),
        Input('button-help', 'n_clicks'),
        State('modal-help', 'is_open')
    )
    def modal_help(click, is_open):
        if click:
            return ~is_open
        return is_open
    
    @app.callback(
        Output('modal-contact', 'is_open'),
        Input('button-contact', 'n_clicks'),
        State('modal-contact', 'is_open')
    )
    def modal_contact(click, is_open):
        if click:
            return ~is_open
        return is_open
    
    return
