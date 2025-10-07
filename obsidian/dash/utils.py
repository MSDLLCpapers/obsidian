import traceback

import dash_bootstrap_components as dbc
from dash import dcc, html, Dash, dash_table, callback, Output, Input, State, ALL, MATCH
from dash.dash_table.Format import Format, Scheme
import dash_daq as daq
import pandas as pd

from obsidian.parameters import ParamSpace, Param_Categorical, Param_Ordinal, Param_Continuous
from obsidian.optimizer import BayesianOptimizer


def center(element):
    return html.Div(html.Div(element, className="d-inline-block"), className="text-center")


def load_optimizer(config, opt_save):
    optimizer = BayesianOptimizer.load_state(opt_save)
    return optimizer


def load_Xspace(config, Xspace_save):
    X_space = ParamSpace.load_state(Xspace_save)
    return X_space


def add_tab(target, elements, id, label):
    target.children = list(target.children) if target.children else []
    tab = dbc.Tab(elements, label=label, id=id)
    target.children.append(tab)
    return target.children

def make_input(
    property_name, help_text, default_value=None, id=None, kwargs={}, required=True
):
    components = [
        dbc.Label(property_name, className="obsd-form-label"),
        dbc.Input(
            value=default_value,
            id=f"input-[{property_name}]" if id is None else id,
            debounce=True,
            required=required,
            **kwargs,
        ),
        html.Div(help_text, className="obsd-help-text"),
    ]
    return html.Div(components, className="mb-4 obsd-input-row")


def make_dropdown(property_name, help_text, options=[], id=None, kwargs={}):
    components = [
        dbc.Label(property_name, className="obsd-form-label"),
        dcc.Dropdown(
            options,
            id=f"input-[{property_name}]" if id is None else id,
            clearable=False,
            **kwargs,
        ),
        dbc.FormText(help_text, className="obsd-help-text"),
    ]
    return html.Div(components, className="mb-4 obsd-input-row")


def make_switch(property_name, help_text, id=None, kwargs={}):
    components = [
        dbc.Label(property_name, className="obsd-form-label"),
        html.Div(
            dbc.Switch(
                id=f"toggle-[{property_name}]" if id is None else id,
                value=True,
                **kwargs,
            )
        ),
        dbc.FormText(help_text, className="obsd-help-text"),
    ]
    return html.Div(components, className="obsd-input-row")


def make_slider(property_name, help_text, min, max, id=None, kwargs={}):
    components = [
        dbc.Label(property_name, className="obsd-form-label"),
        dcc.Slider(
            min, max, id=f"input-[{property_name}]" if id is None else id, **kwargs
        ),
        dbc.FormText(help_text, className="obsd-help-text"),
    ]
    return html.Div(components, className="obsd-input-row")


def make_knob(property_name, help_text, min, max, id=None, kwargs={}):
    components = [
        dbc.Label(property_name, className="obsd-form-label"),
        daq.Knob(
            min=min,
            max=max,
            id=f"input-[{property_name}]" if id is None else id,
            **kwargs,
        ),
        dbc.FormText(help_text, className="obsd-help-text"),
    ]
    return html.Div(components, className="obsd-input-row")

def make_table(df, fill_width=False):
    table = html.Div([dash_table.DataTable(data=df.to_dict('records'),
                                           columns=[{'id': c, 'name': c, 'type': 'numeric',
                                                     'format': {'specifier': '.5g'}} for c in df.columns],
                                           page_size=10, style_table={'overflowY': 'auto', 'overflowX': 'auto'},
                                           style_cell={'textAlign': 'center'}, style_header={'fontWeight': 'bold'},
                                           style_data_conditional=[{'if': {'row_index': 'odd'},
                                                                    'backgroundColor': 'rgb(220, 220, 220)'}],
                                           fill_width=fill_width,
                                           )], className='dbc')
    
    return table


def make_collapse(id, contents, label):
    components = [
        html.Div(dbc.Button(label, id=f'button-collapse-{id}', className='mb-3', color='primary', n_clicks=0),
                 className="text-center"),
        dbc.Collapse(contents, id=f'collapse-{id}', is_open=False)
    ]
    return dbc.Card(dbc.CardBody(components))

def is_input_empty(value):
    if isinstance(value, (list, tuple)):
        return len(value) == 0
    if isinstance(value, str):
        return value.strip() == ''

def error_message_handling(name, message, verbosity=1, tb=None):
    if verbosity >= 1:
        print(f'Updating "{name}" failed: {message}')
    if tb is not None and verbosity > 1:
        if isinstance(tb, str):
            print(tb)
        else:
            traceback.print_exc()