import dash_bootstrap_components as dbc
from dash import dcc, html, Dash, dash_table, callback, Output, Input, State, ALL, MATCH
from dash.dash_table.Format import Format, Scheme
import dash_daq as daq
import pandas as pd

from obsidian.parameters import ParamSpace, Param_Categorical, Param_Ordinal, Param_Continuous
from obsidian.optimizer import BayesianOptimizer


def center(element):
    return html.Div(html.Div(element, style={'display': 'inline-block'}), style={'textAlign': 'center'})


def load_optimizer(config, opt_save):
    # Note: it doesn't need config anymore?
    optimizer = BayesianOptimizer.load_state(opt_save)
    return optimizer


def add_tab(target, elements, id, label):
    target.children = list(target.children) if target.children else []
    tab = dbc.Tab(elements, label=label, id=id)
    target.children.append(tab)
    return target.children


def make_input(property_name, help_text, default_value=None, id=None, kwargs={}, required=True):
    components = [
        dbc.Label(property_name, style={'font-weight': 'bold', 'font-size': '0.8em'}),
        dbc.Input(value=default_value, id=f'input-[{property_name}]' if id is None else id,
                  debounce=True, required=required, **kwargs),
        html.Div(f'{help_text}', style={'font-size': '0.7em'}),
    ]
           
    return html.Div(components, className='mb-4')


def make_dropdown(property_name, help_text, options=[], id=None, kwargs={}):
    components = [
        dbc.Label(property_name, style={'font-weight': 'bold', 'font-size': '0.8em'}),
        dcc.Dropdown(options, id=f'input-[{property_name}]' if id is None else id, clearable=False, **kwargs),
        dbc.FormText(f'{help_text}', style={'font-size': '0.7em'}),
        ]

    return html.Div(components, className='mb-4')


def make_switch(property_name, help_text, id=None, kwargs={}):
    components = [
        dbc.Label(property_name, style={'font-weight': 'bold', 'font-size': '0.8em'}),
        html.Div(dbc.Switch(id=f'toggle-[{property_name}]' if id is None else id, value=True, **kwargs)),
        dbc.FormText(f'{help_text}', style={'font-size': '0.7em'}),
    ]

    return html.Div(components)


def make_slider(property_name, help_text, min, max, id=None, kwargs={}):
    components = [
        dbc.Label(property_name, style={'font-weight': 'bold', 'font-size': '0.8em'}),
        dcc.Slider(min, max, id=f'input-[{property_name}]' if id is None else id, **kwargs),
        dbc.FormText(f'{help_text}', style={'font-size': '0.7em'}),
    ]

    return html.Div(components)


def make_knob(property_name, help_text, min, max, id=None, kwargs={}):
    components = [
        dbc.Label(property_name, style={'font-weight': 'bold', 'font-size': '0.8em'}),
        daq.Knob(min=min, max=max, id=f'input-[{property_name}]' if id is None else id,  **kwargs),
        dbc.FormText(f'{help_text}', style={'font-size': '0.7em'}),
    ]

    return html.Div(components)


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
                 style={'textAlign': 'center'}),
        dbc.Collapse(contents, id=f'collapse-{id}', is_open=False)
    ]
    return dbc.Card(dbc.CardBody(components))
