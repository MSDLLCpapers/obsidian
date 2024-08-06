from .utils import add_tab, make_input, make_dropdown, make_switch, make_slider, make_knob, make_table, make_collapse
from .utils import load_optimizer, center
import dash_bootstrap_components as dbc
from dash import dcc, html, Dash, dash_table, callback, Output, Input, State, ALL, MATCH

#from obsidian.experiment import ParamSpace
#from obsidian.optimizer import BayesianOptimizer
#from obsidian.plotting.plotly_plotting import parity_plot

import pandas as pd
import base64
import io
from dash.exceptions import PreventUpdate

from obsidian.experiment import ExpDesigner


def setup_predict(app, app_tabs):
    # Parameter Sapce Table Review
    xspace_df_div = dbc.Container([
        dbc.Button('Refresh Parameter Space Table', id='button-config', n_clicks=0, size='lg'),
        html.Br(),
        html.Br(),
        dbc.Card([
            dbc.CardHeader('Parameter Space'),
            dbc.CardBody([
                html.Div(id='div-xspace_df', children=[], style={'overflow-x': 'scroll'})
                ]),
            ]),
        ],
        style={'textAlign': 'center'}
    )

    template_div = dbc.Container([
        dbc.Button('Generate Template', id='button-template', n_clicks=0, size='lg'),
        html.Br(),
        html.Br(),
        dbc.Card([
            dbc.CardHeader('template'),
            dbc.CardBody([
                html.Div(id='div-template', children=[], style={'overflow-x': 'scroll'})
                ]),
            ]),
            ],
        style={'textAlign': 'center'}
    )
    # candidates store
    store_template = dcc.Store(id='store-template', data={})
    
    # template download
    template_downloader = html.Div(children=[dbc.Button('Download Template Candidates', id='button-download_template',
                                                        className='me-2', color='primary'),
                                             dcc.Download(id='downloader-template')],
                                   style={'textAlign': 'center','margin-top': '15px'})

    default_data = pd.DataFrame()
    
    # Data upload
    uploader_1 = dcc.Upload(id='uploader-X1',
                          children=html.Div(['Upload Data: Drag and Drop or ',
                                             html.A('Select Files')]),
                          style={
                              'width': '100%', 'height': '60px',
                              'lineHeight': '60px', 'borderWidth': '1px',
                              'borderStyle': 'dashed', 'borderRadius': '5px',
                              'textAlign': 'center', 'margin': '10px'
                              },
                          multiple=False,
                          filename='Example Data'
                          )
    
    # Data store
    storage_X1 = dcc.Store(id='store-X1', data=default_data.to_dict())
    
    # Data+Prediction preview
    preview_1 = html.Div(id='table-X1', children=dbc.Card(
        [dbc.CardHeader('Input Data and Predictions'),
         dbc.CardBody(make_table(default_data, fill_width=True), id='table-X1-body'),
         dbc.CardFooter(html.Div([
             html.I(id='info-data_1', className='bi bi-info-circle-fill me-2'),
             dbc.Tooltip('Input data must be a CSV file and must include column headers for the input\
                         parameters and response variable(s), with one row per observation. Download\
                         template data (left) for example.',
                         target='info-data_1', placement='top', style={'text-transform': 'none'}),
             dbc.FormText('Example Data', color='info', style={'font-size': '1em', 'font-style': 'italic'},
                          id='table-X1-footer')
             ],
             style={'textAlign': 'center'}))
         ]))

    row1 = dbc.Row([dbc.Col([xspace_df_div], width=4),
                    dbc.Col([template_div, template_downloader], width=8)],style={'margin-top': '15px'})
    row2 = dbc.Row([uploader_1, preview_1], style={'margin-top': '15px'})
    
    # Add all of these elements to the app
    elements = [html.Br(), store_template, row1, row2, storage_X1]
    add_tab(app_tabs, elements, 'tab-predict', 'Predict')
    setup_predict_callbacks(app)
    
    return


def setup_predict_callbacks(app):
    # Display the x_space as a dataframe
    @app.callback(
        Output('button-config', 'n_clicks'),
        Output('div-xspace_df', 'children'),
        Output('div-xspace_df', 'style'),
        Input('button-config', 'n_clicks'),
        State('store-config', 'data'),
    )
    def config_tableView(clicked, config):
        if config is None:
            return 0, None, {'overflow-x': 'scroll'}
        #print(config)
        df_xspace = pd.DataFrame(config['xspace'])
        tables = [center(make_table(df_xspace))]
        return 0, tables, {'overflow-x': 'scroll'}
    
    # Generate New Data template according to x_space
    @app.callback(
        Output('button-template', 'n_clicks'),
        Output('div-template', 'children'),
        Output('store-template', 'data'),
        Input('button-template', 'n_clicks'),
        State('store-fit', 'data'),
        State('store-config', 'data'),
    )
    def config_InputTemplate(clicked, opt_save, config):
        if config is None:
            return 0, None, {}
        optimizer = load_optimizer(config, opt_save)
        designer = ExpDesigner(optimizer.X_space, seed=0)
        df_template = designer.initialize(3, 'LHS')
        tables = [center(make_table(df_template))]
        return 0, tables, df_template.to_dict()
    # Download Template
    @app.callback(
        Output('downloader-template', 'data'),
        Input('button-download_template', 'n_clicks'),
        State('store-template', 'data'),
        prevent_initial_call=True  # otherwise, the download CSV appears right away
    )
    def download_template(n_clicks, data):
        df = pd.DataFrame(data)
        return dcc.send_data_frame(df.to_csv, 'NewDataTemplate.csv', index=False)
    
    
    # Save the uploaded data into the data-store
    @app.callback(
        Output('store-X1', 'data'),
        Input('uploader-X1', 'contents'),
    )
    def save_X1(contents):
        if contents is None:
            raise PreventUpdate
        content_type, content_string = contents.split(',')
        decoded = base64.b64decode(content_string)
        df = pd.read_csv(io.StringIO(decoded.decode('utf-8')))
        return df.to_dict()
    
    # View the uploaded data in a Data Table
    @app.callback(
        Output('table-X1-body', 'children'),
        Output('table-X1-footer', 'children'),
        Input('store-X1', 'data'),
        State('uploader-X1', 'filename'),
        State('store-fit', 'data'),
        State('store-config', 'data'),
    )
    def preview_X1(data, filename,opt_save, config):
        df = pd.DataFrame(data)
        optimizer = load_optimizer(config, opt_save)
        preds = optimizer.predict(df)
        df_output = pd.concat([df.reset_index(drop=True), preds.reset_index(drop=True)], axis=1)
        return make_table(df_output, fill_width=True), filename
    
    return


