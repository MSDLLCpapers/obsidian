from .utils import add_tab, make_input, make_dropdown, make_switch, make_slider, make_knob, make_table, make_collapse
from .utils import center
import dash_bootstrap_components as dbc
from dash import dcc, html, Dash, dash_table, callback, Output, Input, State, ALL, MATCH
import pandas as pd
import base64
import io
from dash.exceptions import PreventUpdate


def setup_data(app, app_tabs, default_data):
    
    # Data upload
    uploader = dcc.Upload(id='uploader-X0',
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
    
    # Example data download
    template_downloader = html.Div(children=[dbc.Button('Download Template Data', id='button-download_data',
                                                        className='me-2', color='primary'),
                                             dcc.Download(id='downloader-X0_template')],
                                   style={'textAlign': 'center'})
    
    # Data preview
    preview = html.Div(id='table-X0', children=dbc.Card(
        [dbc.CardHeader('Input Data'),
         dbc.CardBody(make_table(default_data, fill_width=True), id='table-X0-body'),
         dbc.CardFooter(html.Div([
             html.I(id='info-data', className='bi bi-info-circle-fill me-2'),
             dbc.Tooltip('Input data must be a CSV file and must include column headers for the input\
                         parameters and response variable(s), with one row per observation. Download\
                         template data (left) for example.',
                         target='info-data', placement='top', style={'text-transform': 'none'}),
             dbc.FormText('Example Data', color='info', style={'font-size': '1em', 'font-style': 'italic'},
                          id='table-X0-footer')
             ],
             style={'textAlign': 'center'}))
         ]))
    
    preview_uploader = dbc.Row([dbc.Col([uploader, template_downloader], width=4),
                                dbc.Col(preview, width=8)],
                               style={'margin-top': '15px'})
    
    # Data store
    storage_X0 = dcc.Store(id='store-X0', data=default_data.to_dict())
    storage_X0_template = dcc.Store(id='store-X0_template', data=default_data.to_dict())
    
    # Parameter space table
    xspace = html.Div([html.Div(id='div-xspace', children=[])])

    # Selecting the response variable
    ycol = dbc.Row(dbc.Col(dbc.Card([dbc.CardHeader([html.I(id='info-response_col',
                                                            className='bi bi-info-circle-fill me-2'),
                                                     dbc.Tooltip('The response is the variable which is\
                                                                 chosen for optimization (e.g. yield, productivity);\
                                                                 it is the target of the objective function before\
                                                                 including optional transformations.',
                                                                 target='info-response_col', placement='top',
                                                                 style={'text-transform': 'none'}),
                                                     'Response Selection']),
                   dbc.CardBody(html.Div(id='div-response_name',
                                         children=[make_dropdown('Data Column',
                                                                 'Select column for response',
                                                                 default_data.columns,
                                                                 id='input-response_name',
                                                                 kwargs={'value': default_data.columns[-1]})]))
                    ]), width=4), justify='center')
    
    # Extra div for printing outputs, for troubleshooting
    troubleshoot = html.Div(id='debug-print-data')
    
    # Add all of these elements to the app
    elements = [html.Br(), preview_uploader, html.Hr(), ycol, xspace, storage_X0,
                storage_X0_template, html.Hr(), troubleshoot]
    add_tab(app_tabs, elements, 'tab-data', 'Data')
    setup_data_callbacks(app)
    
    return


def setup_data_callbacks(app):
    
    # Save the uploaded data into the data-store
    @app.callback(
        Output('store-X0', 'data'),
        Input('uploader-X0', 'contents'),
    )
    def save_X0(contents):
        if contents is None:
            raise PreventUpdate
        content_type, content_string = contents.split(',')
        decoded = base64.b64decode(content_string)
        df = pd.read_csv(io.StringIO(decoded.decode('utf-8')))
        return df.to_dict()
    
    # View the uploaded data in a Data Table
    @app.callback(
        Output('table-X0-body', 'children'),
        Output('table-X0-footer', 'children'),
        Input('store-X0', 'data'),
        State('uploader-X0', 'filename')
    )
    def preview_X0(data, filename):
        df = pd.DataFrame(data)

        return make_table(df, fill_width=True), filename
    
    # Download template data
    @app.callback(
        Output('downloader-X0_template', 'data'),
        Input('button-download_data', 'n_clicks'),
        State('store-X0_template', 'data'),
        prevent_initial_call=True  # otherwise, the download CSV appears right away
    )
    def download_X0_template(n_clicks, data):
        df = pd.DataFrame(data)
        return dcc.send_data_frame(df.to_csv, 'Example_Data.csv', index=False)
    
    # Select a response variable
    @app.callback(
        Output('input-response_name', 'options'),
        Output('input-response_name', 'value'),
        Input('store-X0', 'data')
    )
    def choose_col(data):
        df = pd.DataFrame(data)
        return df.columns, df.columns[-1]
            
    # Select types for each input variable
    @app.callback(
        Output('div-xspace', 'children'),
        Input('store-X0', 'data'),
        Input('input-response_name', 'value')
    )
    def update_xspace_types(data, ycol):
        
        df_xspace = pd.DataFrame(data)
        xcols = [x for x in df_xspace.columns if x != ycol]
        
        #param_types = ['Numeric', 'Categorical', 'Ordinal']
        param_types = ['Numeric', 'Categorical'] # Note: the Ordinal var may cause fitting issues
        
        cols = []
        for i, x in enumerate(xcols):
            cols.append(dbc.Col(children=dbc.Card([
                dbc.CardHeader(f'{x}'),
                dbc.CardBody([make_dropdown('Type', f'Select parameter type for {x}', param_types,
                                            id={'type': 'input-param_type', 'index': x},
                                            kwargs={'value': param_types[0]}),
                              html.Div(id={'type': 'div-param_vals', 'index': x}, children=[]),
                              dcc.Store(id={'type': 'store-param_xspace', 'index': x}, data={})],)
                ],
                color='primary', outline=True), width=2))
 
        return dbc.Container(dbc.Row(cols, style={'margin-top': '15px'},
                                     justify='center'),
                             fluid=True)
    
    # Select min, max, and/or categories for each input variable
    @app.callback(
        Output({'type': 'div-param_vals', 'index': MATCH}, 'children'),
        Input({'type': 'input-param_type', 'index': MATCH}, 'value'),
        State({'type': 'input-param_type', 'index': MATCH}, 'id'),
        State('store-X0', 'data'),
        State('input-response_name', 'value')
    )
    def update_xspace_vals(param_type, param_id, data, ycol):
        
        df = pd.DataFrame(data)
        x = param_id['index']
        ser_x = df[x]
        
        # All keys must be present, so Numeric still needs "None" categories, etc.
        if param_type == 'Numeric':
            choices = [
                make_input(f'{x} Min Value', f'Select optimization minimum for {x}', df[x].min(),
                           id={'type': 'input-param_min', 'index': x}),
                make_input(f'{x} Max Value', f'Select optimization maximum for {x}', df[x].max(),
                           id={'type': 'input-param_max', 'index': x}),
                dcc.Store(id={'type': 'store-param_categories', 'index': x}, data=None)
                ]
                           
        elif param_type in ['Categorical', 'Ordinal']:
            choices = [dbc.Card(dbc.CardBody([
                       dbc.InputGroup([dbc.Input(id={'type': 'input-new_category', 'index': x},
                                                 placeholder='New category'),
                                       dbc.Button('Add', id={'type': 'button-category_add', 'index': x},
                                                  className='me-2', color='primary'),
                                       dbc.Button('Clear All', id={'type': 'button-category_delete', 'index': x},
                                                  className='me-2', color='danger')]),
                       html.Hr(),
                       html.Div(id={'type': 'div-param_categories', 'index': x}, children=None,
                                style={'textAlign': 'center'})])),
                       dcc.Store(id={'type': 'store-param_categories', 'index': x},
                                 data=', '.join(list(ser_x.sort_values().astype('str').unique()))),
                       html.Div(id={'type': 'input-param_min', 'index': x}),
                       html.Div(id={'type': 'input-param_max', 'index': x}),
                       ]
        return choices

    # ategory management for categorical variables
    @app.callback(
        Output({'type': 'store-param_categories', 'index': MATCH}, 'data'),
        Output({'type': 'input-new_category', 'index': MATCH}, 'value'),
        Output({'type': 'button-category_add', 'index': MATCH}, 'n_clicks'),
        Output({'type': 'button-category_delete', 'index': MATCH}, 'n_clicks'),
        Input({'type': 'button-category_add', 'index': MATCH}, 'n_clicks'),
        Input({'type': 'button-category_delete', 'index': MATCH}, 'n_clicks'),
        State({'type': 'store-param_categories', 'index': MATCH}, 'data'),
        State({'type': 'input-new_category', 'index': MATCH}, 'value'),
        State({'type': 'store-param_categories', 'index': MATCH}, 'id'),
        State('store-X0', 'data'),
        prevent_initial_call=True
    )
    def cat_buttons(add_button, clear_button, current_cats, new_cat, param_id, data):
        
        x = param_id['index']
        df = pd.DataFrame(data)
        ser_x = df[x]
        
        if clear_button:
            return ', '.join(list(ser_x.sort_values().astype('str').unique())), '', 0, 0
        
        if add_button:
            if new_cat != '':
                if current_cats is None:
                    current_cats = new_cat
                else:
                    if new_cat not in current_cats:
                        current_cats += ', '+new_cat
            return current_cats, '', 0, 0
    
    # Preview categories for categorical variables
    @app.callback(
        Output({'type': 'div-param_categories', 'index': MATCH}, 'children'),
        Input({'type': 'store-param_categories', 'index': MATCH}, 'data'),
    )
    def preview_cats(current_cats):
        if current_cats is None:
            return None
        else:
            cat_list = current_cats.split(', ')
            
            preview = []
            for cat in cat_list:
                preview += [html.Div(cat, style={'font-size': '0.8em'})]
            return preview

    @app.callback(
        Output('debug-print-data', 'children'),
        Input('store-X0', 'data'),
    )
    def troubleshoot_config(data):
        return
    
    return
