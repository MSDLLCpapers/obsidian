from .utils import add_tab, make_input, make_dropdown, make_switch, make_slider, make_knob, make_table, make_collapse
import dash_bootstrap_components as dbc
from dash import dcc, html, Dash, dash_table, callback, Output, Input, State, ALL, MATCH

import pandas as pd

from obsidian.parameters import ParamSpace, Param_Categorical, Param_Ordinal, Param_Continuous
#from obsidian.optimizer import BayesianOptimizer
from obsidian.campaign import Campaign
from obsidian.parameters import Target

from obsidian.plotting.plotly_plotting import parity_plot
from .utils import load_optimizer, center


def setup_optimize(app, app_tabs):
    
    fit_div = dbc.Container([
        dbc.Spinner(color="primary", children=[
            dbc.Button('Fit', id='button-fit', n_clicks=0, size='lg')
        ]),
        html.Br(),
        html.Br(),
        dbc.Card([
            dbc.CardHeader('Regression Statistics'),
            dbc.CardBody([
                html.Div(id='div-fit', children=[])
                ]),
            ]),
        html.Div(id='graph-parity', children=[])
        ],
        style={'textAlign': 'center'}
    )
    
    predict_div = dbc.Container([
        dbc.Spinner(color="primary", children=[
            dbc.Button('Optimize', id='button-predict', n_clicks=0, size='lg'),
            ]),
        html.Br(),
        html.Br(),
        dbc.Card([
            dbc.CardHeader('Optimal Experiments'),
            dbc.CardBody([
                html.Div(id='div-predict', children=[], style={})
                ]),
            ]),
        ],
        style={'textAlign': 'center'}
    )
    
    storage_fit = dcc.Store(id='store-fit', data=None)
    
    # candidates store
    store_candidates = dcc.Store(id='store-candidates', data={})
    
    # Suggested candidates download
    candidates_downloader = html.Div(children=[dbc.Button('Download Suggested Candidates', id='button-download_candidates',
                                                        className='me-2', color='primary'),
                                             dcc.Download(id='downloader-candidates')],
                                   style={'textAlign': 'center','margin-top': '15px'})
    
    # Add all of these elements to the app
    columns = dbc.Row([dbc.Col(fit_div, width=6), dbc.Col([predict_div, candidates_downloader], width=6)])
    elements = [html.Br(), columns, storage_fit, store_candidates]
    add_tab(app_tabs, elements, 'tab-optimize', 'Optimize')
    setup_optimize_callbacks(app)
    
    return


def setup_optimize_callbacks(app):
    
    @app.callback(
        Output('button-fit', 'n_clicks'),
        Output('store-fit', 'data'),
        Input('button-fit', 'n_clicks'),
        State('store-config', 'data'),
        State('store-X0', 'data'),
        prevent_initial_call=True
    )
    def fit_optimizer(fit_clicked, config, X0):
        
        if config['xspace'] == {}:
            return 0, None
        
        xspace = []
        for param_xspace in config['xspace']:
            if param_xspace['Type']=='Numeric':
                this_para = Param_Continuous(param_xspace['Name'], param_xspace['Low'], param_xspace['High'])
            elif param_xspace['Type']=='Categorical':
                this_para = Param_Categorical(param_xspace['Name'], param_xspace['Categories'])
            else:
                this_para = Param_Ordinal(param_xspace['Name'], param_xspace['Categories'])
            xspace.append(this_para)
        X_space = ParamSpace(xspace)       
        my_campaign = Campaign(X_space)
        my_campaign.add_data(pd.DataFrame(X0))
        target = Target(config['response_name'], aim='max')
        my_campaign.set_target(target)
        my_campaign.fit()
        
        return 0, my_campaign.optimizer.save_state()
    
    @app.callback(
        Output('div-fit', 'children'),
        Input('store-fit', 'data'),
        State('store-config', 'data'),
        State('uploader-X0', 'filename')
    )
    def fit_statistics(opt_save, config, filename):
        if opt_save is None:
            return dbc.Alert('Model must be fit first', color='info')
        
        optimizer = load_optimizer(config, opt_save)
        
        fit_stats = dbc.ListGroup(
            [
             dbc.ListGroupItem(['Model Type: ', f'{optimizer.surrogate_type}']),
             dbc.ListGroupItem(['Data Name: ', filename]),
             dbc.ListGroupItem(['R', html.Sup('2'), ' Score: ', f'{optimizer.surrogate[0].score: .4g}']), # for SOO only
             dbc.ListGroupItem(['Marginal Log Likelihood: ', f'{optimizer.surrogate[0].loss: .4g}']), # for SOO only
            ], flush=True
        )
        
        return fit_stats
    
    @app.callback(
        Output('graph-parity', 'children'),
        Input('store-fit', 'data'),
        State('store-config', 'data'),
        prevent_initial_call=True
    )
    def graph_parity_plot(opt_save, config):
        
        if opt_save is None:
            return None
        
        optimizer = load_optimizer(config, opt_save)
        pplot = parity_plot(optimizer)
        pplot.update_layout(height=400, width=600)
        
        graph = dcc.Graph(figure=pplot)
        
        return center(graph)
    
    @app.callback(
        Output('div-predict', 'children'),
        Output('div-predict', 'style'),
        Output('button-predict', 'n_clicks'),
        Output('store-candidates', 'data'),
        Input('button-predict', 'n_clicks'),
        State('store-config', 'data'),
        State('store-fit', 'data'),
        # prevent_initial_call = True
    )
    def predict_optimizer(predict_clicked, config, opt_save):
        
        if opt_save is None:
            if predict_clicked:
                alert_color = 'danger'
            else:
                alert_color = 'info'
            return dbc.Alert('Model must be fit first', color=alert_color), {}, predict_clicked, {}
        
        optimizer = load_optimizer(config, opt_save)
        X_suggest, eval_suggest = optimizer.suggest(**config['aq_params'])
        df_suggest = pd.concat([X_suggest, eval_suggest], axis=1)
        df_suggest.insert(loc=0, column='CandidatesID', value=df_suggest.index)
        tables = [center(make_table(df_suggest))]
        
        return tables, {'overflow-x': 'scroll'}, 0, df_suggest.to_dict()

    # Download Suggested Candidates
    @app.callback(
        Output('downloader-candidates', 'data'),
        Input('button-download_candidates', 'n_clicks'),
        State('store-candidates', 'data'),
        prevent_initial_call=True  # otherwise, the download CSV appears right away
    )
    def download_candidates(n_clicks, data):
        df = pd.DataFrame(data)
        return dcc.send_data_frame(df.to_csv, 'Suggested_Candidates.csv', index=False)
    
    return
