import threading
import time

import dash
import dash_bootstrap_components as dbc
import plotly.express as px
import plotly.graph_objects as go
from dash import dcc, html
from dash.dependencies import Input, Output
from incense import ExperimentLoader
from utils import *

# reference later
# https://medium.com/plotly/how-to-create-a-beautiful-interactive-dashboard-layout-in-python-with-plotly-dash-a45c57bb2f3c#:~:text=We%E2%80%99ll%20look%20at%20how%20to%20develop%20a%20dashboard%20grid%20and

loader = ExperimentLoader(mongo_uri='mongodb://localhost:27017', 
                            db_name='metatop')
filter_tags = ['bad']

experiments_cache = load_experiments(loader, 'extremal', filter_tags)
last_update_time = None

def load_experiments_async(loader, experiment_name, filter_tags=[], poll_interval=60):
    global experiments_cache, last_update_time
    while True:
        experiments_cache = load_experiments(loader, experiment_name, filter_tags)
        last_update_time = time.time()
        print(f"Updated experiments cache at {time.ctime(last_update_time)}.")
        time.sleep(poll_interval)

app = dash.Dash(__name__)
app.layout = dbc.Container([
    # Navigation
    html.Div(
        [
        # Top Info
        html.Div(
            [
            html.H1("Welcome"),
            html.P("This is a dashboard for analyzing the results of the Meta-Top experiments."),
            ],
            style={'vertical-alignment': 'top',
                   'border': '1px solid black',
                #    'height': 260,
                   },
            className='level-1'),
        # Metrics
        html.Div([
            html.H2("Metrics"),
            # X-axis Metric
            html.Div([
                html.Label('X-axis'),
                dcc.Dropdown(id='x-axis-dropdown',
                             options=[],
                             value='Normed_Eigenvalue_0',
                             optionHeight=40,
                             clearable=False),
                ]),
            # Y-axis Metric
            html.Div([
                html.Label('Y-axis'),
                dcc.Dropdown(id='y-axis-dropdown',
                             options=[],
                             value='Normed_Eigenvalue_1',
                             clearable=False),
                ]),
        ],
            style={'margin-left': 15,
                   'margin-right': 15,
                   'margin-top': 30,
                #    'height': 300,
                    'padding': 15,
                   'border': '1px solid black',
                   },
            className='level-1'),
        html.Div([
            html.H2("Filters"),
            html.Div([
                html.Label('Marker Filters'),
                dcc.Dropdown(id='marker-filters',
                             options=[],
                             value=['extremal_mode'],
                             multi=True,
                             clearable=True),
            ]),
            html.Div([
                html.Label('Color Filters'),
                dcc.Dropdown(id='color-filters',
                             options=[],
                             value=['basis_v'],
                             multi=True,
                             clearable=True),
            ]),
            html.Div([
                html.Label("Material Properties"),
                html.P("TBD")
                ]
            ),
        ],
                 style={'margin-left': 15,
                        'margin-right': 15,
                        'margin-top': 30,
                        # 'height': 300,
                        'padding': 15,
                        'border': '1px solid black',
                        }
        ),
            html.Div([
                html.H2("Plot Controls"),
                dcc.Checklist(id='yx-line-toggle',
                              options=[
                                  {'label': 'Show y=x line', 'value': 'plot_yx'}],
                              value=['plot_yx'],
                )
            ],
                     style={'margin': 15,
                            'border': '1px solid black',
                            'padding': 15,}),
        ],
        style={'width': 340,
               'margin-left': 35,
               'margin-top': 35,
               'margin-bottom': 35,
               }),
    # Main content
    html.Div([
        # Graph
        html.Div(dcc.Graph(id='scatter-plot',), 
                 style={'width': 1200,
                        'display': 'flex',
                        'border': '1px solid black',}),
        # Hover info
        html.Div([
            # Image
                html.Img(id='hover-image',
                         src='assets/placeholder.jpg',
                         style={'max-height': '200px',
                                'max-width': '200px',}),
                # html.Pre(id='hover-text',
                #          style={'width': '200px',
                #                 'padding': '10px'}),
            ],
            # Text
            style={'width': 400,
                   'display': 'flex',}),
        ],
        style={'width': 1400,
                'margin-top': 35,
                'margin-right': 35,
                'margin-bottom': 35,
                'display': 'flex',
                'border': '1px solid black',},
        className='level-1'),
    ],
    fluid=True,
    style={'display': 'flex',
           'border': '1px solid black',},
    className='dashboard-container',
    id='dashboard-container',)

@app.callback(
    [Output('scatter-plot',    'figure'),
     Output('x-axis-dropdown', 'options'),
     Output('y-axis-dropdown', 'options'),
     Output('marker-filters',  'options'),
     Output('color-filters',   'options')],
    Input('x-axis-dropdown', 'value'),
    Input('y-axis-dropdown', 'value'),
    Input('marker-filters',  'value'),
    Input('color-filters',   'value'),
    Input('yx-line-toggle',  'value'),
)
def update_scatter_plot(x_metric, y_metric, marker_filters, color_filters, plot_yx_line):

    global experiments_cache
    experiments = experiments_cache

    df = build_scatter_dataframe(x_metric, y_metric, experiments, marker_filters=marker_filters, color_filters=color_filters)
    
    # symbol_map = build_symbol_map(experiments, marker_filter_value)
    print(marker_filters)
    print(df.head())
    fig = px.scatter(
        df,
        x='x',
        y='y',
        hover_name='Run ID',
        symbol='marker',
        color='color',
    )
    
    fig.update_traces(marker=dict(size=12,
                                  line=dict(width=2,
                                            color='DarkSlateGrey')),
                      selector=dict(mode='markers'))
    
    customize_figure(x_metric, y_metric, experiments, fig, plot_yx_line, size=(500,500))

    metric_dropdowns, config_dropdowns = update_dropdown_options(experiments)
    basis_dropdowns = get_fields(experiments, 'basis_v')
    
    return fig, metric_dropdowns, metric_dropdowns, config_dropdowns, config_dropdowns


@app.callback(
    Output('hover-image', 'src'),
    #  Output('hover-text', 'children')],
    Input('scatter-plot', 'hoverData')
)
def update_on_hover(hover_data):
    if hover_data:
        run_id = int(hover_data['points'][0]['hovertext'].split(': ')[1])
        exp_img = get_image_from_experiment(loader, run_id)
        img_src = f"data:image/png;base64,{encode_image(exp_img)}"
        # exp_txt = get_text_from_experiment(loader, run_id)
        return img_src#, ''
    return ''#, 'Hover over a point to see details here.'

def main():
    load_thread = threading.Thread(target=load_experiments_async, args=(loader, 'extremal', filter_tags))
    load_thread.daemon = True
    load_thread.start()
    app.run_server(debug=True, host='localhost', port=8050)

if __name__ == '__main__':
    main()