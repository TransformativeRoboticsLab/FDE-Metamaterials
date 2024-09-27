import base64
import threading
import time

import dash
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from dash import dcc, html
from dash.dependencies import Input, Output
from incense import ExperimentLoader
from utils import *

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


load_thread = threading.Thread(target=load_experiments_async, args=(loader, 'extremal', filter_tags))
load_thread.start()
    
app = dash.Dash(__name__)
app.layout = html.Div([
    html.H1('Meta-Top Analysis Dashboard'),
    html.Div([
        html.Div([
            html.Label('X-axis Metric'),
            dcc.Dropdown(
                id='x-axis-dropdown',
                options=[],
                value='Normed_Eigenvalue_0'
            ),
        ], style={'width': '20%', 'display': 'inline-block'}),
        html.Div([
            html.Label('Y-axis Metric'),
            dcc.Dropdown(
                id='y-axis-dropdown',
                options=[],
                value='Normed_Eigenvalue_1'
            ),
        ], style={'width': '20%', 'display': 'inline-block', 'marginLeft': '5%'}),
        html.Div([
            html.Label('Color by Config Parameter'),
            dcc.Dropdown(
                id='color-dropdown',
                options=[],
                value='basis_v'
            ),
        ], style={'width': '20%', 'display': 'inline-block', 'marginLeft': '5%'}),
        html.Div([
            dcc.Checklist(
                id='yx-line-toggle',
                options=[{'label': 'Plot y=x', 'value': 'plot_yx'}],
                value=['plot_yx']  # Empty by default (unchecked)
            )
        ]),
    ]),
    
        
    dcc.Interval(
        id='interval-component',
        interval=30*1000,  # in milliseconds
        n_intervals=0
    ),
    dcc.Graph(id='scatter-plot'),
    html.Img(id='hover-image', src='', style={'max-height': '400px', 'max-width': '400px'})
])

@app.callback(
    [Output('scatter-plot', 'figure'),
     Output('x-axis-dropdown', 'options'),
     Output('y-axis-dropdown', 'options'),
     Output('color-dropdown', 'options')],
    Input('x-axis-dropdown', 'value'),
    Input('y-axis-dropdown', 'value'),
    Input('color-dropdown', 'value'),
    Input('yx-line-toggle', 'value'),
    Input('interval-component', 'n_intervals')
)

def update_scatter_plot(x_metric, y_metric, color_param, toggle_values, n_intervals):

    global experiments_cache
    experiments = experiments_cache

    scatter_data = {
        'x': [],
        'y': [],
        color_param: [],
        'Run ID': [],
        'Mode': []
    }

    for e in experiments:
        if x_metric not in e.metrics or y_metric not in e.metrics:
            print(f"Skipping experiment {e.id} because it doesn't have the required metrics.")
            continue
        scatter_data['x'].append(e.metrics.get(x_metric, None).iloc[-1])
        scatter_data['y'].append(e.metrics.get(y_metric, None).iloc[-1])
        scatter_data[color_param].append(e.config.get(color_param, None))
        scatter_data['Run ID'].append(f'Run ID: {e.id}')  # Format run_id here
        mode = 'Unimode' if e.config.get('extremal_mode', None) == 1 else 'Bimode'
        scatter_data['Mode'].append(mode)

    df = pd.DataFrame(scatter_data)

    fig = px.scatter(
        df,
        x='x',
        y='y',
        color=color_param,
        hover_name='Run ID',
        hover_data=['Mode'],
        symbol='Mode',
        symbol_map={'Unimode': 'square', 'Bimode': 'triangle-up'},
    )
    
    if 'plot_yx' in toggle_values:
        fig.update_layout(
            shapes=[
                {
                    'type': 'line',
                    'x0': 0,
                    'y0': 0,
                    'x1': 1,
                    'y1': 1,
                    'line': {
                        'color': 'Black',
                        'width': 2,
                        'dash': 'dash'
                    },
                }
            ]
        )
    
    fig.update_layout(title=f"{len(experiments):d} Data Points", width=1000, height=1000)

    # x_lower = min(min(df['x']), -.05)
    # x_upper = max(max(df['x']), 1.05)
    # y_lower = min(min(df['y']), -.05)
    # y_upper = max(max(df['y']), 1.05)
    fig.update_xaxes(title_text=x_metric)
    # fig.update_xaxes(range=[x_lower, x_upper], title_text=x_metric)
    fig.update_yaxes(scaleanchor='x', scaleratio=1, title_text=y_metric)
    fig.update_traces(marker=dict(size=12), mode='markers')
    metric_dropdowns, config_dropdowns = update_dropdown_options(experiments)
    
    return fig, metric_dropdowns, metric_dropdowns, config_dropdowns

@app.callback(
    Output('hover-image', 'src'),
    Input('scatter-plot', 'hoverData')
)
def update_image_on_hover(hover_data):
    if hover_data:
        run_id = int(hover_data['points'][0]['hovertext'].split(': ')[1])
        image_data = get_image_from_experiment(loader, run_id)
        img_src = f"data:image/png;base64,{encode_image(image_data)}"
        return img_src
    return ''

if __name__ == '__main__':
    app.run_server(debug=True, host='localhost', port=8050)
