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
                id='color-multiselect',
                options=[],
                value=['basis_v', 'objective_type'],
                multi=True
            ),
        ], style={'width': '20%', 'display': 'inline-block', 'marginLeft': '5%'}),
        html.Div([
            dcc.Checklist(
                id='yx-line-toggle',
                options=[{'label': 'Plot y=x', 'value': 'plot_yx'}],
                value=['plot_yx']  # Empty by default (unchecked)
            )
        ]),
        html.Button('Refresh Data', id='refresh-button', n_clicks=0)
    ]),
    dcc.Graph(id='scatter-plot'),
    html.Img(id='hover-image', src='', style={'max-height': '400px', 'max-width': '400px'})
])

@app.callback(
    [Output('scatter-plot', 'figure'),
     Output('x-axis-dropdown', 'options'),
     Output('y-axis-dropdown', 'options'),
     Output('color-multiselect', 'options')],
    Input('x-axis-dropdown', 'value'),
    Input('y-axis-dropdown', 'value'),
    Input('color-multiselect', 'value'),
    Input('yx-line-toggle', 'value'),
    Input('refresh-button', 'n_clicks')
)

def update_scatter_plot(x_metric, y_metric, color_params, toggle_values, _):

    # print(f"Updating scatter plot with x={x_metric}, y={y_metric}, color={color_params}, toggle={toggle_values}")
    global experiments_cache
    experiments = experiments_cache

    df = build_scatter_dataframe(x_metric, y_metric, color_params, experiments)

    fig = px.scatter(
        df,
        x='x',
        y='y',
        color='Combined Color',
        hover_name='Run ID',
        hover_data=['Mode'],
        symbol='Mode',
        symbol_map={'Unimode': 'square', 'Bimode': 'circle'},
    )
    
    customize_figure(x_metric, y_metric, experiments, fig, toggle_values)

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

def main():
    load_thread = threading.Thread(target=load_experiments_async, args=(loader, 'extremal', filter_tags))
    load_thread.daemon = True
    load_thread.start()
    app.run_server(debug=True, host='localhost', port=8050)

if __name__ == '__main__':
    main()