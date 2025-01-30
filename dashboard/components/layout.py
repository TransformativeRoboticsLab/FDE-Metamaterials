# components/layout.py
import dash_bootstrap_components as dbc
from dash import dcc, html


def create_layout():
    return dbc.Container([
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
                                value='E1',
                                optionHeight=40,
                                clearable=False),
                    ]),
                # Y-axis Metric
                html.Div([
                    html.Label('Y-axis'),
                    dcc.Dropdown(id='y-axis-dropdown',
                                options=[],
                                value='nu21',
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
            # Filters
            html.Div([
                html.H2("Filters"),
                html.Div([
                    html.Label('Basis'),
                    dcc.Dropdown(id='basis-filter',
                                options=['BULK', 'VERT', 'SHEAR', 'HSA'],
                                value=[],
                                multi=True,
                                clearable=True),
                ]),
                html.Div([
                    html.Label('Mode'),
                    dcc.Dropdown(id='mode-filter',
                                options=['Unimode', 'Bimode'],
                                value=[],
                                multi=True,
                                clearable=True),
                ]),
                html.Div([
                    html.Label("Poisson's Ratio"),
                    dcc.Dropdown(id='nu-filter',
                                options=[],
                                value=[],
                                multi=True,
                                clearable=True)
                    ]
                ),
                html.Div([
                    html.Label("Young's Modulus"),
                    dcc.Dropdown(id='E-filter',
                                 options=[],
                                 value=[],
                                 multi=True,
                                 clearable=True)
                ])
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