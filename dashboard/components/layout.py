import dash_bootstrap_components as dbc
from dash import dcc, html
from loguru import logger

from data.experiment_loader import get_cached_dropdown_options


def create_layout():
    logger.info("LAYOUT: Creating")
    doc = get_cached_dropdown_options()

    # Combine Sidebar and Main Content in a Single Row
    layout = dbc.Container(
        dbc.Row([sidebar(doc),
                 main_content()]),
        fluid=True
    )
    logger.info("LAYOUT: Finished")
    return layout


def main_content():
    main_content = dbc.Col(
        [
            # First row: Scatter plot (left) and Image (right)
            dbc.Row(
                [
                    dbc.Col(
                        dcc.Graph(
                            id='scatter-plot',
                            config={'displayModeBar': False},
                            style={
                                'height': '400px',
                                'width': '100%'
                            }
                        ),
                        width=6
                    ),
                    dbc.Col(
                        html.Div(
                            html.Img(
                                id='hover-image',
                                src='assets/placeholder.jpg',
                                style={
                                    'max-width': '100%',
                                    'max-height': '400px',
                                    'display': 'block',
                                    'margin': 'auto'
                                }
                            ),
                            style={
                                'height': '400px',
                                'display': 'flex',
                                'align-items': 'center',
                                'justify-content': 'center'
                            }
                        ),
                        width=6
                    ),
                ],
                className="mb-4"  # Add margin at the bottom
            ),
            # Second row: EG polar plot (left) and Nu polar plot (right)
            dbc.Row(
                [
                    dbc.Col(
                        dcc.Graph(
                            id='EG-polar-plot',
                            config={'displayModeBar': False},
                            style={
                                'height': '400px',
                                'width': '100%'
                            }
                        ),
                        width=6
                    ),
                    dbc.Col(
                        dcc.Graph(
                            id='Nu-polar-plot',
                            config={'displayModeBar': False},
                            style={
                                'height': '400px',
                                'width': '100%'
                            }
                        ),
                        width=6
                    )
                ]
            )
        ],
        width=9,
        style={'border': '1px solid #ddd', 'margin': '15px'}
    )

    return main_content


def sidebar(doc):
    sidebar = dbc.Col(
        [
            welcome_div(),
            metrics_div(doc),
            filters_div(doc),
        ],
        width=2,
        style={'border': '1px solid #ddd', 'margin': '15px'}
    )

    return sidebar


def filters_div(doc):
    input_div = html.Div(
        [
            html.H2("Input Parameter Filters"),
            html.Label('Matrix Distance Method'),
            dcc.Dropdown(
                id='dist-filter',
                options=['fro', 'log_euc', 'affine', 'sqrt', 'airm'],
                value=[],
                multi=True,
                clearable=True
            ),
            html.Label('Basis'),
            dcc.Dropdown(
                id='basis-filter',
                options=['BULK', 'VERT', 'SHEAR', 'HSA'],
                value=[],
                multi=True,
                clearable=True
            ),
            html.Label('Mode'),
            dcc.Dropdown(
                id='mode-filter',
                options=['Unimode', 'Bimode'],
                value=[],
                multi=True,
                clearable=True
            ),
            html.Label("Poisson's Ratio"),
            dcc.Dropdown(
                id='nu-filter',
                options=doc['nu'],
                value=[],
                multi=True,
                clearable=True
            ),
            html.Label("Young's Modulus Ratio"),
            dcc.Dropdown(
                id='E-filter',
                options=doc['E'],
                value=[],
                multi=True,
                clearable=True
            ),
            # dcc.Checklist(
            #     options=[
            #         {'label': 'Show only reruns', 'value': 'rerun'},
            #     ],
            #     value=[],
            #     id='rerun-filter',
            #     inline=True,
            # ),

            html.Button('Clear All Filters', id='clear-button',
                        style={'margin-top': '10px'})
        ],
        className='filters-container',
        style={'padding': '10px', 'border-top': '1px solid #ccc'}
    )

    return input_div


def metrics_div(doc):
    metrics_div = html.Div(
        [
            html.H2("Metrics"),
            html.Div(
                [
                    html.Label('X-axis'),
                    dcc.Dropdown(
                        id='x-axis-dropdown',
                        options=doc['x-axis'],
                        value='E2',
                        clearable=False,
                        # optionHeight can be set if you have many options
                    )
                ],
                className='dropdown-container',
                style={'margin-bottom': '10px'}
            ),
            html.Div(
                [
                    html.Label('Y-axis'),
                    dcc.Dropdown(
                        id='y-axis-dropdown',
                        options=doc['y-axis'],
                        value='nu21',
                        clearable=False
                    )
                ],
                className='dropdown-container'
            ),
        ],
        className='metrics-container',
        style={'padding': '10px', 'border-top': '1px solid #ccc'}
    )

    return metrics_div


def welcome_div():
    welcome_div = html.Div(
        [
            html.H1("Welcome"),
            html.P(
                "This is a dashboard for analyzing the results of the Meta-Top experiments.")
        ],
        className='sidebar-header',
        style={'padding': '10px'}
    )

    return welcome_div
