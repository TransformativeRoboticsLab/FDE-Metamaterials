import re

import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from loguru import logger
from utils.utils import log_execution_time

from data import get_cached_experiments


# Extract metric types and check if they are the same
def get_metric_type(metric):
    """Extract the type of metric (e.g., 'E' from 'E1', 'nu' from 'nu12')."""
    match = re.match(r'^([a-zA-Z]+)', str(metric))
    return match.group(1) if match else str(metric)


def metrics_match(m1, m2):
    # If metrics are exactly the same or of the same type (E1 vs E2)
    return m1 == m2 or get_metric_type(m1) == get_metric_type(m2)


def negative_ok(metric_name):
    negative_metrics = ['nu', 'eta']
    return any(n in metric_name for n in negative_metrics)


@log_execution_time()
def build_scatter_figure(df, x_metric, y_metric, size=(800, 800)):
    logger.info("Building scatter figure")
    logger.debug(f"x_metric: {x_metric}")
    logger.debug(f"y_metric: {y_metric}")

    fig = px.scatter(
        df,
        x=x_metric,
        y=y_metric,
        hover_name='Run ID',
        symbol='extremal_mode',
        color='basis_v',
    )

    # # Calculate min and max for x and y
    # x_min = df[x_metric].min() if not df.empty else 0
    # x_max = df[x_metric].max() if not df.empty else 1
    # y_min = df[y_metric].min() if not df.empty else 0
    # y_max = df[y_metric].max() if not df.empty else 1

    # # Set axis ranges based on data and whether negative values are allowed
    # if negative_ok(x_metric):
    #     x_range = [x_min - 0.1, x_max + 0.1]
    # else:
    #     x_range = [0, x_max + 0.1]

    # if negative_ok(y_metric):
    #     y_range = [y_min - 0.1, y_max + 0.1]
    # else:
    #     y_range = [0, y_max + 0.1]

    fig.update_layout(
        transition=dict(
            duration=500,
            easing='cubic-in-out',
            ordering='traces first'
        ),
        title=f"{len(df)} Data Points",
        clickmode='event+select',
        # xaxis=dict(range=x_range),
        # yaxis=dict(range=y_range)
    )

    if metrics_match(x_metric, y_metric):
        fig.update_yaxes(scaleanchor='x', scaleratio=1)

        # Add y=x line without affecting axis limits
        fig.add_shape(
            type="line",
            x0=0, y0=0, x1=1, y1=1,
            line=dict(
                color="Black",
                width=2,
                dash="dash",
            ),
            opacity=0.5,
            layer="below",
        )

    fig.update_traces(
        marker=dict(
            opacity=0.5,
            size=12,
            line=dict(width=1, color='DarkSlateGrey'),
        ),
        selected=dict(
            marker=dict(
                size=18,
                opacity=1.0,
            )
        ),
        unselected=dict(
            marker=dict(opacity=0.2)
        ),
    )

    logger.info("Done building scatter figure")
    return fig


@log_execution_time()
def build_polar_figure(df, theta_col, r_metrics, colors=[], size=(800, 800), r_limits=None, title=None):
    """
    Build a polar plot figure with transition effects.

    Parameters:
    df (pandas.DataFrame): The dataframe containing the data to plot.
    r_metric (str): The name of the column to use for the radial dimension.
    theta_metric (str): The name of the column to use for the angular dimension.
    size (tuple, optional): The size of the figure (width, height). Defaults to (800, 800).
    r_limits (tuple, optional): The range limits for the radial axis (min, max). Defaults to None.

    Returns:
    plotly.graph_objs._figure.Figure: The polar plot figure.
    """
    logger.info("Building polar figure")

    if isinstance(r_metrics, str):
        r_metrics = [r_metrics]

    if not colors:
        colors = px.colors.qualitative.D3

    logger.debug(f"theta_col: {theta_col}")
    logger.debug(f"r_metrics: {r_metrics}")
    logger.debug(f"r_limits: {r_limits}")
    logger.debug(f"colors: {colors}")

    # Create base figure with first metric
    fig = px.line_polar(
        df,
        r=r_metrics[0],
        theta=theta_col,
        color_discrete_sequence=[colors[0]],
    )

    fig.data[0].name = r_metrics[0]
    fig.data[0].showlegend = True

    # Add additional metrics if there are any
    for i, metric in enumerate(r_metrics[1:], 1):
        fig.add_scatterpolar(
            r=df[metric],
            theta=df[theta_col],
            mode='lines',
            name=metric,
            line=dict(color=colors[i % len(colors)])
        )

    # Create unified polar configuration so that it is aligned how we want
    polar_config = {
        "angularaxis": {
            "rotation": 0,  # Rotate by 90 degrees
            "direction": "counterclockwise",
            "thetaunit": "radians"
        }
    }

    if r_limits:
        polar_config["radialaxis"] = {"range": r_limits}

    fig.update_layout(polar=polar_config)

    if title:
        fig.update_layout(title=title)

    logger.info("Done building polar figure")
    return fig


def reapply_current_zoom(relayout_data, fig):
    """
    Reapply the current zoom level to a Plotly figure based on the provided relayout data.
    Parameters:
    relayout_data (dict): A dictionary containing the current zoom level information. 
                          Expected keys are 'xaxis.range[0]', 'xaxis.range[1]', 
                          'yaxis.range[0]', and 'yaxis.range[1]'.
    fig (plotly.graph_objs._figure.Figure): The Plotly figure object to update.
    Returns:
    None: This function updates the figure in place and does not return any value.
    """

    if relayout_data and 'xaxis.range[0]' in relayout_data and 'xaxis.range[1]' in relayout_data:
        fig.update_xaxes(
            range=[relayout_data['xaxis.range[0]'], relayout_data['xaxis.range[1]']])
    if relayout_data and 'yaxis.range[0]' in relayout_data and 'yaxis.range[1]' in relayout_data:
        fig.update_yaxes(
            range=[relayout_data['yaxis.range[0]'], relayout_data['yaxis.range[1]']])


def plot_yx(fig, xs=(0., 1.), ys=(0., 1.)):
    """
    Adds a dashed y=x line to the given figure.
    Parameters:
    fig (plotly.graph_objs._figure.Figure): The figure to which the line will be added.
    xs (tuple, optional): A tuple containing the start and end x-coordinates of the line. Defaults to (0., 1.).
    ys (tuple, optional): A tuple containing the start and end y-coordinates of the line. Defaults to (0., 1.).
    Returns:
    None
    """

    x0, x1 = xs
    y0, y1 = ys

    x2 = np.linspace(0, 1/np.sqrt(2), 100)
    y2 = x2 / np.sqrt(1. - x2**2)
    p2 = f'M {x2[0]},{y2[0]} ' + \
        ' '.join(f'L {x},{y}' for x, y in zip(x2[1:], y2[1:]))

    theta = np.linspace(np.pi/4, np.pi/2, 100)
    x3 = np.cos(theta)
    y3 = np.sin(theta)
    p3 = f'M {x3[0]},{y3[0]} ' + \
        ' '.join(f'L {x},{y}' for x, y in zip(x3[1:], y3[1:]))

    x4 = np.linspace(0, 0.5, 100)
    y4 = x4 / (1 - x4)
    p4 = f'M {x4[0]},{y4[0]} ' + \
        ' '.join(f'L {x},{y}' for x, y in zip(x4[1:], y4[1:]))

    fig.update_layout(
        shapes=[
            {
                'type': 'line',
                'x0': x0,
                'y0': y0,
                'x1': x1,
                'y1': y1,
                'line': {
                        'color': 'Black',
                        'width': 2,
                        'dash': 'dash'
                },
            },           {
                'type': 'line',
                'x0': 0.,
                'y0': 1.,
                'x1': 0.5,
                'y1': 0.5,
                'line': {
                        'color': 'Black',
                        'width': 2,
                        'dash': 'dash'
                },
            },
            {
                'type': 'path',
                'path': p3,
                'line': {
                        'color': 'Black',
                        'width': 2,
                        'dash': 'dash'
                },
            },
            {
                'type': 'path',
                'path': p2,
                'line': {
                        'color': 'Black',
                        'width': 2,
                        'dash': 'dash'
                }
            },
            {
                'type': 'path',
                'path': p4,
                'line': {
                        'color': 'Black',
                        'width': 2,
                        'dash': 'dash'
                }
            }
        ]
    )
