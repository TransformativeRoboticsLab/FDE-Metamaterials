import numpy as np
import plotly.express as px
from loguru import logger


def build_scatter_figure(df, x_metric, y_metric, plot_yx_line=False, size=(750, 750)):
    logger.info("Building scatter figure")
    fig = px.scatter(
        df,
        x=x_metric,
        y=y_metric,
        hover_name='Run ID',
        symbol='extremal_mode',
        color='basis_v',
    )
    
    # if plot_yx_line:
    #     fig.add_shape(
    #         type='line',
    #         x0=df['x'].min(),
    #         y0=df['y'].min(),
    #         x1=df['x'].max(),
    #         y1=df['y'].max(),
    #         line=dict(color="Black", dash='dash')
    #     )
        
    fig = customize_figure(fig, x_metric, y_metric, len(df), size=size)
    
    logger.info("Done building scatter figure")
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
        fig.update_xaxes(range=[relayout_data['xaxis.range[0]'], relayout_data['xaxis.range[1]']])
    if relayout_data and 'yaxis.range[0]' in relayout_data and 'yaxis.range[1]' in relayout_data:
        fig.update_yaxes(range=[relayout_data['yaxis.range[0]'], relayout_data['yaxis.range[1]']])

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
    p2 = f'M {x2[0]},{y2[0]} ' + ' '.join(f'L {x},{y}' for x, y in zip(x2[1:], y2[1:]))
    
    theta = np.linspace(np.pi/4, np.pi/2, 100)
    x3 = np.cos(theta)
    y3 = np.sin(theta)
    p3 = f'M {x3[0]},{y3[0]} ' + ' '.join(f'L {x},{y}' for x, y in zip(x3[1:], y3[1:]))

    x4 = np.linspace(0, 0.5, 100)
    y4 = x4 / (1 - x4)
    p4 = f'M {x4[0]},{y4[0]} ' + ' '.join(f'L {x},{y}' for x, y in zip(x4[1:], y4[1:]))

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
                } ,           {
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

def customize_figure(fig, x_metric, y_metric, num_exps, plot_yx_line=[], size=(1000, 1000)):
    """
    Customizes a given figure with specified metrics, layout, and toggle options.
    Parameters:
    x_metric (str): The label for the x-axis.
    y_metric (str): The label for the y-axis.
    relayout_data (dict): Data used to reapply the current zoom level.
    experiments (list): A list of experiments to be plotted.
    fig (plotly.graph_objs._figure.Figure): The figure object to be customized.
    toggle_values (list, optional): A list of toggle options to apply specific customizations. Defaults to an empty list.
    Returns:
    None
    """
    print(f"x: {x_metric}, y: {y_metric}")
    plot_yx(fig) if 'plot_yx' in plot_yx_line else None
    fig.update_layout(title=f"{num_exps:d} Data Points", width=size[0], height=size[1])

    fig.update_xaxes(title_text=x_metric)
    fig.update_yaxes(scaleanchor='x', scaleratio=1, title_text=y_metric)
    # reapply_current_zoom(relayout_data, fig)

    fig.update_traces(marker=dict(size=12), mode='markers')
    
    return fig
    
    