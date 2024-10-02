import base64
import io

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image


def filter_experiments_by_tag(experiments, filter_tags):
    """
    Filters out experiments that contain any of the specified omniboard tags.
    Args:
        experiments (list): A list of experiment objects. Each experiment object may have an 'omniboard' attribute 
                            which contains a 'tags' attribute (a list of tags).
        filter_tags (list): A list of tags to filter out. If an experiment contains any of these tags (case-insensitive), 
                            it will be excluded from the returned list.
    Returns:
        list: A list of experiments that do not contain any of the specified tags.
    """
    
    get_tags = lambda e: getattr(e.omniboard, 'tags', []) if hasattr(e, 'omniboard') else []
    experiments = [e for e in experiments if not any(t.lower() in filter_tags for t in get_tags(e))]

    return experiments

def load_experiments(loader, experiment_name, filter_tags=[]):
    """
    Load and filter experiments based on the given experiment name and tags.
    This function retrieves experiments from the loader, filters them by their
    status and omniboard tags, and reorganizes their artifacts.
    Args:
        loader (object): An object that provides a method `find` to retrieve experiments.
        experiment_name (str): The name of the experiment to load.
        filter_tags (list, optional): A list of tags to filter the experiments. Defaults to an empty list.
    Returns:
        list: A list of filtered and processed experiments. If an error occurs, an empty list is returned.
    """
    
    try:
        experiments = loader.find({'experiment.name': experiment_name})
        experiments = [e for e in experiments if e.status == 'COMPLETED']
        experiments = filter_experiments_by_tag(experiments, filter_tags)
        return process_experiments(experiments)
    except Exception as e:
        print(f"Error loading experiments: {e}")
        return []

def process_experiments(experiments):
    
    for e in experiments:
        reorganize_artifacts(e.artifacts)
        if 'volume_fraction' not in e.metrics:
            e.metrics['volume_fraction'] = pd.Series([compute_volume_fraction(e)])
            
    return experiments

def compute_volume_fraction(e):
    img_byte = e.artifacts['cells'][-1].as_content_type('image/png').content
    img = Image.open(io.BytesIO(img_byte)).convert('1')
    return 1. - np.mean(img)

def reorganize_artifacts(artifacts):
    """
    Reorganizes a dictionary of artifacts into specific categories.
    The function processes the input dictionary `artifacts` and categorizes 
    its items into 'timelines', 'cells', 'array', 'pickle', and 'misc'. 
    The categorized items are then stored in a new dictionary which updates 
    the original `artifacts` dictionary.
    
    Why? This makes referencing artifacts that are similar in nature easier.

    Args:
        artifacts (dict): A dictionary where keys are artifact names and 
                          values are artifact objects.
    Modifies:
        artifacts (dict): The original dictionary is updated with categorized 
                          artifacts and the original keys are removed.
    Categories:
        - 'timelines': List of artifacts containing 'timeline' in their key.
        - 'cells': List of artifacts containing 'cell' in their key.
        - 'array': Single artifact containing 'array' in its key.
        - 'pickle': Single artifact with a key ending in '.pkl'.
        - 'misc': List of all other artifacts.
    """

    new_artifacts = {
        'timelines': [],
        'cells': [],
        'array': None,
        'pickle': None,
        'misc': []
    }
    
    for key, artifact in list(artifacts.items()):
        key_lower = key.lower()
        
        if 'timeline' in key_lower:
            new_artifacts['timelines'].append(artifact.as_content_type('image/png'))
        elif 'cell' in key_lower:
            new_artifacts['cells'].append(artifact.as_content_type('image/png'))
        elif 'array' in key_lower:
            new_artifacts['array'] = artifact.as_content_type('image/png')
        elif key.endswith('.pkl'):
            new_artifacts['pickle'] = artifact
        else:
            new_artifacts['misc'].append(artifact)
        
        del artifacts[key]
    
    artifacts.update(new_artifacts)

def update_dropdown_options(experiments):
    """
    Generates dropdown options for configuration parameters and metrics from a list of experiments.
    Args:
        experiments (list): A list of Incense Experiment objects. Each experiment object should have 
                            'config' and 'metrics' attributes, where 'config' is a dictionary 
                            of configuration parameters and 'metrics' is a dictionary of metrics.
    Returns:
        tuple: A tuple containing two lists:
            - metric_dropdowns (list): A list of dictionaries with 'label' and 'value' keys for each metric.
            - config_dropdowns (list): A list of dictionaries with 'label' and 'value' keys for each configuration parameter.
    """

    all_config_params = {k for e in experiments for k in e.config.keys()}
        
    config_dropdowns = [{'label': p, 'value': p} for p in sorted(all_config_params)]
    metric_dropdowns = [{'label': m, 'value': m} for m in sorted(experiments[-2].metrics.keys())]
    
    return metric_dropdowns, config_dropdowns

def get_fields(experiments, field):
    
    return sorted({e.config.get(field, 'None') for e in experiments})

def get_image_from_experiment(loader, id):
    """
    Retrieves an image from an experiment based on the provided ID. Goes and does a new search for the experiment by ID using the loader.
    Args:
        loader (object): An Incense ExperimentLoader that provides a method `find_by_id` to retrieve experiments.
        id (str or int): The run ID of the experiment from which to retrieve the image.
    Returns:
        bytes: The image data content in 'image/png' format if found, otherwise None.
    """
    
    exp=loader.find_by_id(id)
    img = None
    for k, v in exp.artifacts.items():
        if 'array' in k.lower():
            img = v.as_content_type('image/png').content
    return img

def encode_image(data):
    """
    Encodes binary image data to a base64 ASCII string.
    Args:
        data (bytes): The binary image data to encode.
    Returns:
        str: The base64 encoded ASCII string representation of the image data.
    """
    
    return base64.b64encode(data).decode('ascii')

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
                }
            ]
        )

def customize_figure(x_metric, y_metric, experiments, fig, plot_yx_line=[], size=(1000, 1000)):
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
    
    plot_yx(fig) if 'plot_yx' in plot_yx_line else None
    fig.update_layout(title=f"{len(experiments):d} Data Points", width=size[0], height=size[1])

    fig.update_xaxes(title_text=x_metric)
    fig.update_yaxes(scaleanchor='x', scaleratio=1, title_text=y_metric)
    # reapply_current_zoom(relayout_data, fig)

    fig.update_traces(marker=dict(size=12), mode='markers')

def build_scatter_dataframe(x_metric, y_metric, experiments):
    # Create a list of dictionaries (records) to build DataFrame efficiently in one step
    records = []
    
    mode_map = {1: 'Unimode', 2: 'Bimode'}
    
    for e in experiments:
        if x_metric not in e.metrics or y_metric not in e.metrics:
            continue

        x_value = e.metrics.get(x_metric, None).iloc[-1]
        y_value = e.metrics.get(y_metric, None).iloc[-1]
        
        # combined_color = '_'.join([str(e.config.get(param, 'None')) for param in config_filter_values])
        
        # mode = mode_map.get(e.config.get('extremal_mode', None), 'Unknown')
        
        records.append({
            'x': x_value,
            'y': y_value,
            # 'Combined Color': combined_color,
            # 'Combined Symbol': combined_color,
            'Run ID': f'Run ID: {e.id}',
            # 'extremal_mode': e.config.get('extremal_mode', -1),
            # 'basis_v': e.config.get('basis_v', 'None'),
            'extremal_mode': e.config.get('extremal_mode', 'None'),
            'basis_v': e.config.get('basis_v', 'None'),
            # color_filter_value: e.config.get(color_filter_value, 'None'),
            # marker_filter_value: e.config.get(marker_filter_value, 'None'),
        })

    df = pd.DataFrame.from_records(records)
    
    df.sort_values(by='extremal_mode', ascending=True, inplace=True)
    
    return df

PLOT_SYMBOLS = ['circle', 'square', 'diamond', 'cross', 'x', 'triangle-up', 'star', 'hexagram', 'pentagon', 'hourglass']

def build_symbol_map(experiments, marker_filter_value):
    unique_marker_values = {e.config.get(marker_filter_value) for e in experiments if e.config.get(marker_filter_value) is not None}
    return {marker_value: PLOT_SYMBOLS[i % len(PLOT_SYMBOLS)] for i, marker_value in enumerate(unique_marker_values)}