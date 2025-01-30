import io

import numpy as np
import pandas as pd
from PIL import Image
from utils.mechanics import generate_planar_values, isotropic_elasticity_matrix


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

def process_experiments(experiments):
    print(f"Processing experiments!")
    for e in experiments:
        reorganize_artifacts(e.artifacts)
        # if 'volume_fraction' not in e.metrics:
            # e.metrics['volume_fraction'] = pd.Series([compute_volume_fraction(e)])
    print(f"Processed {len(experiments)} experiments")
    return experiments


def prepare_scatter_data(x_metric, y_metric, experiments, nu_filter=[], E_filter=[]):
    print("Preparing scatter plot data")
    # Create a list of dictionaries (records) to build DataFrame efficiently in one step
    records = []
    
    # mode_map = {1: 'Unimode', 2: 'Bimode'}
    
    for e in experiments:
        if x_metric not in e.metrics or y_metric not in e.metrics:
            continue

        x_value = e.metrics[x_metric].iloc[-1]
        y_value = e.metrics[y_metric].iloc[-1]
        
        records.append({
            'x': x_value,
            'y': y_value,
            'Run ID': f'Run ID: {e.id}',
            'extremal_mode': e.config.get('extremal_mode', 'None'),
            'basis_v': e.config.get('basis_v', 'None'),
            'marker': e.config.get('extremal_mode', 'None'),
            'color': e.config.get('basis_v', 'None'),
            'nu': e.config.nu,
            'E_max': e.config.E_max,
            'E_min': e.config.E_min,
        })

    df = pd.DataFrame.from_records(records)
    
    df.sort_values(by='extremal_mode', ascending=True, inplace=True)
    if nu_filter:
        df = df[df['nu'].isin(nu_filter)]
    if E_filter:
        df = df[df['E_min'].isin(E_filter)]
    
    print("Done preparing data")
    return df

PLOT_SYMBOLS = ['circle', 'square', 'diamond', 'cross', 'x', 'triangle-up', 'star', 'hexagram', 'pentagon', 'hourglass']

def build_symbol_map(experiments, marker_filter_value):
    unique_marker_values = {e.config.get(marker_filter_value) for e in experiments if e.config.get(marker_filter_value) is not None}
    return {marker_value: PLOT_SYMBOLS[i % len(PLOT_SYMBOLS)] for i, marker_value in enumerate(unique_marker_values)}

def flatten_C(C):
    return np.array([C[0, 0],
                     C[1, 1],
                     C[2, 2],
                     C[1, 2],
                     C[0, 2],
                     C[0, 1]])
    
def get_fields(experiments, field):
    
    return sorted({e.config.get(field, 'None') for e in experiments})