import io

import numpy as np
import pandas as pd
from loguru import logger
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
            new_artifacts['timelines'].append(
                artifact.as_content_type('image/png'))
        elif 'cell' in key_lower:
            new_artifacts['cells'].append(
                artifact.as_content_type('image/png'))
        elif 'array' in key_lower:
            new_artifacts['array'] = artifact.as_content_type('image/png')
        elif key.endswith('.pkl'):
            new_artifacts['pickle'] = artifact
        else:
            new_artifacts['misc'].append(artifact)

        del artifacts[key]

    artifacts.update(new_artifacts)


def process_experiments(experiments):
    logger.info(f"Processing experiments!")
    for e in experiments:
        reorganize_artifacts(e.artifacts)
        # if 'volume_fraction' not in e.metrics:
        # e.metrics['volume_fraction'] = pd.Series([compute_volume_fraction(e)])
    logger.info(f"Processed {len(experiments)} experiments")
    return experiments


def prepare_scatter_data(x_metric, y_metric, exps, filters={}):
    logger.info("Preparing scatter plot data")
    # Create a list of dictionaries (records) to build DataFrame efficiently in one step

    # helpers
    mode_map = {1: 'Unimode', 2: 'Bimode', }
    def last(x): return x.iloc[-1]

    df = exps.project(on=[{f"metrics.{x_metric}": last},
                          {f"metrics.{y_metric}": last},
                          "id",
                          "config.extremal_mode",
                          "config.basis_v",
                          "config.nu",
                          "config.E_min"
                          ])
    # the projection to dataframe concatenates the metric with the function handle of last
    # in this case last is a lambda so it looks like {x_metric}_<lambda>
    # this remove _<lambda> from the metric column names
    df.rename(columns={c: c.split('_<')[0] for c in df.columns},
              inplace=True)
    # Prettify other columns
    df['id'] = df['id'].apply(lambda id: f"Run ID: {id}")
    df.rename(columns={'id': 'Run ID'}, inplace=True)

    df['extremal_mode'] = df['extremal_mode'].map(mode_map)

    df.sort_values(by='extremal_mode', ascending=False, inplace=True)

    # apply filters
    for name, filter in filters.items():
        # skip if filter is empty
        df = df[df[name].isin(filter)] if filter else df

    logger.info("Done preparing data")
    return df


PLOT_SYMBOLS = ['circle', 'square', 'diamond', 'cross', 'x',
                'triangle-up', 'star', 'hexagram', 'pentagon', 'hourglass']


def build_symbol_map(experiments, marker_filter_value):
    unique_marker_values = {e.config.get(
        marker_filter_value) for e in experiments if e.config.get(marker_filter_value) is not None}
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
