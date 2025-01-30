# data/experiment_loader.py
import threading
import time
from io import StringIO

import numpy as np
from data.data_processing import process_experiments
from incense import ExperimentLoader
from loguru import logger

DEFAULT_EXP_NAME = 'extremal'
DEFAULT_FILTER_TAGS = ['bad']

try:
    loader = ExperimentLoader(mongo_uri='mongodb://localhost:27017', 
                            db_name='metatop')
    logger.success("Incense loader created")
except Exception as e:
    logger.exception(f"Error creating incense loader: {e}")

experiments_cache = []
dropdown_options_cache = {}
cache_lock = threading.Lock()

def load_experiments(experiment_name, filter_tags=[], process_exps=True):
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
        experiments = process_experiments(experiments) if process_exps else experiments
        logger.info(f"Loaded {len(experiments)} experiments")
        metric_dropdowns, nu_dropdowns, E_dropdowns = update_dropdown_options(experiments)
        dropdown_options = {'x-axis': metric_dropdowns, 
                             'y-axis': metric_dropdowns,
                             'nu': nu_dropdowns,
                             'E': E_dropdowns,}
        logger.success(f"Updated experiments cache at {time.ctime()}")
        return experiments, dropdown_options
    except Exception as e:
        logger.exception(f"Error loading experiments: {e}")
        return [], {}

def init_experiments_load(experiment_name=DEFAULT_EXP_NAME, 
                          filter_tags=DEFAULT_FILTER_TAGS, ):
    logger.info("Initializing experiments")
    global experiments_cache, dropdown_options_cache
    with cache_lock:
        experiments_cache, dropdown_options_cache = load_experiments(experiment_name, filter_tags)
    logger.success(f"Done initializing experiments")

def async_load_experiments(experiment_name, filter_tags=[], poll_interval=600):
    global experiments_cache, dropdown_options_cache
    while True:
        # delay up front because we do an initial sync load
        time.sleep(poll_interval)
        try:
            with cache_lock:
                experiments_cache, dropdown_options_cache = load_experiments(loader, experiment_name, filter_tags)
        except Exception as e:
            logger.exception(f"Error updating experiments cache: {e}")
        
def start_experiment_loader_thread():
    logger.info("Starting async experiment loader thread")
    load_thread = threading.Thread(target=async_load_experiments, args=('extremal', DEFAULT_FILTER_TAGS))
    load_thread.daemon = True
    load_thread.start()
    logger.success("Async loader thread started")
    
def get_cached_experiments():
    with cache_lock:
        return experiments_cache

def get_cached_dropdown_options():
    with cache_lock:
        return dropdown_options_cache

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

def get_text_from_experiment(loader, id):
    exp=loader.find_by_id(id)
    sio = StringIO()
    sio.write('C:\n')
    sio.write(np.array2string(np.matrix(exp.info.final_C), precision=3, separator=', ', suppress_small=True, max_line_width=26))
    print(sio.getvalue())
    # np.savetxt(sio, exp.info.final_C, fmt='%.3f', delimiter=', ')
    # print(sio.getvalue())   
    # return sio.getvalue()
    return sio.getvalue()

def get_image_from_experiment(id, img_type='array'):
    """
    Retrieves an image from an experiment based on the provided ID. Goes and does a new search for the experiment by ID using the loader.
    Args:
        loader (object): An Incense ExperimentLoader that provides a method `find_by_id` to retrieve experiments.
        id (str or int): The run ID of the experiment from which to retrieve the image.
        img_type: Can be 'array' or 'cell' depending on what you want to return.
    Returns:
        bytes: The image data content in 'image/png' format if found, otherwise None.
    """
    
    exp=loader.find_by_id(id)
    img = None
    for k, v in exp.artifacts.items():
        if img_type in k.lower():
            img = v.as_content_type('image/png').content
    return img

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

    # all_config_params = {k for e in experiments for k in e.config.keys()}
        
    # config_dropdowns = [{'label': p, 'value': p} for p in sorted(all_config_params)]
    metric_dropdowns = [{'label': m, 'value': m} for m in sorted(experiments[-2].metrics.keys())]
    
    # unique_nus = list(set(v for k, v in e.config.items() if 'nu' == k))
    unique_nus = list(set(e.config.nu for e in experiments))
    nu_dropdowns = [{'label': n, 'value': n} for n in sorted(unique_nus)]
    unique_Es = list(set(e.config.E_min for e in experiments))
    E_dropdowns = [{'label': n, 'value': n} for n in sorted(unique_Es)]
    
    return metric_dropdowns, nu_dropdowns, E_dropdowns