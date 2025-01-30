# data/experiment_loader.py
import threading
import time
from io import StringIO

import numpy as np
from data.data_processing import process_experiments
from incense import ExperimentLoader

loader = ExperimentLoader(mongo_uri='mongodb://localhost:27017', 
                            db_name='metatop')
experiments_cache = []
cache_lock = threading.Lock()

def load_experiments(loader, experiment_name, filter_tags=[], process_exps=True):
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
        print(f"Loading experiments")
        experiments = loader.find({'experiment.name': experiment_name})
        experiments = [e for e in experiments if e.status == 'COMPLETED']
        experiments = filter_experiments_by_tag(experiments, filter_tags)
        # a very, very bad manual coding to exclude all the sims that were run before we realized the foam PR is actually much closer to zero than anything else
        # experiments = [e for e in experiments if e.config.nu < 1e-3]
        experiments = process_experiments(experiments) if process_exps else experiments
        print(f"Loaded {len(experiments)} experiments")
        return experiments
    except Exception as e:
        print(f"Error loading experiments: {e}")
        return []

def load_experiments_async(loader, experiment_name, filter_tags=[], poll_interval=60):
    global experiments_cache
    while True:
        try:
            with cache_lock:
                experiments_cache = load_experiments(loader, experiment_name, filter_tags)
                print(f"Updated experiments cache at {time.ctime()}")
        except Exception as e:
            print(f"Error updating experiments cache: {e}")
        time.sleep(poll_interval)
        
def start_experiment_loader_thread():
    filter_tags = ['bad']
    load_thread = threading.Thread(target=load_experiments_async, args=(loader, 'extremal', filter_tags))
    load_thread.daemon = True
    load_thread.start()
    
def get_cached_experiments():
    with cache_lock:
        return experiments_cache

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