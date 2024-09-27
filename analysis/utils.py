import base64


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
        for e in experiments:
            reorganize_artifacts(e.artifacts)
        return experiments
    except Exception as e:
        print(f"Error loading experiments: {e}")
        return []

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
    
    if experiments:
        metric_names = sorted(list(set(experiments[-2].metrics.keys())))
        config_names = sorted(list(set(experiments[-2].config.keys())))
        metric_dropdowns = [{'label': name, 'value': name} for name in metric_names]
        config_dropdowns = [{'label': name, 'value': name} for name in config_names]
        
        return metric_dropdowns, config_dropdowns
    return [], [], []

def get_image_from_experiment(loader, id):
    exp=loader.find_by_id(id)
    img = None
    for k, v in exp.artifacts.items():
        if 'array' in k.lower():
            img = v.as_content_type('image/png').content
    return img

def encode_image(data):
    return base64.b64encode(data).decode('ascii')