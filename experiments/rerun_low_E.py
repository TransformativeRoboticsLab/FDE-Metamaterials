from extremal import ex
from incense import ExperimentLoader
from utils import *

from analysis.utils import load_experiments

if __name__ == "__main__":
    loader = ExperimentLoader(
        mongo_uri=MONGO_URI,
        db_name=DB_NAME
    )
    filter_tags = ['bad']
    experiments = load_experiments(loader, 'extremal', filter_tags, process_exps=False)
    
    new_E_min = 1/10
    new_nu = 0.1

    for exp in experiments:
        config = exp.to_dict()['config']
        config['single_sim'] = True
        config['E_min'] = new_E_min
        config['nu'] = new_nu
        config['init_run_idx'] = exp.id
        config['start_beta'] = 64.
        
        print(f"Running new sim based on id {exp.id:d}")
        ex.name = 'extremal-rerun'
        ex.run(config_updates=config)
        
        