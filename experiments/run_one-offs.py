from extremal import ex
from incense import ExperimentLoader
from run_experiment import run_experiment
from sacred import Experiment

from experiments.utils import *

MONGO_URI = 'mongodb://localhost:27017'
DB_NAME = 'metatop'

def main(idxs):
    new_E_min = 1/10
    new_nu = 0.1

    loader = ExperimentLoader(
        mongo_uri=MONGO_URI,
        db_name=DB_NAME
    )

    configs = {}
    for idx in idxs:   
        exp = loader.find_by_id(idx)
        configs[idx] = exp.to_dict()['config']
        
    # change config to new values
    # for idx, c in configs.items():
    #     c['E_min'] = new_E_min
    #     c['nu'] = new_nu
        
    for idx, c in configs.items():
        ex.run(config_updates=c)

if __name__ == '__main__':
    idxs = (509, 510, 529, 539, 553, )

    main(idxs)