from extremal import ex
from incense import ExperimentLoader
from incense.artifact import PickleArtifact
from run_experiment import run_experiment
from sacred import Experiment

from experiments.utils import *

MONGO_URI = 'mongodb://localhost:27017'
DB_NAME = 'metatop'

def main(idxs, E_min, nu):
    loader = ExperimentLoader(
        mongo_uri=MONGO_URI,
        db_name=DB_NAME
    )

    exps = {}
    configs = {}
    for idx in idxs:   
        exp = loader.find_by_id(idx)
        exps[idx] = exp
        configs[idx] = exp.to_dict()['config']
        
    # change config to new values
    for idx, c in configs.items():
        c['E_min'] = E_min
        c['nu'] = nu
        # seed the run with the final output density from run idx
        c['init_run_idx'] = idx
        c['tighten_vector_constraint'] = False
        c['g_vec_eps'] = 1e-2
        c['n_epochs'] = 1
        c['start_beta'] = 64
        c['n_betas'] = 1
        
    for idx, c in configs.items():
        ex.run(config_updates=c)

if __name__ == '__main__':
    idxs = (509, 510, 529, 539, 553, )
    E_min = 0.1
    nu = 0.1
    main(idxs, E_min, nu)