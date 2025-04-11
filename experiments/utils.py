import os
import pickle
import random
import sys
from os import getenv as env
from pathlib import Path

import dotenv
import nlopt
import numpy as np
from incense import ExperimentLoader
from incense.artifact import PickleArtifact as PA
from loguru import logger
from PIL import Image
from pymongo.mongo_client import MongoClient
from pymongo.server_api import ServerApi
from sacred.observers import FileStorageObserver, MongoObserver

from metatop import V_DICT
from metatop.image import bitmapify
from metatop.mechanics import (anisotropy_index, calculate_elastic_constants,
                               mandelize, matrix_invariants)
from metatop.optimization import OptimizationState
from metatop.optimization.OptimizationComponents import OptimizationComponent

dotenv.load_dotenv()
MONGO_URI = env('LOCAL_MONGO_URI')
MONGO_DB_NAME = env('LOCAL_MONGO_DB_NAME')
MONGO_EXP_NAME = env('LOCAL_MONGO_EXP_NAME')

PARAMETER_SPECS = {
    'E_max': {
        'type': 'float',
        'range': (1., 1.),
    },
    'E_min': {
        'type': 'float',
        'values': tuple(10**(-n) for n in range(10)),
    },
    'nu': {
        'type': 'float',
        'range': (0.1, 0.499),
    },
    'start_beta': {
        'type': 'int',
        'range': (1, 8),
    },
    'n_betas': {
        'type': 'int',
        'range': (1, 8),
    },
    'n_epochs': {
        'type': 'int',
        'range': (1, 8),
    },
    'epoch_duration': {
        'type': 'int',
        'range': (1, 10_000),
    },
    'extremal_mode': {
        'type': 'int',
        'values': (1, 2),
    },
    'basis_v': {
        'type': 'cat',
        'values': ['BULK', 'PSHEAR', 'SSHEAR', 'VERT', 'NSC', 'NSC3']
    },
    'objective_type': {
        'type': 'cat',
        'values': ('ray', 'ray_sq', 'ratio')
    },
    'dist_type': {
        'type': 'cat',
        'values': ['fro',]  # 'log_euc', 'airm']
        # the log euclidean and AIRM metrics are supposed to be better for SPD matrices, but I haven't seen any appreciable effect so I'll stick with Frobenius for now
    },
    'nelx': {
        'type': 'int',
        'range': (10, 1_000),
    },
    'nely': {
        'type': 'int',
        'range': (10, 1_000),
    },
    'norm_filter_radius': {
        'type': 'float',
        'range': (0.01, 0.5),
    },
    'g_vec_eps': {
        'type': 'float',
        'values': tuple(10**(-n) for n in range(3)),
    },
    'weight_scaling_factor': {
        'type': 'float',
        'values': tuple(10**(-n) for n in range(4)),
    },
    'mirror_axis': {
        'type': 'cat',
        'values': ['x', 'y', 'xy', 'xyd', None]
    }
}


def get_random_value(param):
    spec = PARAMETER_SPECS.get(param, None)
    if spec is None:
        raise ValueError(f"Parameter '{param}' not found in PARAMETER_SPECS")

    s_type = spec.get('type', None)
    if s_type == 'cat':
        return random.choice(spec['values'])
    elif s_type == 'float':
        if 'range' in spec:
            return random.uniform(*spec['range'])
        elif 'values' in spec:
            return random.choice(spec['values'])
    elif s_type == 'int':
        if 'range' in spec:
            return random.randint(*spec['range'])
        elif 'values' in spec:
            return random.choice(spec['values'])


def validate_param_value(param, value, verbose=True):
    logger.info(f"Validating parameter '{param}' with value: {value}")
    spec = PARAMETER_SPECS.get(param, None)
    if spec is None:
        logger.info(f"Parameter '{param}' not found in PARAMETER_SPECS")
        return False

    s_type = spec.get('type', None)
    if s_type == 'cat':
        if value not in spec['values']:
            raise ValueError(
                f"Invalid categorical value for parameter {param}: {value}. Values must be one of {spec['values']}")
    elif s_type == 'float':
        try:
            value = float(value)
        except ValueError:
            raise ValueError(
                f"Invalid float value for parameter '{param}': {value}")
        if 'values' in spec and value not in spec['values']:
            raise ValueError(
                f"Invalid float value for parameter '{param}': {value}. Values must be one of {spec['values']}")
        elif 'range' in spec and not spec['range'][0] <= value <= spec['range'][1]:
            raise ValueError(
                f"Invalid float value for parameter '{param}': {value}. Must be between {spec['range'][0]} and {spec['range'][1]}")
    elif s_type == 'int':
        try:
            value = int(value)
        except ValueError:
            raise ValueError(
                f"Invalid integer value for parameter '{param}': {value}")
        if 'values' in spec and value not in spec['values']:
            raise ValueError(
                f"Invalid integer value for parameter '{param}': {value}")
        elif 'range' in spec and not spec['range'][0] <= value <= spec['range'][1]:
            raise ValueError(
                f"Invalid integer value for parameter '{param}': {value}. Must be between {spec['range'][0]} and {spec['range'][1]}")

    if verbose:
        logger.info(
            f"Parameter '{param}' validated successfully with value: {value}")


def setup_mongo_observer(mongo_uri, db_name):
    """
    Sets up a MongoDB observer for experiment tracking.

    This function attempts to connect to a MongoDB instance using the provided
    URI and database name. If the connection is successful, it returns a 
    MongoObserver object. If the connection fails, it falls back to a 
    FileStorageObserver.

    Args:
        mongo_uri (str): The URI for connecting to the MongoDB instance.
        db_name (str): The name of the database to use for storing experiment data.

    Returns:
        MongoObserver: If the connection to MongoDB is successful.
        FileStorageObserver: If the connection to MongoDB fails.

    Raises:
        Exception: If there is an error connecting to the MongoDB instance.
    """
    try:
        client = MongoClient(mongo_uri, server_api=ServerApi('1'))
        if client.address and client.address[0] == 'localhost':
            logger.info("Connected to local MongoDB observer.")
        else:
            logger.info("Connected to remote MongoDB observer.")
        return MongoObserver(client=client, db_name=db_name)
    except Exception as e:
        logger.info(f"MongoDB connection failed: {e}")
        logger.info(
            "Falling back to file observer in directory ./{db_name}_runs")
        return FileStorageObserver('metatop_runs')


def print_summary(optim_type, nelx, nely, E_max, E_min, nu, vol_frac, betas, eta, pen, epoch_duration, a):
    summary = f"""
    Summary of Input Values:
    ------------------------
    optim_type: {optim_type}
    nelx: {nelx}
    nely: {nely}
    E_max: {E_max}
    E_min: {E_min}
    nu: {nu}
    vol_frac: {vol_frac}
    betas: {betas}
    eta: {eta}
    pen: {pen}
    epoch_duration: {epoch_duration}
    a: {a}
    """
    logger.info(summary)


def log_values(experiment, M):
    try:
        for i in range(3):
            for j in range(3):
                experiment.log_scalar(f"M_{i}{j}", M[i, j])
        logger.info(f'M:\n{M}',)
        eig_vals, eig_vecs = np.linalg.eigh(M)
        for i, v in enumerate(eig_vals):
            experiment.log_scalar(f"Eigenvalue_{i}", v)
            logger.info(f"Eigenvalue_{i}: {v}")
        for i, v in enumerate(eig_vals / np.max(eig_vals)):
            experiment.log_scalar(f"Normed_Eigenvalue_{i}", v)
            logger.info(f"Normed_Eigenvalue_{i}: {v}")
        logger.info(f"Eigenvectors:\n{eig_vecs}")
        ASU = anisotropy_index(M, input_style='mandel')
        for k, v in ASU.items():
            experiment.log_scalar(k, v)
            logger.info(f"{k}: {v}")
        constants = calculate_elastic_constants(M, input_style='mandel')
        for k, v in constants.items():
            experiment.log_scalar(k, v)
            logger.info(f"{k}: {v}")
    except Exception as e:
        logger.info(f"Error logging values: {e}")


def save_fig_and_artifact(experiment, fig, dirname: Path, artifact_name: str):
    try:
        fname = dirname / artifact_name
        fig.savefig(fname)
        experiment.add_artifact(fname, artifact_name)
        logger.info(
            f"Successfully saved figure {fname} and added to artifacts under name {artifact_name}")
    except Exception as e:
        logger.info(f"Error saving figure: {e}")


def save_bmp_and_artifact(experiment, data, dirname, artifact_name):
    try:
        data = data.astype(np.uint8)
        fname = dirname / artifact_name
        img = Image.fromarray(data, mode='L').convert('1')
        img.save(fname)
        experiment.add_artifact(fname, artifact_name)
        logger.info(
            f"Successfully saved image {fname} and added to artifacts under name {artifact_name}")
    except Exception as e:
        logger.info(f"Error saving image: {e}")


def save_history(ex, outdir: Path, ops: OptimizationState):
    x_history = ops.x_history
    evals = ops.evals
    iter_tracker = ops.epoch_iter_tracker
    try:
        pickle_fname = outdir / 'history.pkl'
        with open(pickle_fname, 'wb') as f:
            pickle.dump({'x_history': x_history,
                         'evals': evals,
                         'iter_tracker': iter_tracker},
                        f)
        ex.add_artifact(str(pickle_fname))
        return True
    except Exception as e:
        logger.error(f"Issue with saving pickle: {e}")
        return False


def run_optimization(epoch_duration, betas, ops, x, g_ext, opt, x_history, n, beta):
    logger.info(f"===== Beta: {beta} ({n}/{len(betas)}) =====")
    ops.beta, ops.epoch = beta, n
    try:
        x[:] = opt.optimize(x)
    except nlopt.ForcedStop as e:
        logger.info(f"Optimization stopped: {e}")
        sys.exit(1)
    x_history.append(x.copy())
    opt.set_maxeval(epoch_duration)

    ops.epoch_iter_tracker.append(len(ops.evals))


def generate_output_dir(ex, ops: OptimizationState):
    basis_v = get_basis_str_from_array(ops)
    extremal_mode = ops.extremal_mode
    run_id = ex.current_run._id
    main_name = Path(sys.argv[0]).stem

    if extremal_mode == 1:
        mode = 'unimode'
    elif extremal_mode == 2:
        mode = 'bimode'
    else:
        mode = 'undefined'
    outdir = Path("./output/") / main_name / str(mode) / basis_v / str(run_id)
    outdir.mkdir(parents=True, exist_ok=True)
    return outdir


def print_epoch_summary(opt, i):
    logger.info(f"\n===== Epoch Summary: {i+1} =====")
    logger.info(f"Final Objective: {opt.last_optimum_value():.3f}")
    logger.info(f"Result Code: {opt.last_optimize_result()}")
    logger.info(f"===== End Epoch Summary: {i+1} =====\n")


def save_intermediate_results(ex, ops: OptimizationState, val: float, i: int):
    ops.draw(update_images=True)

    outdir = generate_output_dir(ex, ops)

    save_fig_and_artifact(ex, ops.opt_plot.fig, outdir, f'timeline_e-{i}.png')

    M = np.asarray(mandelize(ops.Chom))
    log_values(ex, M)

    ex.log_scalar('volume_fraction', ops.metamaterial.volume_fraction)
    ex.log_scalar('objective', val)

    x_img = bitmapify(ops.metamaterial.x, ops.img_shape,
                      ops.img_resolution, invert=True)
    save_bmp_and_artifact(ex, x_img, outdir, f'cell_e-{i}.png')


def save_final_results(ex, ops: OptimizationState):
    ops.draw(update_images=True)
    outdir = generate_output_dir(ex, ops)

    # strip off any extra design variables in case we were running like an epigraph or something else with extra variables
    metamate = ops.metamaterial

    final_M = np.asarray(mandelize(ops.Chom))

    w, v = np.linalg.eigh(final_M)
    logger.info(f'Final M:\n{final_M}')
    logger.info(f'Final Eigenvalues: {w}')
    logger.info(f'Final Eigenvalue Ratios: {w / np.max(w)}')
    logger.info(f'Final Eigenvectors:\n{v}')

    elastic_constants = calculate_elastic_constants(
        final_M, input_style='mandel')
    invariants = matrix_invariants(final_M)
    logger.info(f'Final elastic constants:\n{elastic_constants}')
    logger.info(f'Final invariants: \n{invariants}')

    save_history(ex, outdir, ops)

    save_fig_and_artifact(ex, ops.opt_plot.fig, outdir, 'timeline.png')
    # we use the metamate.x here because it was already filtered and projected
    x_img = bitmapify(metamate.x, ops.img_shape,
                      ops.img_resolution, invert=True)
    save_bmp_and_artifact(ex, x_img, outdir, 'cell.png')
    save_bmp_and_artifact(ex, np.tile(x_img, (4, 4)), outdir, 'array.png')

    ex.info['final_M'] = final_M
    ex.info['eigvals'] = w
    ex.info['norm_eigvals'] = w / np.max(w)
    ex.info['elastic_constants'] = elastic_constants
    ex.info['invariants'] = invariants


def get_basis_str_from_array(ops):
    basis_v_str = [k for k, v in V_DICT.items() if np.allclose(v, ops.basis_v)]
    if len(basis_v_str) > 1:
        raise ValueError("There is ambiguity with which basis you used.")
    return basis_v_str[0]


def seed_density(init_run_idx, size, epigraph=False):
    '''
    Seed the initial density, either with the final output density from another run, or with random distribution
    '''
    if init_run_idx is not None:
        pickle_artifact = extract_pickle_artifact(init_run_idx)
        if pickle_artifact is not None:
            # Use the final output density from another run, dropping the final t value
            pickle = pickle_artifact.as_type(PA).render()
            try:
                x = pickle['x']
                logger.warning(
                    f"I think this final density was generated using the epigraph form, the size is {x.size:d}")
            except:
                x = pickle['x_history'][-1]
                logger.warning(
                    f"I think this final density was not generated using the epigraph form, the size is {x.size:d}")

            if not isinstance(x, np.ndarray):
                logger.info(
                    f"It looks like 'x' from the pickle artifact from run {init_run_idx} was not a numpy array.")
                logger.info(f"Falling back to random density instead.")
                init_run_idx = None
            elif np.size(x) != size:
                logger.info(
                    f"It looks like the loaded initial density from run {init_run_idx} is a different size than needed for this run.")
                logger.info(
                    f"loaded density size {np.size(x)}, function space size: {size}")
                logger.info(f"Falling back to random density instead.")
                init_run_idx = None
            else:
                logger.info(
                    f"Successfully seeded intial density with final density from run {init_run_idx}.")
        else:
            logger.info(
                f"Pickle artifact not found for index {init_run_idx:d}")
            logger.info("Seeding with random density instead.")
            init_run_idx = None  # Fallback to random seeding
    if init_run_idx is None:
        logger.info("Seeding with random density")
        x = np.random.uniform(0., 1., size)

    return x


def extract_pickle_artifact(init_run_idx):
    loader = ExperimentLoader(mongo_uri=MONGO_URI, db_name=MONGO_DB_NAME)
    exp = loader.find_by_id(init_run_idx)
    # sift through the artifacts of the experiment and return the pickle artifact or None if nothing is found
    return next((v for k, v in exp.artifacts.items() if '.pkl' in k), None)


if __name__ == "__main__":
    for _ in range(10_000):
        for param in PARAMETER_SPECS.keys():
            v = get_random_value(param)
            validate_param_value(param, v, verbose=False)
