import os
import pickle
import random
import sys

import nlopt
import numpy as np
from PIL import Image
from pymongo.mongo_client import MongoClient
from pymongo.server_api import ServerApi
from sacred.observers import FileStorageObserver, MongoObserver

from metatop import V_DICT
from metatop.filters import jax_projection, jax_simp
from metatop.image import bitmapify
from metatop.mechanics import (anisotropy_index, calculate_elastic_constants,
                               matrix_invariants)

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
    'epoch_duration':{
        'type': 'int',
        'range': (1, 10_000),
    },
    'extremal_mode': {
        'type': 'int',
        'values': (1, 2),
    },
    'basis_v': {
        'type': 'cat',
        'values': ['BULK', 'VERT', 'HSA', 'SHEAR']
    },
    'objective_type': {
        'type': 'cat',
        'values': ('ray', 'ray_sq')
    },
    'nelx': {
        'type': 'int',
        'range': (1, 1_000),
    },
    'nely': {
        'type': 'int',
        'range': (1, 1_000),
    },
    'norm_filter_radius': {
        'type': 'float',
        'range': (0.01, 0.5),
    },
    'g_vec_eps': {
        'type': 'float',
        'values': tuple(10**(-n) for n in range(6)),
    },
    'weight_scaling_factor': {
        'type': 'float',
        'values': tuple(10**(-n) for n in range(4)),
    },
}

def get_random_value(param):
    spec = PARAMETER_SPECS.get(param, None)
    if spec is None:
        raise ValueError(f"Parameter '{param}' not found in PARAMETER_SPECS")

    s_type = spec.get('type', None)
    if s_type== 'cat':
        return random.choice(spec['values'])
    elif s_type  == 'float':
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
    print(f"Validating parameter '{param}' with value: {value}")
    spec = PARAMETER_SPECS.get(param, None)
    if spec is None:
        print(f"Parameter '{param}' not found in PARAMETER_SPECS")
        return False

    s_type = spec.get('type', None)
    if s_type == 'cat':
        if value not in spec['values']:
            raise ValueError(f"Invalid categorical value for parameter {param}: {value}. Values must be one of {spec['values']}")
    elif s_type == 'float':
        try:
            value = float(value)
        except ValueError:
            raise ValueError(f"Invalid float value for parameter '{param}': {value}")
        if 'values' in spec and value not in spec['values']:
            raise ValueError(f"Invalid float value for parameter '{param}': {value}. Values must be one of {spec['values']}")
        elif 'range' in spec and not spec['range'][0] <= value <= spec['range'][1]:
            raise ValueError(f"Invalid float value for parameter '{param}': {value}. Must be between {spec['range'][0]} and {spec['range'][1]}")
    elif s_type == 'int':
        try:
            value = int(value)
        except ValueError:
            raise ValueError(f"Invalid integer value for parameter '{param}': {value}")
        if 'values' in spec and value not in spec['values']:
            raise ValueError(f"Invalid integer value for parameter '{param}': {value}")
        elif 'range' in spec and not spec['range'][0] <= value <= spec['range'][1]:
            raise ValueError(f"Invalid integer value for parameter '{param}': {value}. Must be between {spec['range'][0]} and {spec['range'][1]}")
    
    if verbose:
        print(f"Parameter '{param}' validated successfully with value: {value}")


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
            print("Connected to local MongoDB observer.")
        else:
            print("Connected to remote MongoDB observer.")
        return MongoObserver(client=client, db_name=db_name)
    except Exception as e:
        print(f"MongoDB connection failed: {e}")
        print("Falling back to file observer in directory ./{db_name}_runs")
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
    print(summary)

def forward_solve(x, metamaterial, ops, simp=False):
    metamaterial.x.vector()[:] = jax_projection(ops.filt_fn(x), ops.beta, ops.eta)
    if simp:
        metamaterial.x.vector()[:] = jax_simp(metamaterial.x.vector()[:], ops.pen)
    m = np.diag(np.array([1, 1, np.sqrt(2)]))
    C = m @ np.asarray(metamaterial.solve()[1]) @ m
    return C

def log_values(experiment, C):
    try:
        for i in range(3):
            for j in range(3):
                experiment.log_scalar(f"C_{i}{j}", C[i, j])
        print('C:\n', C)
        w = np.linalg.eigvalsh(C)
        for i, v in enumerate(w):
            experiment.log_scalar(f"Eigenvalue_{i}", v)
            print(f"Eigenvalue_{i}: {v}")
        for i, v in enumerate(w / np.max(w)):
            experiment.log_scalar(f"Normed_Eigenvalue_{i}", v)
            print(f"Normed_Eigenvalue_{i}: {v}")
        ASU = anisotropy_index(C, input_style='mandel')
        for k, v in ASU.items():
            experiment.log_scalar(k, v)
            print(f"{k}: {v}")
        constants = calculate_elastic_constants(C, input_style='mandel')
        for k, v in constants.items():
            experiment.log_scalar(k, v)
            print(f"{k}: {v}")
    except Exception as e:
        print(f"Error logging values: {e}")

def save_fig_and_artifact(experiment, fig, outname, artifact_name):
    try:
        fname = outname + f'_{artifact_name}'
        fig.savefig(fname)
        experiment.add_artifact(fname, artifact_name)
        print(f"Successfully saved figure {fname} and added to artifacts under name {artifact_name}")
    except Exception as e:
        print(f"Error saving figure: {e}")

def save_bmp_and_artifact(experiment, data, outname, artifact_name):
    try:
        data = data.astype(np.uint8)
        fname = outname + f'_{artifact_name}'
        img = Image.fromarray(data, mode='L').convert('1')
        img.save(fname)
        experiment.add_artifact(fname, artifact_name)
        print(f"Successfully saved image {fname} and added to artifacts under name {artifact_name}")
    except Exception as e:
        print(f"Error saving image: {e}")

def run_optimization(epoch_duration, betas, ops, x, g_ext, opt, x_history, n, beta):
    print(f"===== Beta: {beta} ({n}/{len(betas)}) =====")
    ops.beta, ops.epoch = beta, n
    try:
        x[:] = opt.optimize(x)
    except nlopt.ForcedStop as e:
        print(f"Optimization stopped: {e}")
        sys.exit(1)
    x_history.append(x.copy())
    opt.set_maxeval(epoch_duration)

    ops.epoch_iter_tracker.append(len(ops.evals))

if __name__ == "__main__":
    for _ in range(10_000):
        for param in PARAMETER_SPECS.keys():
            v = get_random_value(param)
            validate_param_value(param, v, verbose=False)

        
def generate_output_filepath(ex, extremal_mode, basis_v, seed):
    run_id = ex.current_run._id
    dirname = './output/epigraph'
    if not os.path.exists(dirname):
        os.makedirs(dirname)
    fname = str(run_id)
    fname += f'_{basis_v}'
    fname += f'_m_{extremal_mode}'
    fname += f'_seed_{seed}'
    outname = dirname + '/' + fname
    return run_id,outname

def print_epoch_summary(opt, i):
    print(f"\n===== Epoch Summary: {i+1} =====")
    print(f"Final Objective: {opt.last_optimum_value():.3f}")
    print(f"Result Code: {opt.last_optimize_result()}")
    print(f"===== End Epoch Summary: {i+1} =====\n")

def log_and_save_results(ex, run_id, outname, metamate, img_rez, img_shape, ops, x, g_ext, i):
    g_ext.update_plot(x[:-1])
    save_fig_and_artifact(ex, g_ext.fig, outname, f'{run_id}_timeline_e-{i+1}.png')

    metamate.x.vector()[:] = x[:-1]
    ex.log_scalar('volume_fraction', metamate.volume_fraction)
    log_values(ex, forward_solve(x[:-1], metamate, ops))

    x_img = bitmapify(metamate.x, img_shape, img_rez, invert=True)
    save_bmp_and_artifact(ex, x_img, outname, f'{run_id}_cell_e-{i+1}.png')

def save_results(ex, run_id, outname, metamate, img_rez, img_shape, ops, x, g_ext, x_history):
    final_C = forward_solve(x[:-1], metamate, ops)
    
    w, v = np.linalg.eigh(final_C)
    print('Final C:\n', final_C)
    print('Final Eigenvalues:\n', w)
    print('Final Eigenvalue Ratios:\n', w / np.max(w))
    print('Final Eigenvectors:\n', v)

    ASU = anisotropy_index(final_C, input_style='mandel')
    elastic_constants = calculate_elastic_constants(final_C, input_style='mandel')
    invariants = matrix_invariants(final_C)
    print('Final ASU:', ASU)
    print('Final Elastic Constants:', elastic_constants)
    print('Final Invariants:', invariants)

    with open(f'{outname}.pkl', 'wb') as f:
        pickle.dump({'x': x,
                     'x_history': x_history,
                     'evals': ops.evals},
                    f)

    save_fig_and_artifact(ex, g_ext.fig, outname, f'{run_id}_timeline.png')
    x_img = bitmapify(metamate.x, img_shape, img_rez, invert=True)
    save_bmp_and_artifact(ex, x_img, outname, f'{run_id}_cell.png')
    save_bmp_and_artifact(ex, np.tile(x_img, (4,4)), outname, f'{run_id}_array.png')

    ex.info['final_C'] = final_C
    ex.info['eigvals'] = w
    ex.info['norm_eigvals'] = w / np.max(w)
    ex.info['eigvecs'] = v
    ex.info['ASU'] = ASU
    ex.info['elastic_constants'] = elastic_constants
    ex.info['invariants'] = invariants
    ex.add_artifact(f'{outname}.pkl')