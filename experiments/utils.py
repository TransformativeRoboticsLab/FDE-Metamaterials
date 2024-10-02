import numpy as np
from PIL import Image
from pymongo.mongo_client import MongoClient
from pymongo.server_api import ServerApi
from sacred.observers import FileStorageObserver, MongoObserver

from metatop.filters import jax_projection, jax_simp
from metatop.mechanics import anisotropy_index, calculate_elastic_constants


def setup_observer(mongo_uri, db_name):
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
