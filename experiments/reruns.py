import json  # Needed to safely pass complex dicts/lists if necessary
import subprocess
import sys
from os import getenv as env

import dotenv
from incense import ExperimentLoader
from loguru import logger
# We don't need 'ex' from matrix_matching here anymore for running
# from matrix_matching import ex
# from run_experiment import run_experiment # Likely not needed
from sacred import \
    Experiment  # Still useful for config scope? Maybe not needed.

from experiments.utils import *

dotenv.load_dotenv()
MONGO_URI = env('LOCAL_MONGO_URI')
MONGO_DB_NAME = env('LOCAL_MONGO_DB_NAME')
MONGO_EXP_NAME = env('LOCAL_MONGO_EXP_NAME')

DEFAULT_FILTER_TAGS = ['BAD', 'DUPE']
DIST_TYPES = ['fro']
DB_QUERY = {"$and": [
    {'experiment.name': MONGO_EXP_NAME},
    {'status': 'COMPLETED'},
    {'omniboard.tags': {'$nin': DEFAULT_FILTER_TAGS}},
    {'config.nu': {'$eq': 0.4}},
    {'config.objective_type': {'$eq': None}},
    {'config.dist_type': {'$eq': 'fro'}},
    {'$or': [
        {'config.init_run_idx': None},
        {'config.init_run_idx': {'$exists': False}}
    ]}
]}

# Define the path to your main experiment script
# Adjust if needed
MAIN_EXPERIMENT_SCRIPT = "./experiments/matrix_matching.py"


def rerun_via_subprocess(exps, config_mods):

    for i, exp in enumerate(exps, 1):
        # Create a copy of the original config to avoid side effects
        original_config = exp.to_dict()['config'].copy()
        original_config.pop('_id', None)  # Good practice

        # --- Prepare the config updates for this specific rerun ---
        current_config_updates = original_config.copy()  # Start with original

        # Calculate final beta (same logic as before)
        if 'start_beta' in current_config_updates and 'n_betas' in current_config_updates and isinstance(current_config_updates.get('n_betas'), int) and current_config_updates['n_betas'] > 0:
            current_config_updates['start_beta'] = [current_config_updates['start_beta'] * 2 ** j
                                                    for j in range(current_config_updates['n_betas'])][-1]
        else:
            logger.warning(
                f"Could not calculate final beta for exp {exp.id}. Using default 64.")
            current_config_updates['start_beta'] = 64

        # Apply the general modifications
        current_config_updates.update(config_mods)

        # Set the reference to the original run ID
        current_config_updates['init_run_idx'] = exp.id

        # --- Construct the command line arguments ---
        # Base command: python executable, script path, 'with'
        command = [sys.executable, MAIN_EXPERIMENT_SCRIPT, 'with']

        # Add config updates as key=value pairs
        # Handle potential spaces or special characters using JSON encoding for safety
        for key, value in current_config_updates.items():
            # Simple types can often be passed directly as strings
            # For lists/dicts, JSON encoding is safer
            if isinstance(value, (dict, list)):
                command.append(f"{key}='{json.dumps(value)}'")
            elif isinstance(value, str):
                # Basic quoting for strings, might need more robust handling
                command.append(f"{key}='{value}'")
            else:
                # Numbers, booleans
                command.append(f"{key}={value}")

        logger.info(f"Rerunning {i}/{len(exps)} - Initial Run Index: {exp.id}")
        logger.debug(f"Executing command: {' '.join(command)}")

        try:
            # Execute the command as a separate process
            # Check=True will raise CalledProcessError if the script exits with non-zero code
            result = subprocess.run(
                command, check=True, capture_output=True, text=True)
            logger.info(f"Rerun for {exp.id} completed successfully.")
            # You can log result.stdout or result.stderr if needed
            # logger.debug(f"Stdout:\n{result.stdout}")
            # logger.debug(f"Stderr:\n{result.stderr}")

        except subprocess.CalledProcessError as e:
            logger.error(
                f"Rerun for {exp.id} failed with exit code {e.returncode}.")
            logger.error(f"Command: {' '.join(e.cmd)}")
            logger.error(f"Stderr:\n{e.stderr}")
            logger.error(f"Stdout:\n{e.stdout}")
            # Decide whether to continue or stop
            # break # Example: stop on first failure
        except Exception as e:
            logger.error(
                f"An unexpected error occurred launching subprocess for {exp.id}: {e}", exc_info=True)
            # break # Example: stop on unexpected errors


if __name__ == '__main__':
    loader = ExperimentLoader(
        mongo_uri=MONGO_URI,
        db_name=MONGO_DB_NAME
    )

    exps = loader.find(DB_QUERY)
    logger.info(f"Found {len(exps):d} experiments to rerun")

    if not exps:
        logger.info("No experiments matched the query. Exiting.")
    else:
        config_mods = dict(
            eta=0.15,
            n_betas=1,
            warm_start_duration=0,
            epoch_duration=1,
        )
        # Call the subprocess-based rerun function
        rerun_via_subprocess(exps, config_mods)
