import argparse
import subprocess
import sys
from pathlib import Path

from nlopt import ForcedStop
from utils import get_random_value, validate_param_value


def run_experiment(experiment_name, num_runs, sacred_args, random_params):
    experiment_name = experiment_name.rstrip('.py') + '.py'

    print(f"Running experiment '{experiment_name}' with config parameters.")

    run_id = 1
    while num_runs == -1 or run_id <= num_runs:
        random_args = []

        # Randomize specified parameters
        if random_params:
            for param in random_params:
                try:
                    value = get_random_value(param)
                except ValueError as e:
                    print(f"Randomization failed for parameter '{param}': {e}")
                    continue
                random_args.append(f"{param}={value}")
                print(f"Randomized parameter '{param}' with value: {value}")

        num_runs_str = 'infinity' if num_runs == -1 else str(num_runs)
        print(f"Starting run {run_id}/{num_runs_str}")
        proc = ['python', 'experiments/' + experiment_name]
        if len(sacred_args) == 0 and len(random_args) > 0:
            proc += ['with'] + random_args
        else:
            proc += sacred_args + random_args
        print(proc)

        try:
            subprocess.run(proc)
        except ForcedStop as e:
            print(f"Experiment was stopped by program: {e}")
        except Exception as e:
            print(f"An error occurred while running the experiment: {e}")

        run_id += 1


if __name__ == '__main__':
    description = """
Run the specific experiment with Sacred, allowing parameter specification or randomization. 
A typical use looks like:

`python run_experiment.py extremal 10 with param1=value1 param2=value2 --randomize param3 param4`

This will run the `extremal` experiment 10 times with the specified parameters for params 1 and 2 and randomize `param3` and `param4` each time.

Note:
- `num_runs` can be set to -1 for infinite runs.
    """
    parser = argparse.ArgumentParser(description=description,
                                     formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument('experiment_name', type=str,
                        help='The python experiment file to run. Can use the format `experiment_name.py` or `experiment_name`.')
    parser.add_argument(
        'num_runs', type=int, help='The number of runs to execute. -1 for infinite runs.')
    parser.add_argument('sacred_args', nargs='*',
                        help="Sacred CLI style arguments to pass to the experiment (e.g., `with param1=value1 param2=value2`). Note: Parameters must be preceded by the word 'with'.")
    parser.add_argument('--randomize', nargs='*',
                        help="List of parameters to randomize (e.g., `--randomize param3 param4`)")

    args = parser.parse_args()

    for arg in args.sacred_args:
        if '=' in arg:
            param, value = arg.split('=')
            try:
                validate_param_value(param, value, verbose=False)
            except ValueError as e:
                print(f"Validation failed for parameter '{param}': {e}")
                raise

    run_experiment(args.experiment_name, args.num_runs,
                   args.sacred_args, args.randomize)
