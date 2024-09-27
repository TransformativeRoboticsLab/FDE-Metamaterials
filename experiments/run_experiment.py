import argparse
import subprocess

from andreassen import ex as andreassen_ex
from extremal import ex as extremal_ex

from metatop import V_DICT


def select_experiment(name):
    name = name.rstrip('.py')
    
    experiments = {
        'andreassen': andreassen_ex,
        'extremal': extremal_ex
    }

    if name not in experiments:
        valid_names = ', '.join(experiments.keys())
        raise ValueError(f"Experiment '{name}' not found. Valid names are: [{valid_names}]")
    
    return experiments[name]

def run_experiment(experiment_name, num_runs, sacred_args):
    # ex = select_experiment(experiment_name)
    experiment_name = experiment_name.rstrip('.py') + '.py'
    
    print(f"Running experiment '{experiment_name}' with updated config parameters: {sacred_args[1:]}")
    
    for run_id in range(1, num_runs+1):
        print(f"Starting run {run_id}/{num_runs}")
        # the dummy is needed because Sacred expects the first argument to be the script name
        # otherwise the args are passed formated as if they were passed to the script
        # ex.run_commandline(['dummy'] + sacred_args)
        subprocess.run(['python', 'experiments/' + experiment_name] + sacred_args)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Run the specific experiment with Sacred')
    parser.add_argument('experiment_name', type=str, help='The experiment to run')
    parser.add_argument('num_runs', type=int, help='The number of runs to execute')
    parser.add_argument('sacred_args', nargs=argparse.REMAINDER, help="CLI style arguments to pass to the experiment (e.g., with seed=0)")

    args = parser.parse_args()

    run_experiment(args.experiment_name, args.num_runs, args.sacred_args)