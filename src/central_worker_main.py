import argparse
import datetime
import os
import sys
import threading
from copy import deepcopy
from os.path import dirname, abspath
from maenv.core import RoleTypes, UnitAttackTypes
from league.processes import REGISTRY as experiment_REGISTRY
from league.components.matchmaking import REGISTRY as matchmaking_REGISTRY
from league.processes.central_worker import CentralWorker
from torch.multiprocessing import set_start_method

set_start_method('spawn', force=True)

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # Lower tf logging level
os.environ['PYGAME_HIDE_SUPPORT_PROMPT'] = "hide"  # Deactivate message from envs built pygame


def main():
    # Handle pre experiment start arguments without sacred
    params = deepcopy(sys.argv)
    parser = argparse.ArgumentParser()
    parser.add_argument('--team_size', default=3, type=int,
                        help="Define the team size. (Only for symmetric, same sized teams in the league")
    parser.add_argument('--league_size', default=1, type=int, help="Define the size of the league (= how many teams)")

    experiment_choices = list(experiment_REGISTRY.keys())
    parser.add_argument('--experiment', default=experiment_choices[0], choices=experiment_choices, type=str,
                        help="Define the type of experiment to run.")
    matchmaking_choices = list(matchmaking_REGISTRY.keys())
    parser.add_argument('--matchmaking', default=matchmaking_choices[2], choices=matchmaking_choices, type=str,
                        help="Define the matchmaking used if the experiment is using matchmaking.")

    parser.add_argument('--desired_role', choices=list(RoleTypes), type=lambda role: RoleTypes[role],
                        help="Define the role each team has to contain")
    parser.add_argument('--desired_attack', choices=list(UnitAttackTypes), type=lambda attack: UnitAttackTypes[attack],
                        help="Define the attack type each team has to contain")
    parser.add_argument('--unique', dest='unique', action='store_true',
                        help="Enforce the desired unit within a team to be unique.")
    parser.set_defaults(unique=True)

    args, _ = parser.parse_known_args(sys.argv)

    # Basics to start a experiment
    src_dir = dirname(abspath(__file__))
    unique_token = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    log_dir = f'{dirname(dirname(abspath(__file__)))}/results/league_{unique_token}'

    worker = CentralWorker(params, args, log_dir, src_dir)
    worker.start()
    worker.join()

    # Clean up after finishing
    print("Exiting Main")
    print("Stopping all threads")
    for t in threading.enumerate():
        if t.name != "MainThread":
            print("Thread {} is alive! Is daemon: {}".format(t.name, t.daemon))
            t.join(timeout=1)
            print("Thread joined")

    print("Exiting script")


if __name__ == '__main__':
    main()
