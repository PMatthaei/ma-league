import argparse
import datetime
import json
import os
import sys
import threading
from copy import deepcopy
from os.path import dirname, abspath
from pathlib import Path

from maenv.core import RoleTypes, UnitAttackTypes
from maenv.utils.enums import EnumEncoder

from league.processes import REGISTRY as experiment_REGISTRY
from league.components.matchmaking import REGISTRY as matchmaking_REGISTRY
from league.processes.central_worker import CentralWorker
from torch.multiprocessing import set_start_method


set_start_method('spawn', force=True)

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # Lower tf logging level
os.environ['PYGAME_HIDE_SUPPORT_PROMPT'] = "hide"  # Deactivate message from envs built pygame


def main():
    # Handle pre experiment start arguments without sacred
    params = deepcopy(sys.argv[1:])
    parser = argparse.ArgumentParser()
    sub_parsers = parser.add_subparsers(dest="cmd", help='Sub-Commands')

    #
    # Main Command
    #
    parser.add_argument('--team_size', default=5, type=int,
                        help="Define how many agents comprise a team. Each team has same size")
    parser.add_argument('--league_size', default=2, type=int,
                        help="Define the size of the league (= how many teams)")

    experiment_choices = list(experiment_REGISTRY.keys())
    parser.add_argument('--experiment', default=experiment_choices[0], choices=experiment_choices, type=str,
                        help="Define the type of experiment to run.")
    matchmaking_choices = list(matchmaking_REGISTRY.keys())
    parser.add_argument('--matchmaking', default=matchmaking_choices[0], choices=matchmaking_choices, type=str,
                        help="Define the matchmaking used if the experiment is using matchmaking.")
    #
    # Arguments for instances
    #
    parser.add_argument('--league-config', required=True, help="Define which league to use.")
    parser.add_argument('--env-config', required=True, help="Define which env to use.")
    parser.add_argument('--config', required=True, help="Define which algorithm to use.")

    #
    # Sub-Command: Force Unit in Team
    #
    force_unit_parser = sub_parsers.add_parser('force-unit',
                                               help='Forces the team composer to create teams with one ore more specified unit(s).'
                                                    'A unit is specified via its role and attack type.')
    force_unit_parser.add_argument('--role', choices=list(RoleTypes),
                                   type=lambda role: RoleTypes[role],
                                   default=list(RoleTypes)[0],
                                   help="Define a role of an unit the team has to contain")
    force_unit_parser.add_argument('--attack', choices=list(UnitAttackTypes),
                                   type=lambda attack: UnitAttackTypes[attack],
                                   default=list(UnitAttackTypes)[0],
                                   help="Define an attack type eof an unit the team has to contain")
    force_unit_parser.add_argument('--unique', dest='unique', action='store_true',
                                   help="Enforce the desired unit within a team to be unique.")
    force_unit_parser.set_defaults(unique=True)

    args, _ = parser.parse_known_args(params)

    # Basics to start a experiment
    src_dir = f"{dirname(abspath(__file__))}"  # Path to src directory
    unique_token = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    log_dir = f'{dirname(dirname(abspath(__file__)))}/results/league_{unique_token}'  # Logs of the league

    Path(log_dir).mkdir(parents=True, exist_ok=True)  # Save config
    with open(f'{log_dir}/league_config.json', 'w') as fp:
        json.dump(vars(args), fp, cls=EnumEncoder, indent=4, sort_keys=True)

    #
    # Start Central Experiment
    #
    worker = CentralWorker(params, args, log_dir, src_dir)
    worker.start()
    worker.join()
    #
    #
    #

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
