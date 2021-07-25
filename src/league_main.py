import argparse
import datetime
import os
import sys
import threading
from copy import deepcopy
from os.path import dirname, abspath
from typing import Dict

import torch as th
import json

from maenv.utils.enums import EnumEncoder

from custom_logging.platforms import CustomConsoleLogger
from league.components import PayoffEntry
from league.components.agent_pool import AgentPool
from league.components.matchmaking import PlayEvenlyMatchmaking
from torch.multiprocessing import Barrier, Queue, Manager, current_process
from maenv.core import RoleTypes, UnitAttackTypes
from pathlib import Path

from league.processes.league_coordinator import LeagueCoordinator
from league.utils.team_composer import TeamComposer, Team
from league.processes import REGISTRY as experiment_REGISTRY

th.multiprocessing.set_start_method('spawn', force=True)

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # Lower tf logging level
os.environ['PYGAME_HIDE_SUPPORT_PROMPT'] = "hide"  # Deactivate message from envs built pygame


def _save_league_config():
    Path(log_dir).mkdir(parents=True, exist_ok=True)
    with open(f'{log_dir}/league_config.json', 'w') as fp:
        json.dump(vars(args), fp, cls=EnumEncoder)


#
# Central Worker Process - Parent process spawning training instances
#
if __name__ == '__main__':
    central_logger = CustomConsoleLogger("central-worker")
    central_worker_id = current_process()
    central_logger.info(f'Central working running in process {central_worker_id}')

    # Handle pre experiment start arguments without sacred
    params = deepcopy(sys.argv)
    parser = argparse.ArgumentParser()
    parser.add_argument('--team_size', default=3, type=int,
                        help="Define the team size. (Only for symmetric, same sized teams in the league")
    parser.add_argument('--league_size', default=2, type=int, help="Define the size of the league (= how many teams)")

    experiment_choices = list(experiment_REGISTRY.keys())
    parser.add_argument('--experiment', default=experiment_choices[0], choices=experiment_choices, type=str,
                        help="Define the type of league to run.")

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

    _save_league_config()

    central_logger.info(f'League Parameters: {vars(args)}')
    central_logger.info(f'Logging league instances into directory: {log_dir}')

    #
    # Build league teams
    #
    team_composer = TeamComposer(team_size=args.team_size, characteristics=[RoleTypes, UnitAttackTypes])
    uid = team_composer.get_unique_uid(role_type=args.desired_role, attack_type=args.desired_attack)
    # train, test = train_test_split(np.array(team_composer.teams)) # TODO!
    teams = team_composer.sample(k=args.league_size, contains=uid,
                                 unique=args.unique)  # Sample random teams containing uid
    teams = team_composer.sort_team_units(teams, uid=uid)  # Sort ranged healer first in all teams for later consistency
    n_teams = len(teams)
    n_entries = len(PayoffEntry)

    #
    # Shared objects
    #
    manager = Manager()
    payoff_dict = manager.dict()
    agents_dict = manager.dict()
    payoff = th.zeros((n_teams, n_teams, n_entries)).share_memory_()
    team_allocation: Dict[int, int] = manager.dict()  # Mapping team -> training instance it is running on

    #
    # Communication Infrastructure
    #
    in_queues, out_queues = zip(*[(Queue(), Queue()) for _ in range(n_teams)])

    #
    # Synchronization
    #
    sync_barrier = Barrier(parties=n_teams)

    #
    # Components
    #
    agent_pool = AgentPool(agents_dict=agents_dict)
    matchmaking = PlayEvenlyMatchmaking(agent_pool=agent_pool, payoff=payoff, allocation=team_allocation, teams=teams)

    #
    # Start experiment instances
    #
    procs = []  # All running processes representing an agent playing in the league
    experiment = experiment_REGISTRY[args.experiment]
    for idx, (in_q, out_q, team) in enumerate(zip(in_queues, out_queues, teams)):
        team_allocation[team.id_] = idx
        proc = experiment(
            idx=idx,
            params=params,
            configs_dir=src_dir,
            log_dir=log_dir,
            home_team=team,
            matchmaking=matchmaking,
            agent_pool=agent_pool,
            communication=(in_q, out_q),
            sync_barrier=sync_barrier
        )
        procs.append(proc)

    # Handle message communication within the league
    coordinator = LeagueCoordinator(
        allocation=team_allocation,
        n_senders=n_teams,
        communication=(in_queues, out_queues),
        payoff=payoff,
        sync_barrier=sync_barrier
    )
    coordinator.start()

    [r.start() for r in procs]

    #
    # Wait for experiments to finish
    #
    [r.join() for r in procs]
    coordinator.join()

    #
    # Print Payoff tensor
    #
    for team in teams:
        tid = team.id_
        index = team_allocation[tid]
        print(f'Stats for {team} from instance {index}')
        for entry in PayoffEntry:
            print(f"{entry.name} {payoff[index, :, entry]}")

    # Clean up after finishing
    print("Exiting Main")
    print("Stopping all threads")
    for t in threading.enumerate():
        if t.name != "MainThread":
            print("Thread {} is alive! Is daemon: {}".format(t.name, t.daemon))
            t.join(timeout=1)
            print("Thread joined")

    print("Exiting script")
