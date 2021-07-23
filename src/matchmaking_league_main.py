import argparse
import os
import sys
import threading
from copy import deepcopy
from os.path import dirname, abspath

import torch as th

from league.components.agent_pool import AgentPool
from league.components.matchmaking import IteratingMatchmaking
from league.components.payoff_matchmaking import MatchmakingPayoff
from league.processes.training.ensemble_league_process import EnsembleLeagueProcess
from torch.multiprocessing import Barrier, Queue, Manager
from maenv.core import RoleTypes, UnitAttackTypes

from league.utils.team_composer import TeamComposer


th.multiprocessing.set_start_method('spawn', force=True)

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # Lower tf logging level

if __name__ == '__main__':
    params = deepcopy(sys.argv)
    # Handle pre experiment start arguments without sacred
    parser = argparse.ArgumentParser()
    parser.add_argument('--team_size', default=3, type=int)
    # Basics to start a experiment
    args, _ = parser.parse_known_args(sys.argv)
    src_dir = dirname(abspath(__file__))

    # Build league teams
    team_composer = TeamComposer(team_size=args.team_size, characteristics=[RoleTypes, UnitAttackTypes])
    uid = team_composer.get_unique_uid(role_type=RoleTypes.HEALER, attack_type=UnitAttackTypes.RANGED)
    # train, test = train_test_split(np.array(team_composer.teams))
    teams = team_composer.sample(k=2, contains=uid, unique=True)  # Sample 5 random teams that contain a ranged healer
    teams = team_composer.sort_team_units(teams, uid=uid)  # Sort ranged healer first in all teams for later consistency

    # Shared objects
    manager = Manager()
    payoff_dict = manager.dict()
    agents_dict = manager.dict()

    # Infrastructure
    procs = []  # All running processes representing an agent playing in the league
    payoff = MatchmakingPayoff(payoff_dict=payoff_dict)  # Hold results of each match
    agent_pool = AgentPool(agents_dict=agents_dict)  # Hold each trained agent
    matchmaking = IteratingMatchmaking(agent_pool=agent_pool, payoff=payoff)  # Match agents against each other

    # Communication
    in_queues, out_queues = zip(*[(Queue(), Queue()) for _ in range(len(teams))])

    # Synchronization across all league instances
    sync_barrier = Barrier(parties=len(teams))

    # Start league instances
    for idx, (in_q, out_q, team) in enumerate(zip(in_queues, out_queues, teams)):
        proc = EnsembleLeagueProcess(
            params=params,
            src_dir=src_dir,
            home_team=team,
            matchmaking=matchmaking,
            agent_pool=agent_pool,
            queue=(in_q, out_q),
            sync_barrier=sync_barrier
        )
        procs.append(proc)

    [r.start() for r in procs]

    # Wait for processes to finish
    [r.join() for r in procs]

    # Print win rates for all players
    print(payoff)

    # Clean up after finishing
    print("Exiting Main")
    print("Stopping all threads")
    for t in threading.enumerate():
        if t.name != "MainThread":
            print("Thread {} is alive! Is daemon: {}".format(t.name, t.daemon))
            t.join(timeout=1)
            print("Thread joined")

    print("Exiting script")
