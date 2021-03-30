import multiprocessing
import multiprocessing as mp
from multiprocessing import Pipe
from multiprocessing import Process
from multiprocessing import Pool
from multiprocessing.connection import Connection
from multiprocessing.managers import BaseManager
from multiprocessing.queues import Queue

import numpy as np

from multiagent.core import RoleTypes, UnitAttackTypes
from league.league import League
from league.payoff import Payoff
from league.run.self_play_run import run_sequential_league
from league.utils.coordinator import Coordinator
from league.utils.team_composer import TeamComposer

import os
import threading
from types import SimpleNamespace as SN

from utils.run_utils import args_sanity_check


def run(_run, _config, _log):
    # check args sanity
    _config = args_sanity_check(_config, _log)

    args = SN(**_config)
    args.device = "cuda" if args.use_cuda else "cpu"

    #
    #
    # League Training
    #
    #
    team_size = 3
    team_composer = TeamComposer(RoleTypes, UnitAttackTypes)
    team_compositions = [team_composer.compose_unique_teams(team_size)[0]]  # TODO change back to all comps
    league = League(initial_agents=team_compositions, payoff=Payoff())
    coordinator = Coordinator(league)

    # players_n = league.roles_per_initial_agent() * len(team_compositions)
    players_n = len(team_compositions)
    mp.set_start_method('spawn')
    processes = []  # processes list - each representing a runner playing a match
    parent_conns = []  # parent connections

    for idx in range(players_n):
        parent_conn, child_conn = Pipe()
        parent_conns.append(parent_conn)

        proc = Process(target=run_sequential_league, args=(args, _log, child_conn, idx))
        processes.append(proc)
        proc.start()

    while not all(parent_conn.closed for parent_conn in parent_conns):
        for parent_conn in parent_conns:
            _handle_match_results(coordinator, parent_conn)

    [proc.join() for proc in processes]
    print("Exiting Main")
    [proc.terminate() for proc in processes]
    print("Stopping all threads")
    for t in threading.enumerate():
        if t.name != "MainThread":
            print("Thread {} is alive! Is daemon: {}".format(t.name, t.daemon))
            t.join(timeout=1)
            print("Thread joined")

    print("Exiting script")

    # Making sure framework really exits
    os._exit(os.EX_OK)


def _handle_match_results(coordinator: Coordinator, parent_conn: Connection):
    data = parent_conn.recv()
    if data['draw'] or all(data['battle_won']):
        result = "draw"
    elif data['battle_won'][0]:
        result = "win"
    else:
        result = "lose"
    coordinator.send_outcome(0, 1, result)
