import os
import sys
from copy import deepcopy
from os.path import dirname, abspath

from multiagent.core import RoleTypes, UnitAttackTypes
from multiprocessing.connection import Pipe, Connection
from multiprocessing.dummy import Manager, Process

from sacred import SETTINGS, Experiment
from sacred.observers import FileStorageObserver
from sacred.utils import apply_backspaces_and_linefeeds

from league.league import League
from league.components.payoff import Payoff
from league.trainer_process import TrainerProcess
from league.components.coordinator import Coordinator
from league.utils.process_message_handler import ProcessMessageHandler
from league.utils.team_composer import TeamComposer
from utils.logging import LeagueLogger
from utils.main_utils import get_default_config, get_config, load_match_build_plan, recursive_dict_update

SETTINGS['CAPTURE_MODE'] = "fd"  # set to "no" if you want to see stdout/stderr in console
logger = LeagueLogger.console_logger()

ex = Experiment("pymarl")
ex.logger = logger
ex.captured_out_filter = apply_backspaces_and_linefeeds

results_path = os.path.join(dirname(dirname(abspath(__file__))), "results")


@ex.main
def league_main():
    # Build league teams
    team_size = 3
    team_composer = TeamComposer(RoleTypes, UnitAttackTypes)
    team_compositions = team_composer.compose_unique_teams(team_size)[:2]  # TODO change back to all comps

    # Shared objects
    manager = Manager()
    p_matrix = manager.dict()
    players = manager.list()

    processes = []  # processes list - each representing a runner playing a match
    league_conns = []  # parent connections

    # Create league
    payoff = Payoff(p_matrix=p_matrix, players=players)
    league = League(initial_agents=team_compositions, payoff=payoff)
    coordinator = Coordinator(league)

    # Start league training
    for idx in range(league.size):
        league_conn, conn = Pipe()
        league_conns.append(league_conn)

        player = league.get_player(idx)

        proc = TrainerProcess(idx, player, conn)
        processes.append(proc)
        proc.start()

    handler = ProcessMessageHandler(coordinator, league_conns)
    handler.start()
    handler.join()

    print(payoff.p_matrix)
    idxs = list(range(league.size))
    for idx in idxs:
        print(f"Win rates for player {idx}")
        print(payoff[idx, idxs])

    # Wait for processes to finish
    [proc.join() for proc in processes]


if __name__ == '__main__':
    params = deepcopy(sys.argv)

    # Get the defaults from default.yaml
    main_path = os.path.dirname(__file__)
    config_dict = get_default_config(main_path)

    # Load algorithm and env base configs
    env_config = get_config(params, "--env-config", "envs", path=main_path)

    # Load build plan if configured
    env_args = env_config['env_args']
    if "match_build_plan" in env_args:
        load_match_build_plan(main_path, env_args)

    alg_config = get_config(params, "--config", "algs", path=main_path)
    config_dict = recursive_dict_update(config_dict, env_config)
    config_dict = recursive_dict_update(config_dict, alg_config)

    # now add all the config to sacred
    ex.add_config(config_dict)

    # Save to disk by default for sacred
    logger.info("Saving to FileStorageObserver in results/sacred.")
    file_obs_path = os.path.join(results_path, "sacred")
    ex.observers.append(FileStorageObserver.create(file_obs_path))

    ex.run_commandline(params)
