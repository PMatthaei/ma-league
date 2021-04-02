import os
import sys
from copy import deepcopy
from os.path import dirname, abspath

from multiagent.core import RoleTypes, UnitAttackTypes
from multiprocessing.connection import Pipe
from multiprocessing.dummy import Manager

from sacred import SETTINGS, Experiment
from sacred.observers import FileStorageObserver
from sacred.utils import apply_backspaces_and_linefeeds

from league.components.payoff import Payoff
from league.league import League
from league.processes.trainer_process import TrainerProcess
from league.components.coordinator import Coordinator
from league.processes.message_handler_process import MessageHandler
from league.utils.team_composer import TeamComposer
from utils.logging import LeagueLogger
from utils.main_utils import get_default_config, get_config, load_match_build_plan, recursive_dict_update

SETTINGS['CAPTURE_MODE'] = "fd"  # set to "no" if you want to see stdout/stderr in console
logger = LeagueLogger.console_logger()

ex = Experiment("ma-league")
ex.logger = logger
ex.captured_out_filter = apply_backspaces_and_linefeeds

results_path = os.path.join(dirname(dirname(abspath(__file__))), "results")


@ex.main
def league_main(_config):
    # Build league teams
    team_size = _config["team_size"]
    team_composer = TeamComposer(RoleTypes, UnitAttackTypes)
    team_compositions = team_composer.compose_unique_teams(team_size)[:2]  # TODO change back to all comps

    # Shared objects
    manager = Manager()
    p_matrix = manager.dict()
    players = manager.list()

    # Infrastructure
    processes = []
    league_conns = []

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

    # Handle message communication within the league
    handler = MessageHandler(coordinator, league_conns)
    handler.start()
    handler.join()

    # Print win rates for all players
    league.print_payoff()

    # Wait for processes to finish
    [proc.join() for proc in processes]


if __name__ == '__main__':
    params = deepcopy(sys.argv)

    # Get the defaults from default.yaml
    main_path = os.path.dirname(__file__)
    config_dict = get_default_config(main_path)

    # Load league base config
    league_config = get_config(params, "--league-config", "leagues", path=main_path)

    # Load env base config
    env_config = get_config(params, "--env-config", "envs", path=main_path)

    # Load build plan if configured
    env_args = env_config['env_args']
    if "match_build_plan" in env_args:
        load_match_build_plan(main_path, env_args)

    # Load algorithm base config
    alg_config = get_config(params, "--config", "algs", path=main_path)

    # Integrate loaded dicts into main dict
    config_dict = recursive_dict_update(config_dict, league_config)
    config_dict = recursive_dict_update(config_dict, env_config)
    config_dict = recursive_dict_update(config_dict, alg_config)

    # now add all the config to sacred
    ex.add_config(config_dict)

    # Save to disk by default for sacred
    logger.info("Saving to FileStorageObserver in results/sacred.")
    file_obs_path = os.path.join(results_path, "sacred")
    ex.observers.append(FileStorageObserver(file_obs_path))

    ex.run_commandline(params)
