import datetime
import os
import pprint
import sys
from copy import deepcopy
from multiprocessing import Barrier
from os.path import dirname, abspath

import torch
from maenv.core import RoleTypes, UnitAttackTypes
from multiprocessing import Pipe
from multiprocessing import Manager

from sacred import SETTINGS, Experiment
from sacred.observers import FileStorageObserver
from sacred.utils import apply_backspaces_and_linefeeds

from league.components.payoff import Payoff
from league.league import League
from league.processes.league_process import LeagueRun
from league.processes.league_coordinator import LeagueCoordinator
from league.utils.team_composer import TeamComposer
from utils.logging import LeagueLogger
from utils.main_utils import get_default_config, get_config, load_match_build_plan, recursive_dict_update, config_copy, \
    set_agents_only

import numpy as np
import torch as th
from types import SimpleNamespace

from utils.run_utils import args_sanity_check

SETTINGS['CAPTURE_MODE'] = "fd"  # set to "no" if you want to see stdout/stderr in console
logger = LeagueLogger.console_logger()

ex = Experiment("ma-league")
ex.logger = logger
ex.captured_out_filter = apply_backspaces_and_linefeeds

results_path = os.path.join(dirname(dirname(abspath(__file__))), "results")


def run(_run, _config, _log):
    _config = args_sanity_check(_config, _log)
    _config['play_mode'] = "self"
    set_agents_only(_config)

    torch.multiprocessing.set_start_method('spawn')

    args = SimpleNamespace(**_config)
    args.device = "cuda" if args.use_cuda else "cpu"

    logger = LeagueLogger(_log)
    _log.info("Experiment Parameters:")
    experiment_params = pprint.pformat(_config,
                                       indent=4,
                                       width=1)
    _log.info("\n\n" + experiment_params + "\n")

    # configure tensorboard logger
    unique_token = "{}__{}".format(args.name, datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S"))
    args.unique_token = unique_token
    if args.use_tensorboard:
        tb_logs_direc = os.path.join(dirname(dirname(abspath(__file__))), "results", "tb_logs")
        tb_exp_direc = os.path.join(tb_logs_direc, "{}").format(unique_token)
        logger.setup_tensorboard(tb_exp_direc)

    # sacred is on by default
    logger.setup_sacred(_run)

    # Build league teams
    team_size = _config["team_size"]
    team_composer = TeamComposer(RoleTypes, UnitAttackTypes)
    team_compositions = [team_composer.compose_unique_teams(team_size)[0]]

    # Shared objects
    manager = Manager()
    p_matrix = manager.dict()  # Payoff matrix
    players = manager.list()  # List of current players

    # Infrastructure
    runs = []  # All running processes representing an agent playing in the league
    run_conns = []  # Connections from the parent process to each run

    # Create league
    payoff = Payoff(p_matrix=p_matrix, players=players)
    league = League(initial_agents=team_compositions, payoff=payoff)

    # Used to ensure all processes have setup before starting to train
    setup_barrier = Barrier(league.size)

    # Start league training
    for idx in range(league.size):
        run_conn, conn = Pipe()  # TODO: downgrade to queue if no msg from here to child conn needed
        run_conns.append(run_conn)

        player = league.get_player(idx)

        league_run = LeagueRun(home=player, barrier=setup_barrier, conn=conn, args=args, logger=logger)
        runs.append(league_run)
        league_run.start()

    # Handle message communication within the league
    coordinator = LeagueCoordinator(league, run_conns)
    coordinator.start()
    coordinator.join()

    # Wait for processes to finish
    [r.join() for r in runs]

    # Print win rates for all players
    league.print_payoff()


@ex.main
def league_main(_run, _config, _log):
    # Load config and logger
    config = config_copy(_config)
    np.random.seed(config["seed"])
    th.manual_seed(config["seed"])
    config['env_args']['seed'] = config["seed"]

    # run the framework
    run(_run, config, _log)


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
