import datetime
import os
import pprint
import sys
import threading
from copy import deepcopy

from torch.multiprocessing import Barrier, Queue, Manager
from os.path import dirname, abspath

import torch
from maenv.core import RoleTypes, UnitAttackTypes

from sacred import SETTINGS, Experiment
from sacred.observers import FileStorageObserver
from sacred.utils import apply_backspaces_and_linefeeds

from league import SimpleLeague
from league.components.payoff import Payoff
from league.processes.league_process import LeagueProcess
from league.processes.league_coordinator import LeagueCoordinator
from league.utils.team_composer import TeamComposer
from custom_logging.logger import MainLogger
from utils.main_utils import get_default_config, get_config, load_match_build_plan, recursive_dict_update, config_copy, \
    set_agents_only

import numpy as np
import torch as th
from types import SimpleNamespace

from utils.run_utils import args_sanity_check

SETTINGS['CAPTURE_MODE'] = "fd"  # set to "no" if you want to see stdout/stderr in console
logger = MainLogger.console_logger()

ex = Experiment("ma-league")
ex.logger = logger
ex.captured_out_filter = apply_backspaces_and_linefeeds

results_path = os.path.join(dirname(dirname(abspath(__file__))), "results")


def run(_run, _config, _log):
    _config = args_sanity_check(_config, _log)
    _config['play_mode'] = "self"
    set_agents_only(_config)

    torch.multiprocessing.set_start_method('spawn', force=True)

    args = SimpleNamespace(**_config)
    args.device = "cuda" if args.use_cuda else "cpu"

    logger = MainLogger(_log)
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
    teams = [team_composer.compose_unique_teams(team_size)[0]]

    # Shared objects
    manager = Manager()
    p_matrix = manager.dict()  # Payoff matrix
    players = manager.list()  # List of current players

    # Infrastructure
    runs = []  # All running processes representing an agent playing in the league
    payoff = Payoff(p_matrix=p_matrix, players=players)  # Hold results of each match

    # Create league
    # league = AlphaStarLeague(initial_agents=team_compositions, payoff=payoff)
    league = SimpleLeague(teams=teams, payoff=payoff)
    # Communication
    in_queues, out_queues = zip(*[(Queue(), Queue()) for _ in range(league.size)])

    # Synchronization across all league instances
    setup_barrier = Barrier(parties=league.size)

    # Start league instances
    for idx, (in_q, out_q) in enumerate(zip(in_queues, out_queues)):
        league_run = LeagueProcess(
            player_id=idx,
            players=players,
            queue=(in_q, out_q),
            args=args,
            logger=logger,
            barrier=setup_barrier
        )
        runs.append(league_run)
    [r.start() for r in runs]

    # Handle message communication within the league
    coordinator = LeagueCoordinator(logger=logger, players=players, queues=(in_queues, out_queues), payoff=payoff)
    coordinator.start()
    coordinator.join()

    # Wait for processes to finish
    [r.join() for r in runs]

    # Print win rates for all players
    league.print_payoff()

    # Clean up after finishing
    print("Exiting Main")
    manager.shutdown()
    print("Stopping all threads")
    for t in threading.enumerate():
        if t.name != "MainThread":
            print("Thread {} is alive! Is daemon: {}".format(t.name, t.daemon))
            t.join(timeout=1)
            print("Thread joined")

    print("Exiting script")

    # Making sure framework really exits
    os._exit(os.EX_OK)


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
