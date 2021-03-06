import os

import datetime
import sys
import threading
import numpy as np
import torch as th

from copy import deepcopy
from random import sample
from torch.multiprocessing import Barrier, Queue, Manager
from os.path import dirname, abspath
from maenv.core import RoleTypes, UnitAttackTypes
from sacred import SETTINGS, Experiment
from sacred.observers import FileStorageObserver
from sacred.utils import apply_backspaces_and_linefeeds
from custom_logging.platforms import CustomConsoleLogger
from league import SimpleLeague
from league.components.payoff_role_based import RolebasedPayoff
from league.processes.role_based_league_process import RolebasedLeagueProcess
from league.processes.league_coordinator import LeagueCoordinator
from league.utils.team_composer import TeamComposer
from custom_logging.logger import MainLogger
from utils.main_utils import get_default_config, get_config, load_match_build_plan, recursive_dict_update, config_copy, \
    set_agents_only

from types import SimpleNamespace
from utils.run_utils import args_sanity_check

th.multiprocessing.set_start_method('spawn', force=True)

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # Lower tf logging level
SETTINGS['CAPTURE_MODE'] = "fd"  # set to "no" if you want to see stdout/stderr in console
logger = CustomConsoleLogger.console_logger()

ex = Experiment("ma-league")
ex.logger = logger
ex.captured_out_filter = apply_backspaces_and_linefeeds

results_path = os.path.join(dirname(dirname(abspath(__file__))), "results")


def run(_run, _config, _log):
    _config = args_sanity_check(_config, _log)
    _config['play_mode'] = "self"
    set_agents_only(_config)

    args = SimpleNamespace(**_config)
    args.device = "cuda" if args.use_cuda else "cpu"

    main_logger = MainLogger(_log, args)

    # configure tensorboard logger
    unique_token = "{}__{}".format(args.name, datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S"))
    args.unique_token = unique_token
    if args.use_tensorboard:
        tb_logs_direc = os.path.join(dirname(dirname(abspath(__file__))), "results", "tb_logs")
        tb_exp_direc = os.path.join(tb_logs_direc, "{}").format(unique_token)
        main_logger.setup_tensorboard(tb_exp_direc)

    # sacred is on by default
    main_logger.setup_sacred(_run)

    # Build league teams
    team_size = _config["team_size"]
    team_composer = TeamComposer(RoleTypes, UnitAttackTypes)
    teams = team_composer.compose_unique_teams(team_size)
    teams = sample(teams, 2)  # Sample 5 random teams to train

    # Shared objects
    manager = Manager()
    payoff_dict = manager.dict()  # Payoff dict
    players = manager.list()  # List of current players

    # Infrastructure
    procs = []  # All running processes representing an agent playing in the league
    payoff = RolebasedPayoff(payoff_dict=payoff_dict, players=players)  # Hold results of each match

    # league = AlphaStarLeague(initial_agents=team_compositions, payoff=payoff)
    league = SimpleLeague(teams=teams, payoff=payoff)

    # Communication
    in_queues, out_queues = zip(*[(Queue(), Queue()) for _ in range(league.size)])

    # Synchronization across all league instances
    sync_barrier = Barrier(parties=league.size)

    # Start league instances
    for idx, (in_q, out_q) in enumerate(zip(in_queues, out_queues)):
        proc = RolebasedLeagueProcess(
            player_id=idx,
            players=players,
            queue=(in_q, out_q),
            args=args,
            logger=main_logger,
            sync_barrier=sync_barrier
        )
        procs.append(proc)

    [r.start() for r in procs]

    # Handle message communication within the league
    coordinator = LeagueCoordinator(
        logger=main_logger,
        players=players,
        queues=(in_queues, out_queues),
        payoff=payoff,
        sync_barrier=sync_barrier
    )
    coordinator.start()

    # Wait for processes to finish
    coordinator.join()
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
