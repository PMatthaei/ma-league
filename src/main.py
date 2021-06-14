import datetime
import pprint
import threading
from types import SimpleNamespace
import torch

import numpy as np
import os
from os.path import dirname, abspath
from copy import deepcopy

from sacred import Experiment, SETTINGS
from sacred.observers import FileStorageObserver
from sacred.utils import apply_backspaces_and_linefeeds
import sys
import torch as th

from custom_logging.logger import MainLogger
from custom_logging.platforms import CustomConsoleLogger
from utils.main_utils import config_copy, get_config, recursive_dict_update, get_default_config, load_match_build_plan, \
    set_agents_only
from utils.run_utils import args_sanity_check

SETTINGS['CAPTURE_MODE'] = "fd"  # set to "no" if you want to see stdout/stderr in console
logger = CustomConsoleLogger.console_logger()

ex = Experiment("pymarl")
ex.logger = logger
ex.captured_out_filter = apply_backspaces_and_linefeeds

results_path = os.path.join(dirname(dirname(abspath(__file__))), "results")

def run(_run, _config, _log):
    # check args sanity
    _config = args_sanity_check(_config, _log)

    if _config['play_mode'] != "normal":
        set_agents_only(_config)

    if _config["runner"] == "parallel" or _config['eval'] == "jpc":
        torch.multiprocessing.set_start_method('spawn', force=True)
        #multiprocessing.set_start_method('spawn')

    args = SimpleNamespace(**_config)
    args.device = "cuda" if args.use_cuda else "cpu"

    # setup loggers
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

    # Run and train
    if _config['play_mode'] == "self":
        eval_method = _config['eval']
        if not eval_method:
            from runs.self_play_run import SelfPlayRun
            play = SelfPlayRun(args=args, logger=logger)
        elif eval_method == "jpc":
            from eval.jpc_eval_run import JointPolicyCorrelationEvaluationRun
            play = JointPolicyCorrelationEvaluationRun(args=args, logger=logger)
        else:
            from runs.self_play_run import SelfPlayRun
            play = SelfPlayRun(args=args, logger=logger)
    else:
        from runs.normal_play_run import NormalPlayRun
        play = NormalPlayRun(args=args, logger=logger)

    play.start()

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
def main(_run, _config, _log):
    # Setting the random seed throughout the modules
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
    ex.observers.append(FileStorageObserver(file_obs_path))

    ex.run_commandline(params)
