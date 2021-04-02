import numpy as np
import os
from os.path import dirname, abspath
from copy import deepcopy

from sacred import Experiment, SETTINGS
from sacred.observers import FileStorageObserver
from sacred.utils import apply_backspaces_and_linefeeds
import sys
import torch as th

from utils.logging import LeagueLogger
from utils.main_utils import config_copy, get_config, recursive_dict_update, get_default_config, load_match_build_plan, \
    set_agents_only

SETTINGS['CAPTURE_MODE'] = "fd"  # set to "no" if you want to see stdout/stderr in console
logger = LeagueLogger.console_logger()

ex = Experiment("pymarl")
ex.logger = logger
ex.captured_out_filter = apply_backspaces_and_linefeeds

results_path = os.path.join(dirname(dirname(abspath(__file__))), "results")


@ex.main
def main(_run, _config, _log):
    # Setting the random seed throughout the modules
    config = config_copy(_config)
    np.random.seed(config["seed"])
    th.manual_seed(config["seed"])
    config['env_args']['seed'] = config["seed"]

    play_mode = config['play_mode']
    if play_mode != "normal":
        set_agents_only(config)

    if play_mode == "self":
        from self_play_run import run
    else:
        from run import run

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
    ex.observers.append(FileStorageObserver.create(file_obs_path))

    ex.run_commandline(params)
