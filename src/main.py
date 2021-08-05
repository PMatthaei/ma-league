import os
import sys
import threading
import traceback

import torch as th
import datetime

from types import SimpleNamespace
from os.path import dirname, abspath
from copy import deepcopy
from sacred import SETTINGS
from league.processes.eval.replay_instance import ReplayInstance
from utils.config_builder import ConfigBuilder
from torch.multiprocessing import set_start_method

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # Lower tf logging level

SETTINGS['CAPTURE_MODE'] = "fd"  # set to "no" if you want to see stdout/stderr in console

set_start_method('spawn', force=True)

if __name__ == '__main__':
    params = deepcopy(sys.argv)
    available_cuda_devices = [f"cuda:{th.cuda.device(i).idx}" for i in range(th.cuda.device_count())]

    # Basics to start a experiment
    src_dir = f"{dirname(abspath(__file__))}"  # Path to src directory
    unique_token = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    log_dir = f'{dirname(dirname(abspath(__file__)))}/results/experiment_{unique_token}'  # Logs of the league

    config_builder = ConfigBuilder(
        worker_args=SimpleNamespace(**{"cuda_devices": available_cuda_devices, "balance_cuda_workload": False}),
        src_dir=src_dir,
        log_dir=log_dir,
        params=params
    )
    config = config_builder.build()

    r = ReplayInstance(idx=0, experiment_config=config)
    r.start()
    r.join()

    # Clean up after finishing
    print("Exiting Main")

    print("Stopping all threads")
    for t in threading.enumerate():
        if t.name != "MainThread":
            print("Thread {} is alive! Is daemon: {}".format(t.name, t.daemon))
            t.join(timeout=1)
            print("Thread joined")

    print("Exiting script")
# ex = build_sacred_experiment(config=config, log_dir=log_dir)
#
#
# @ex.main
# def main(_run, _config, _log):
#     config = _set_seed(_config)
#
#     # run the framework
#     run(_run, config, _log)
#
# ex.run_commandline(params)

# def run(_run, _config, _log):
#     # check args sanity
#     _config = args_sanity_check(_config, _log)
#
#     if _config['play_mode'] != "normal":
#         set_agents_only(_config)
#
#     args = SimpleNamespace(**_config)
#     args.device = "cuda" if args.use_cuda else "cpu"
#
#     # setup loggers
#     main_logger = MainLogger(_log, args)
#
#     # configure tensorboard logger
#     unique_token = "{}__{}".format(args.name, datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S"))
#     args.unique_token = unique_token
#     if args.use_tensorboard:
#         tb_logs_direc = os.path.join(dirname(dirname(abspath(__file__))), "results", "tb_logs")
#         tb_exp_direc = os.path.join(tb_logs_direc, "{}").format(unique_token)
#         main_logger.setup_tensorboard(tb_exp_direc)
#
#     # sacred is on by default
#     main_logger.setup_sacred(_run)
#
#     # Run and train
#     if _config['play_mode'] == "self":
#         eval_method = _config['eval']
#         if not eval_method:
#             from runs.train.sp_ma_experiment import SelfPlayMultiAgentExperiment
#             play = SelfPlayMultiAgentExperiment(args=args, logger=main_logger)
#         elif eval_method == "jpc":
#             from runs.evaluation.jpc_eval_run import JointPolicyCorrelationEvaluationRun
#             play = JointPolicyCorrelationEvaluationRun(args=args, logger=main_logger)
#         elif eval_method == "replay":
#             from runs.evaluation.replay_eval_run import ReplayGenerationRun
#             play = ReplayGenerationRun(args=args, logger=main_logger)
#         else:
#             from runs.train.sp_ma_experiment import SelfPlayMultiAgentExperiment
#             play = SelfPlayMultiAgentExperiment(args=args, logger=main_logger)
#     else:
#         from runs.train.ma_experiment import MultiAgentExperiment
#         play = MultiAgentExperiment(args=args, logger=main_logger)
#
#     play.start()
#
#     # Clean up after finishing
#     print("Exiting Main")
#
#     print("Stopping all threads")
#     for t in threading.enumerate():
#         if t.name != "MainThread":
#             print("Thread {} is alive! Is daemon: {}".format(t.name, t.daemon))
#             t.join(timeout=1)
#             print("Thread joined")
#
#     print("Exiting script")
#
#     # Making sure framework really exits
#     os._exit(os.EX_OK)
#
#
# @ex.main
# def main(_run, _config, _log):
#     # Setting the random seed throughout the modules
#     config = config_copy(_config)
#     np.random.seed(config["seed"])
#     th.manual_seed(config["seed"])
#     config['env_args']['seed'] = config["seed"]
#
#     # run the framework
#     run(_run, config, _log)
#
#
# if __name__ == '__main__':
#     params = deepcopy(sys.argv)
#     # Get the defaults from default.yaml
#     main_path = os.path.dirname(__file__)
#     config_dict = get_default_config(main_path)
#
#     # Load algorithm and env base configs
#     env_config = get_config(params, "--env-config", "envs", path=main_path)
#
#     # Load build plan if configured
#     env_args = env_config['env_args']
#     if "match_build_plan" in env_args:
#         env_args["match_build_plan"] = get_match_build_plan(main_path, env_args)
#
#     alg_config = get_config(params, "--config", "algs", path=main_path)
#     config_dict = recursive_dict_update(config_dict, env_config)
#     config_dict = recursive_dict_update(config_dict, alg_config)
#
#     # now add all the config to sacred
#     ex.add_config(config_dict)
#
#     # Save to disk by default for sacred
#     logger.info("Saving to FileStorageObserver in results/sacred.")
#     file_obs_path = os.path.join(results_path, "sacred")
#     ex.observers.append(FileStorageObserver(file_obs_path))
#
#     ex.run_commandline(params)
