import argparse
from typing import Dict

from torch.multiprocessing import Process, current_process
import os
import datetime
import numpy as np
import torch as th

from sacred import SETTINGS, Experiment
from sacred.observers import FileStorageObserver
from sacred.utils import apply_backspaces_and_linefeeds
from custom_logging.platforms import CustomConsoleLogger
from custom_logging.logger import MainLogger
from utils.main_utils import get_default_config, get_config, get_match_build_plan, recursive_dict_update, config_copy

from types import SimpleNamespace

from utils.run_utils import args_sanity_check

SETTINGS['CAPTURE_MODE'] = "fd"  # set to "no" if you want to see stdout/stderr in console


class ExperimentProcess(Process):
    def __init__(self, params, src_dir: str):
        """
        Captures main.py functionality in a process to spawn multiple experiments.
        :param params:
        :param src_dir:
        """
        super(ExperimentProcess, self).__init__()
        self.params = params
        self.src_dir = src_dir
        self._logger = None
        self._args = None
        self._proc_id = None

    def _run_experiment(self):
        raise NotImplementedError("Please build the experiment")

    def run(self) -> None:
        self._proc_id = current_process()

        ex = self._build_sacred_experiment()

        @ex.automain
        def main(_run, _config, _log):
            # Load config and logger
            config = self._set_seed(_config)

            # run the framework
            self.run_sacred_framework(_run, config, _log)

        additional_config = self._parse_additional_config(self.experiment_config)
        ex.run(config_updates=additional_config)

        os._exit(os.EX_OK)

    def run_sacred_framework(self, _run, _config, _log):
        self._args = SimpleNamespace(**_config)

        self._setup_logger(_log, _run, self._args)

        self._run_experiment()

    def _setup_logger(self, _log, _run, args):
        self._logger = MainLogger(_log, args)
        # configure tensorboard logger
        unique_token = "{}__{}".format(args.name, datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S"))
        args.unique_token = unique_token
        if args.use_tensorboard:
            tb_logs_direc = os.path.join(self.src_dir, "results", "tb_logs")
            tb_exp_direc = os.path.join(tb_logs_direc, "{}").format(unique_token)
            self._logger.setup_tensorboard(tb_exp_direc)
        # sacred is on by default
        self._logger.setup_sacred(_run)

    def _build_sacred_experiment(self) -> Experiment:
        logger = CustomConsoleLogger("ma-league")

        ex = Experiment("ma-league")
        ex.logger = logger
        ex.captured_out_filter = apply_backspaces_and_linefeeds
        self.experiment_config = self._build_experiment_config()
        ex.add_config(self.experiment_config)

        # Save to disk by default for sacred
        logger.info("Saving to FileStorageObserver in results/sacred.")
        results_path = os.path.join(self.src_dir, "results")
        file_obs_path = os.path.join(results_path, "sacred")
        ex.observers.append(FileStorageObserver(file_obs_path))
        return ex

    def _parse_additional_config(self, config: Dict) -> Dict:
        parser = argparse.ArgumentParser()
        for key, default in config.items():
            parser.add_argument(f'--{key}', type=type(default), default=default, required=False)
        args, _ = parser.parse_known_args(self.params)
        return vars(args)

    def _build_experiment_config(self):
        # Get the defaults from default.yaml
        config_dict = get_default_config(self.src_dir)

        # Load league base config
        league_config = get_config(self.params, "--league-config", "leagues", path=self.src_dir)

        # Load env base config
        env_config = get_config(self.params, "--env-config", "envs", path=self.src_dir)

        # Load build plan if configured
        env_args = env_config['env_args']
        if "match_build_plan" in env_args:
            env_args["match_build_plan"] = get_match_build_plan(self.src_dir, env_args)

        # Load algorithm base config
        alg_config = get_config(self.params, "--config", "algs", path=self.src_dir)

        # Integrate loaded dicts into main dict
        config_dict = recursive_dict_update(config_dict, league_config)
        config_dict = recursive_dict_update(config_dict, env_config)
        experiment_config = recursive_dict_update(config_dict, alg_config)
        experiment_config = args_sanity_check(experiment_config)
        experiment_config["device"] = "cuda" if experiment_config["use_cuda"] else "cpu"

        return experiment_config

    def _set_seed(self, _config):
        config = config_copy(_config)
        np.random.seed(config["seed"])
        th.manual_seed(config["seed"])
        config['env_args']['seed'] = config["seed"]
        return config
