import time
import traceback
from typing import Dict

from torch.multiprocessing import Process, current_process
import os
import datetime
import numpy as np
from torch import manual_seed, device

from sacred import SETTINGS, Experiment
from sacred.observers import FileStorageObserver
from sacred.utils import apply_backspaces_and_linefeeds
from custom_logging.platforms import CustomConsoleLogger
from custom_logging.logger import MainLogger
from utils.main_utils import config_copy

from types import SimpleNamespace

SETTINGS['CAPTURE_MODE'] = "fd"  # set to "no" if you want to see stdout/stderr in console
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # Lower tf logging level
os.environ['PYGAME_HIDE_SUPPORT_PROMPT'] = "hide"  # Deactivate message from envs built pygame


class ExperimentInstance(Process):
    def __init__(self, idx: int, experiment_config: Dict):
        """
        Captures main.py functionality in a process to spawn multiple experiments.
        :param params:
        :param configs_dir:
        """
        super(ExperimentInstance, self).__init__()
        self.idx = idx
        self.experiment_config = experiment_config
        self._instance_log_dir = f'{experiment_config["log_dir"]}/instance_{self.idx}'

        self._logger: MainLogger = None
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

        ex.run()

        os._exit(os.EX_OK)

    def _build_sacred_experiment(self) -> Experiment:
        logger = CustomConsoleLogger(f"instance-{self.idx}")

        ex = Experiment(f"instance-{self.idx}")
        ex.logger = logger
        ex.captured_out_filter = apply_backspaces_and_linefeeds
        ex.add_config(self.experiment_config)

        # Save to disk by default for sacred
        logger.info("Saving to FileStorageObserver in results/sacred.")
        results_path = os.path.join(self._instance_log_dir)
        file_obs_path = os.path.join(results_path, "sacred")
        ex.observers.append(FileStorageObserver(file_obs_path))
        return ex

    def run_sacred_framework(self, _run, _config, _log):
        self._args = SimpleNamespace(**_config)
        self._args.device = device(_config["device"] if self._args.use_cuda else "cpu")
        self._args.log_dir = self._instance_log_dir
        self._args.env_args["record"] = self._instance_log_dir if self._args.env_args["record"] else ""

        self._setup_logger(_log, _run, self._args)

        try:
            self._run_experiment()
        except Exception as e:  # Interrupt should not be issued in _run_experiment
            self._logger.info(f"Experiment process ended due to error: {e}")
            traceback.print_exc()

    def _setup_logger(self, _log, _run, args):
        self._logger = MainLogger(_log, args)
        # configure tensorboard logger
        unique_token = "{}__{}".format(args.name, datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S"))
        args.unique_token = unique_token
        if args.use_tensorboard:
            tb_logs_direc = os.path.join(self._instance_log_dir, "tb_logs")
            tb_exp_direc = os.path.join(tb_logs_direc, "{}").format(unique_token)
            self._logger.setup_tensorboard(tb_exp_direc)
        # sacred is on by default
        self._logger.setup_sacred(_run)

    def _set_seed(self, _config):
        config = config_copy(_config)
        np.random.seed(config["seed"])
        manual_seed(config["seed"])
        config['env_args']['seed'] = config["seed"]
        return config


class EmptyInstance(ExperimentInstance):  # For demonstration only

    def _run_experiment(self):
        import torch
        from torch import nn

        epochs = 100
        lr = 0.01
        if torch.cuda.is_available() and self.experiment_config["use_cuda"]:
            dev = "cuda:0"
        else:
            dev = "cpu"

        x = torch.tensor([1, 2, 3, 4, 5], device=torch.device(dev), dtype=torch.float32).view(-1, 1)
        y_target = torch.tensor([2, 4, 9, 16, 25], device=torch.device(dev), dtype=torch.float32).view(-1, 1)
        linear_regression_model = nn.Linear(in_features=1, out_features=1, device=torch.device(dev))
        criterion = torch.nn.MSELoss()

        optimizer = torch.optim.SGD(linear_regression_model.parameters(), lr=lr)

        for epoch in range(epochs):
            y_pred = linear_regression_model(x)
            loss = criterion(y_pred, y_target)
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            loss = loss.item()
            self._logger.log_stat(key="loss", value=loss, t_env=epoch)
            if (epoch + 1) % 10 == 0:
                self._logger.info('Epoch: {} - Loss {}'.format(epoch, loss))
