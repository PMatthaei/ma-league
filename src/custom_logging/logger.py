from __future__ import annotations

from collections import defaultdict
from typing import Any

from custom_logging.collectibles import Collectibles
from custom_logging.platforms import CustomSacredLogger, CustomTensorboardLogger, CustomConsoleLogger
from custom_logging.utils.enums import Originator


def dd_list():
    return []


def dd_dict():
    return {}


def dd_collectible():
    return {"current": Any, "train": Any, "test": Any}


class MainLogger:
    def __init__(self, console):
        """
        Logger for multiple outputs. Supports logging to TensorBoard as well as to file and console via sacred.
        All data is collected, managed, processed and distributed to its respective visualization platform.
        Most of the logs are generated at certain intervals from the current episode to prevent logging overhead and
        lessen memory consumption.
        :param console:
        """
        self._console_logger = CustomConsoleLogger(console)
        self.console_logger = console
        self._tensorboard_logger: CustomTensorboardLogger = None
        self._sacred_logger: CustomSacredLogger = None

        self.stats = defaultdict(dd_list)

        self.episodal_stats = self._build_collectible_dict()

        self.test_mode = False
        self.test_n_episode = 0
        self.runner_log_interval = 0
        self.log_train_stats_t = -1000000  # Log first run

    def _build_collectible_dict(self):
        episodal_stats = defaultdict(dd_collectible)
        for collectible in Collectibles:
            for k in episodal_stats[collectible].keys():
                if collectible.is_global:
                    episodal_stats[collectible][k] = []
                else:
                    is_dict = collectible.collection_type is dict
                    episodal_stats[collectible][k] = defaultdict(dd_dict if is_dict else dd_list)
        return episodal_stats

    def log(self, t_env):
        """
        Log if either an interval condition matches or finished.
        :param t_env:
        :return:
        """
        test_finished = any(
            [len(rs) == self.test_n_episode for rs in self.episodal_stats[Collectibles.RETURN]["test"].values()])
        if self.test_mode and test_finished:  # Collect test data as long as test is running
            self._log_collectibles(t_env)  # ... then process and log collectibles
        elif t_env - self.log_train_stats_t >= self.runner_log_interval:  # Collect train data as defined via interval
            self._log_collectibles(t_env)  # ... then process and log collectibles
            self.log_train_stats_t = t_env

    def log_stat(self, key, value, t_env, log_type="scalar"):
        """
        Print a single value already preprocessed and loggable.
        :param log_type:
        :param key:
        :param value:
        :param t_env:
        :return:
        """
        self.stats[key].append((t_env, value))
        self._tensorboard_logger.log(key, value, t_env, log_type) if self._tensorboard_logger else None
        self._sacred_logger.log(key, value, t_env, log_type) if self._tensorboard_logger else None

    def _log_collectibles(self, t):
        """
        Print all collectibles at the given timestep. A collectible describes a collection of values which are collected
        during an episode and therefore need preprocessing before being logged as a single scalar.
        :param t:
        :return:
        """
        mode = "test" if self.test_mode else "train"
        prefix = "test_" if self.test_mode else ""
        for collectible in Collectibles:
            if collectible.is_global:
                processed_data = list(zip(collectible.keys, self.preprocess_collectible(collectible)))
                for k, v in processed_data:  # Log all data generated from the collected data
                    self.log_stat(f"{prefix}{k}", v, t, log_type=collectible.log_type)
                    self.episodal_stats[collectible][mode].clear()
            else:
                for i, origin in enumerate(Originator.list()):
                    processed_data = list(zip(collectible.keys, self.preprocess_collectible(collectible, origin)))
                    for k, v in processed_data:  # Log all data generated from the collected data
                        self.log_stat(f"{prefix}{origin}_{k}", v, t, log_type=collectible.log_type)
                    self.episodal_stats[collectible][mode][origin].clear()

    def collect(self, collectible: Collectibles, data, origin: Originator = Originator.HOME, parallel=False):
        """
        Collects a given stat for later logging and plotting at each episode end.
        :param collectible: A collectible type associated with the data f.e. episode return
        :param data: Actual episodal data
        :param origin: Player from which the data originated
        :param parallel: True if the data is collected in parallel from multiple environments
        :return: 
        """
        collectible_stat = self.episodal_stats[collectible]
        mode = "test" if self.test_mode else "train"

        if parallel and isinstance(data, list):
            # Collect lists in parallel via extend to keep dims uniform
            if collectible.is_global:
                collectible_stat[mode].extend(data)
            else:
                collectible_stat[mode][origin].extend(data)
        elif not parallel:
            if collectible.is_global:
                collectible_stat[mode].append(data)
            else:
                collectible_stat[mode][origin].append(data)

    def preprocess_collectible(self, collectible: Collectibles, origin: Originator = None):
        """
        Preprocess the collected data of an originator to produce logged values and metrics.
        :param collectible:
        :param origin:
        :return:
        """
        mode = "test" if self.test_mode else "train"

        if collectible.is_global:
            data = self.episodal_stats[collectible][mode]
        else:
            data = self.episodal_stats[collectible][mode][origin]
        processed = [func(data) for func in collectible.preprocessing]
        return processed

    def setup_tensorboard(self, log_dir):
        self._tensorboard_logger = CustomTensorboardLogger(log_dir)

    def setup_sacred(self, sacred_run_dict):
        self._sacred_logger = CustomSacredLogger(sacred_run_dict)

    def update_loggers(self, args):
        self._tensorboard_logger.n_actions = args.n_actions
        self._tensorboard_logger.n_agents = args.n_agents

    def log_console(self):
        self._console_logger.log(self.stats)
