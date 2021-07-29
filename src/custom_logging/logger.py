from __future__ import annotations

from collections import defaultdict, Sized
from typing import Any
import numpy as np
from custom_logging.collectibles import Collectibles
from custom_logging.platforms import CustomSacredLogger, CustomTensorboardLogger
from custom_logging.platforms.console import CustomConsoleLogger
from custom_logging.utils.enums import Originator


def dd_list():
    return []


def dd_dict():
    return {}


def dd_collectible():
    return {"current": Any, "train": Any, "test": Any}


class MainLogger:
    def __init__(self, console_logger: CustomConsoleLogger, args):
        """
        Logger for multiple outputs. Supports logging to TensorBoard as well as to file and console via sacred.
        All data is collected, managed, processed and distributed to its respective visualization platform.
        Most of the logs are generated at certain intervals from the current episode to prevent logging overhead and
        lessen memory consumption.
        :param console:
        """
        self.args = args
        self._console_logger = console_logger
        self._tensorboard_logger: CustomTensorboardLogger = None
        self._sacred_logger: CustomSacredLogger = None

        self.stats = defaultdict(dd_list)

        self._build_collectible_episodal_stats_dict()

        self.test_mode = False
        self.test_n_episode = self.args.test_nepisode
        self.runner_log_interval = self.args.runner_log_interval
        self.log_train_stats_t = -1000000  # Log first run
        self.collected_train_episodes = 0
        self.collected_test_episodes = 0

    def _build_collectible_episodal_stats_dict(self):
        self.episodal_stats = defaultdict(dd_collectible)
        for collectible in Collectibles:
            for k in self.episodal_stats[collectible].keys():
                if collectible.is_global:
                    self.episodal_stats[collectible][k] = []
                else:
                    is_dict = collectible.collection_type is dict
                    self.episodal_stats[collectible][k] = defaultdict(dd_dict if is_dict else dd_list)

    def info(self, info_str: str):
        self._console_logger.info(info_str)

    def error(self, err: str):
        self._console_logger.error(err)

    def log(self, t_env):
        """
        Log if either an interval condition matches or finished.
        :param t_env:
        :return:
        """
        test_returns = self.episodal_stats[Collectibles.RETURN]["test"][Originator.HOME]
        test_finished = len(test_returns) == self.test_n_episode
        # Collect test data as long as test is running
        if self.test_mode and test_finished:
            self._log_collectibles(t_env)  # ... then process and log collectibles
        # Collect train data as defined via interval
        elif not self.test_mode and t_env - self.log_train_stats_t >= self.runner_log_interval:
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
        self._sacred_logger.log(key, value, t_env, log_type) if self._sacred_logger else None

    def _log_collectibles(self, t_env):
        """
        Print all collectibles at the given timestep. A collectible describes a collection of values which are collected
        during an episode and therefore need preprocessing before being logged as a single scalar.
        :param t_env:
        :return:
        """
        mode = "test" if self.test_mode else "train"
        prefix = "test_" if self.test_mode else ""
        for collectible in Collectibles:
            if collectible.is_global:
                processed_data = list(zip(collectible.keys, self.preprocess_collectible(collectible)))
                for k, v in processed_data:  # Log all data generated from the collected data via the preprocessing
                    self.log_stat(f"{prefix}{k}", v, t_env, log_type=collectible.log_type)
                    self.episodal_stats[collectible][mode].clear()
            else:
                for i, origin in enumerate(Originator.list()):
                    processed_data = list(zip(collectible.keys, self.preprocess_collectible(collectible, origin)))
                    for k, v in processed_data:
                        self.log_stat(f"{prefix}{origin}_{k}", v, t_env, log_type=collectible.log_type)
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
            if collectible.is_global:  # Global data does not have a specific actor as origin
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

        def filled(d):
            return isinstance(d, Sized) and len(d) > 0

        mode = "test" if self.test_mode else "train"

        if collectible.is_global:
            data = self.episodal_stats[collectible][mode]
        else:
            data = self.episodal_stats[collectible][mode][origin]
        processed = [func(data) if filled(data) else np.nan for func in collectible.preprocessing]
        return processed

    def setup_tensorboard(self, log_dir):
        self._tensorboard_logger = CustomTensorboardLogger(log_dir)

    def setup_sacred(self, sacred_run_dict):
        self._sacred_logger = CustomSacredLogger(sacred_run_dict)

    def log_report(self):
        self._console_logger.info(f"Logging stats of {len(self.stats['episode'])} episodes")
        self._console_logger.log_stats_report(self.stats)

    def update_scheme(self, scheme):
        # TODO: manage meta data for logging somewhere else
        if self._tensorboard_logger:
            self._tensorboard_logger.scheme = scheme
