from collections import defaultdict
import logging
import numpy as np


class LeagueLogger:
    @staticmethod
    def console_logger():
        logger = logging.getLogger('ma-league')
        logger.handlers = []
        ch = logging.StreamHandler()
        formatter = logging.Formatter('[%(levelname)s %(asctime)s] %(name)s %(message)s', '%H:%M:%S')
        ch.setFormatter(formatter)
        logger.addHandler(ch)
        logger.setLevel(logging.INFO)
        return logger

    def __init__(self, console_logger):
        """

        :param console_logger:
        """
        self.console_logger = console_logger

        self.use_tb = False
        self.use_sacred = False
        self.use_hdf = False

        self.stats = defaultdict(lambda: [])

        self.tb_logger = None
        self.sacred_info = None

        self.train_returns = []
        self.test_returns = []
        self.train_stats = {}
        self.test_stats = {}

        self.test_mode = False
        self.test_n_episode = 0
        self.runner_log_interval = 0

        self.log_train_stats_t = -1000000  # Log first run

        self.ep_returns = []
        self.ep_stats = {}

    def setup_tensorboard(self, dir):
        from tensorboard_logger import configure, log_value
        configure(dir)
        self.tb_logger = log_value
        self.use_tb = True

    def setup_sacred(self, sacred_run_dict):
        self.sacred_info = sacred_run_dict.info
        self.use_sacred = True

    def log_recent_stats(self):
        """
        Format collected stats and print into console.
        :return:
        """
        log_str = "Recent Stats | t_env: {:>10} | Episode: {:>8}\n".format(*self.stats["episode"][-1])
        i = 0
        for (k, v) in sorted(self.stats.items()):
            if k == "episode":
                continue
            i += 1
            window = 5 if k != "epsilon" else 1
            item = "{:.4f}".format(np.mean([x[1] for x in self.stats[k][-window:]]))
            log_str += "{:<25}{:>8}".format(k + ":", item)
            log_str += "\n" if i % 4 == 0 else "\t"
        self.console_logger.info(log_str)

    def add_stats(self, t_env, epsilon=None):
        """
        Add all episodal stats depending on the mode and logging intervals.
        :param t_env:
        :return:
        """
        log_prefix = "test_" if self.test_mode else ""
        if self.test_mode and (len(self.test_returns) == self.test_n_episode):
            self.add_episodal_stats(log_prefix, t_env)
        elif t_env - self.log_train_stats_t >= self.runner_log_interval:
            self.add_episodal_stats(log_prefix, t_env)
            if epsilon:
                self.add_stat("epsilon", epsilon, t_env)
            self.log_train_stats_t = t_env

    def add_episodal_stats(self, prefix, t_env):
        """
        Adds all episodal stats for later printing. Clear episodal data before reusing in next episode.
        :param prefix:
        :param t_env:
        :return:
        """
        self.add_stat(prefix + "return_mean", np.mean(self.ep_returns), t_env)
        self.add_stat(prefix + "return_std", np.std(self.ep_returns), t_env)
        self.ep_returns.clear()

        for k, v in self.ep_stats.items():
            if k == "battle_won":  # TODO 0 selects which team!?!
                self.add_stat(prefix + k + "_mean", v[0] / self.ep_stats["n_episodes"], t_env)
            elif k == "draw":
                self.add_stat(prefix + k + "_mean", v / self.ep_stats["n_episodes"], t_env)
            elif k != "n_episodes":
                self.add_stat(prefix + k + "_mean", v / self.ep_stats["n_episodes"], t_env)
        self.ep_stats.clear()

    def add_stat(self, key, value, t_env, to_sacred=True):
        """
        Adds a single stat into the dictionary and tracks it via sacred and/or tensorboard.
        :param key:
        :param value:
        :param t_env:
        :param to_sacred:
        :return:
        """
        self.stats[key].append((t_env, value))

        if self.use_tb:
            self.tb_logger(key, value, t_env)

        if self.use_sacred and to_sacred:
            if key in self.sacred_info:
                self.sacred_info["{}_T".format(key)].append(t_env)
                self.sacred_info[key].append(value)
            else:
                self.sacred_info["{}_T".format(key)] = [t_env]
                self.sacred_info[key] = [value]

    def collect_episode_returns(self, episode_return):
        """
        Collect episodal returns depending on the current mode.
        :param episode_return:
        :return:
        """
        self.ep_returns = self.test_returns if self.test_mode else self.train_returns
        self.ep_returns.append(episode_return)

    def collect_episode_stats(self, env_info, t):
        """
        Collect episodal training stats from the environment depending on the current mode.
        :param env_info:
        :param t:
        :return:
        """
        self.ep_stats = self.test_stats if self.test_mode else self.train_stats
        self.ep_stats.update({k: self.update_stats(k, env_info) for k in set(self.ep_stats) | set(env_info)})
        self.ep_stats["n_episodes"] = 1 + self.ep_stats.get("n_episodes", 0)
        self.ep_stats["ep_length"] = t + self.ep_stats.get("ep_length", 0)

    def update_stats(self, k, env_info):
        """
        Integrate environment information into stats dict depending on the incoming data type.
        :param k:
        :param env_info:
        :return:
        """
        if k in env_info:
            stat_type = type(env_info[k])
        elif k in self.ep_stats:
            stat_type = type(self.ep_stats[k])
        else:
            raise KeyError("Key not found in supplied env_info dict which is used to update the current stats.")

        if stat_type is int or stat_type is float or stat_type is bool:
            return self.ep_stats.get(k, 0) + env_info.get(k, 0)
        elif stat_type is list:
            return self.ep_stats.get(k, []) + env_info.get(k, [])