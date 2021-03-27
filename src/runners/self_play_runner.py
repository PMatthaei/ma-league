from envs import REGISTRY as env_REGISTRY
from functools import partial
from components.episode_buffer import EpisodeBatch
import numpy as np
import torch as th

from utils.logging import update_stats


class SelfPlayRunner:

    def __init__(self, args, logger):
        """
        Runner to train two multi-agents (home and opponent) in the same environment.
        The runner steps the environment and creates two batches of episode data one per agent.
        The resulting batches are returned per run()-cycle and served to its corresponding learner and replay buffer.
        :param args:
        :param logger:
        """
        self.args = args
        self.logger = logger
        self.batch_size = self.args.batch_size_run
        assert self.batch_size == 1

        self.env = env_REGISTRY[self.args.env](**self.args.env_args)
        self.episode_limit = self.env.episode_limit
        self.t = 0

        self.t_env = 0

        self.train_returns = {"home": [], "opponent": []}
        self.test_returns = {"home": [], "opponent": []}
        self.train_stats = {}
        self.test_stats = {}

        # Log the first run
        self.log_train_stats_t = -1000000

        self.new_batch = None
        self.home_mac = None
        self.opponent_mac = None
        self.home_batch = None
        self.opponent_batch = None

    def setup(self, scheme, groups, preprocess, home_mac, opponent_mac):
        self.new_batch = partial(EpisodeBatch, scheme, groups, self.batch_size, self.episode_limit + 1,
                                 preprocess=preprocess, device=self.args.device)
        self.home_mac = home_mac
        self.opponent_mac = opponent_mac

    def get_env_info(self):
        return self.env.get_env_info()

    def save_replay(self):
        raise NotImplementedError()

    def close_env(self):
        self.env.close()

    def reset(self):
        self.home_batch = self.new_batch()
        self.opponent_batch = self.new_batch()
        self.env.reset()
        self.t = 0

    def run(self, test_mode=False):
        self.reset()

        terminated = False
        home_episode_return = 0
        opp_episode_return = 0

        self.home_mac.init_hidden(batch_size=self.batch_size)
        self.opponent_mac.init_hidden(batch_size=self.batch_size)

        #
        #
        # Main Loop - Run while episode is not terminated
        #
        #
        while not terminated:
            home_pre_transition_data, opponent_pre_transition_data = self._build_pre_transition_data()

            self.home_batch.update(home_pre_transition_data, ts=self.t)
            self.opponent_batch.update(opponent_pre_transition_data, ts=self.t)

            # Pass the entire batch of experiences up till now to the agents
            # Receive the actions for each agent at this timestep in a batch of size 1
            home_actions = self.home_mac.select_actions(self.home_batch, t_ep=self.t, t_env=self.t_env,
                                                        test_mode=test_mode)
            opponent_actions = self.opponent_mac.select_actions(self.opponent_batch, t_ep=self.t, t_env=self.t_env,
                                                                test_mode=test_mode)

            all_actions = th.cat((home_actions[0], opponent_actions[0]))
            #
            # Environment Step
            #
            obs, reward, done_n, env_info = self.env.step(all_actions)
            #
            #
            #
            self.env.render()

            assert len(reward) == 2, \
                "League runner expects to receive global rewards for home and opponent. " \
                "More or less than 2 rewards f.e. local rewards per agent are currently not supported."

            home_reward, opponent_reward = reward
            home_episode_return += home_reward
            opp_episode_return += opponent_reward

            home_post_transition_data = {
                "actions": home_actions,
                "reward": [(home_reward,)],
                "terminated": [(done_n[0],)],
            }
            opponent_post_transition_data = {
                "actions": opponent_actions,
                "reward": [(opponent_reward,)],
                "terminated": [(done_n[1],)],
            }
            # Termination is dependent on all team-wise terminations
            terminated = any(done_n)
            self.home_batch.update(home_post_transition_data, ts=self.t)
            self.opponent_batch.update(opponent_post_transition_data, ts=self.t)

            self.t += 1
        #
        #
        #
        #
        #

        #
        #
        # Last (state,action) transition pair per learner
        #
        #
        home_last_data, opponent_last_data = self._build_pre_transition_data()

        self.home_batch.update(home_last_data, ts=self.t)
        self.opponent_batch.update(opponent_last_data, ts=self.t)

        # Select actions in the last stored state
        actions = self.home_mac.select_actions(self.home_batch, t_ep=self.t, t_env=self.t_env,
                                               test_mode=test_mode)
        self.home_batch.update({"actions": actions}, ts=self.t)

        actions = self.home_mac.select_actions(self.opponent_batch, t_ep=self.t, t_env=self.t_env,
                                               test_mode=test_mode)
        self.opponent_batch.update({"actions": actions}, ts=self.t)

        #
        # Stats and Logging for two learners
        #
        cur_stats = self.test_stats if test_mode else self.train_stats
        cur_returns = self.test_returns if test_mode else self.train_returns
        log_prefix = "test_" if test_mode else ""

        cur_stats.update({k: update_stats(cur_stats, k, env_info) for k in set(cur_stats) | set(env_info)})
        cur_stats["n_episodes"] = 1 + cur_stats.get("n_episodes", 0)
        cur_stats["ep_length"] = self.t + cur_stats.get("ep_length", 0)

        if not test_mode:
            self.t_env += self.t

        cur_returns["home"].append(home_episode_return)
        cur_returns["opponent"].append(opp_episode_return)

        if test_mode and (len(self.test_returns) == self.args.test_nepisode):
            self._log_returns(cur_returns["home"], "home_" + log_prefix)
            self._log_returns(cur_returns["opponent"], "opponent_" + log_prefix)
            self._log_stats(cur_stats, log_prefix)
        elif self.t_env - self.log_train_stats_t >= self.args.runner_log_interval:
            self._log_returns(cur_returns["home"], "home_" + log_prefix)
            self._log_returns(cur_returns["opponent"], "opponent_" + log_prefix)
            self._log_stats(cur_stats, log_prefix)

            if hasattr(self.home_mac.action_selector, "epsilon"):
                self.logger.add_stat("home_epsilon", self.home_mac.action_selector.epsilon, self.t_env)
            if hasattr(self.opponent_mac.action_selector, "epsilon"):
                self.logger.add_stat("opponent_epsilon", self.opponent_mac.action_selector.epsilon, self.t_env)
            self.log_train_stats_t = self.t_env

        return self.home_batch, self.opponent_batch

    def _build_pre_transition_data(self):
        state = self.env.get_state()
        actions = self.env.get_avail_actions()
        obs = self.env.get_obs()
        # TODO: only supports same team sizes!
        home_avail_actions = actions[:len(actions) // 2]
        home_obs = obs[:len(obs) // 2]
        home_pre_transition_data = {
            "state": [state],
            "avail_actions": [home_avail_actions],
            "obs": [home_obs]
        }
        opponent_avail_actions = actions[len(actions) // 2:]
        opponent_obs = obs[len(obs) // 2:]
        opponent_pre_transition_data = {
            "state": [state],
            "avail_actions": [opponent_avail_actions],
            "obs": [opponent_obs]
        }
        return home_pre_transition_data, opponent_pre_transition_data

    def _log_returns(self, returns, prefix):
        self.logger.add_stat(prefix + "return_mean", np.mean(returns), self.t_env)
        self.logger.add_stat(prefix + "return_std", np.std(returns), self.t_env)
        returns.clear()

    def _log_stats(self, stats, prefix):
        for k, v in stats.items():
            if k == "battle_won":
                self.logger.add_stat("home_" + prefix + k + "_mean", v[0] / stats["n_episodes"], self.t_env)
                self.logger.add_stat("opponent_" + prefix + k + "_mean", v[1] / stats["n_episodes"], self.t_env)
            elif k != "n_episodes":
                self.logger.add_stat(prefix + k + "_mean", v / stats["n_episodes"], self.t_env)
        stats.clear()
