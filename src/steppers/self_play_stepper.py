import torch as th

from steppers import EpisodeStepper
from custom_logging.logger import Originator


class SelfPlayStepper(EpisodeStepper):

    def __init__(self, args, logger):
        """
        Runner to train two multi-agents (home and opponent) in the same environment.
        The runner steps the environment and creates two batches of episode data one per agent.
        The resulting batches are returned per run()-cycle and served to its corresponding learner and replay buffer.

        :param args:
        :param logger:
        """
        super().__init__(args, logger)
        self.away_mac = None
        self.away_batch = None

    def initialize(self, scheme, groups, preprocess, home_mac, away_mac=None):
        super().initialize(scheme, groups, preprocess, home_mac)
        self.away_mac = away_mac

    @property
    def epsilons(self):
        return getattr(self.home_mac.action_selector, "epsilon", None), \
               getattr(self.away_mac.action_selector, "epsilon", None)

    def save_replay(self):
        raise NotImplementedError()

    def reset(self):
        super().reset()
        self.away_batch = self.new_batch_fn()

    def run(self, test_mode=False):
        """
        Run a single episode while performing self-play
        :param test_mode:
        :return:
        """
        self.reset()

        terminated = False
        home_episode_return = 0
        away_episode_return = 0

        self.logger.test_mode = test_mode
        self.logger.test_n_episode = self.args.test_nepisode
        self.logger.runner_log_interval = self.args.runner_log_interval

        self.home_mac.init_hidden(batch_size=self.batch_size)
        self.away_mac.init_hidden(batch_size=self.batch_size)

        env_info = {}

        while not terminated:
            home_pre_transition_data, away_pre_transition_data = self._build_pre_transition_data()

            self.home_batch.update(home_pre_transition_data, ts=self.t)
            self.away_batch.update(away_pre_transition_data, ts=self.t)

            home_actions, h_is_greedy = self.home_mac.select_actions(self.home_batch, t_ep=self.t, t_env=self.t_env,
                                                        test_mode=test_mode)
            away_actions, a_is_greedy = self.away_mac.select_actions(self.away_batch, t_ep=self.t, t_env=self.t_env,
                                                        test_mode=test_mode)

            all_actions = th.cat((home_actions[0], away_actions[0]))
            obs, reward, done_n, env_info = self.env.step(all_actions)
            terminated = any(done_n)  # Termination is dependent on all team-wise terminations

            self.env.render()

            home_reward, away_reward = reward
            home_episode_return += home_reward
            away_episode_return += away_reward

            home_post_transition_data = {
                "actions": home_actions,
                "reward": [(home_reward,)],
                "terminated": [(terminated,)],
            }
            away_post_transition_data = {
                "actions": away_actions,
                "reward": [(away_reward,)],
                "terminated": [(terminated,)]
            }

            self.home_batch.update(home_post_transition_data, ts=self.t)
            self.away_batch.update(away_post_transition_data, ts=self.t)

            self.t += 1

        home_last_data, away_last_data = self._build_pre_transition_data()

        self.home_batch.update(home_last_data, ts=self.t)
        self.away_batch.update(away_last_data, ts=self.t)

        # Select actions in the last stored state
        home_actions, h_is_greedy = self.home_mac.select_actions(self.home_batch, t_ep=self.t, t_env=self.t_env, test_mode=test_mode)
        self.home_batch.update({"actions": home_actions}, ts=self.t)

        away_actions, a_is_greedy = self.away_mac.select_actions(self.away_batch, t_ep=self.t, t_env=self.t_env, test_mode=test_mode)
        self.away_batch.update({"actions": away_actions}, ts=self.t)

        if not test_mode:
            self.t_env += self.t

        #
        # Stats and Logging for two learners
        #
        self.logger.collect_episode_returns(home_episode_return, org=Originator.HOME)
        self.logger.collect_episode_returns(away_episode_return, org=Originator.AWAY)
        self.logger.collect_episode_stats(env_info, self.t)
        self.logger.add_stats(self.t_env, epsilons=self.epsilons)

        return self.home_batch, self.away_batch, env_info

    def _build_pre_transition_data(self):
        state = self.env.get_state()
        actions = self.env.get_avail_actions()
        obs = self.env.get_obs()
        # TODO: only supports same team sizes! and only two teams
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
