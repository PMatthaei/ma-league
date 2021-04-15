from bin.controls.headless_controls import HeadlessControls

from envs import REGISTRY as env_REGISTRY
from functools import partial
from components.episode_buffer import EpisodeBatch
import torch as th

from utils.logging import Originator


class SelfPlayStepper:

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

        if self.args.headless_controls:
            controls = HeadlessControls(env=self.env)
            controls.daemon = True
            controls.start()

        self.episode_limit = self.env.episode_limit
        self.t = 0  # current time step within the episode

        self.t_env = 0  # total time steps for this runner in the provided environment across multiple episodes

        self.new_batch = None
        self.home_mac = None
        self.opponent_mac = None
        self.home_batch = None
        self.opponent_batch = None

    def initialize(self, scheme, groups, preprocess, home_mac, opponent_mac):
        self.new_batch = partial(EpisodeBatch, scheme, groups, self.batch_size, self.episode_limit + 1,
                                 preprocess=preprocess, device=self.args.device)
        self.home_mac = home_mac
        self.opponent_mac = opponent_mac

    @property
    def epsilons(self):
        return getattr(self.home_mac.action_selector, "epsilon", None), \
               getattr(self.opponent_mac.action_selector, "epsilon", None)

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

        self.logger.test_mode = test_mode
        self.logger.test_n_episode = self.args.test_nepisode
        self.logger.runner_log_interval = self.args.runner_log_interval

        self.home_mac.init_hidden(batch_size=self.batch_size)
        self.opponent_mac.init_hidden(batch_size=self.batch_size)

        env_info = {}

        #
        #
        # Main Loop - Run while episode is not terminated
        #
        #
        while not terminated:
            home_pre_transition_data, opponent_pre_transition_data = self._build_pre_transition_data()

            self.home_batch.update(home_pre_transition_data, ts=self.t)
            self.opponent_batch.update(opponent_pre_transition_data, ts=self.t)

            home_actions = self.home_mac.select_actions(self.home_batch, t_ep=self.t, t_env=self.t_env,
                                                        test_mode=test_mode)
            opponent_actions = self.opponent_mac.select_actions(self.opponent_batch, t_ep=self.t, t_env=self.t_env,
                                                                test_mode=test_mode)

            all_actions = th.cat((home_actions[0], opponent_actions[0]))
            obs, reward, done_n, env_info = self.env.step(all_actions)
            self.env.render()

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
        # Last (state,action) transition pair per learner
        #
        #
        home_last_data, opponent_last_data = self._build_pre_transition_data()

        self.home_batch.update(home_last_data, ts=self.t)
        self.opponent_batch.update(opponent_last_data, ts=self.t)

        # Select actions in the last stored state
        actions = self.home_mac.select_actions(self.home_batch, t_ep=self.t, t_env=self.t_env, test_mode=test_mode)
        self.home_batch.update({"actions": actions}, ts=self.t)

        actions = self.home_mac.select_actions(self.opponent_batch, t_ep=self.t, t_env=self.t_env, test_mode=test_mode)
        self.opponent_batch.update({"actions": actions}, ts=self.t)

        if not test_mode:
            self.t_env += self.t

        #
        # Stats and Logging for two learners
        #
        self.logger.collect_episode_returns(home_episode_return, org=Originator.HOME)
        self.logger.collect_episode_returns(opp_episode_return, org=Originator.AWAY)
        self.logger.collect_episode_stats(env_info, self.t)
        self.logger.add_stats(self.t_env, epsilons=self.epsilons)

        return self.home_batch, self.opponent_batch, env_info

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
