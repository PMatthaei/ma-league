import torch as th

from custom_logging.collectibles import Collectibles
from steppers import EpisodeStepper
from custom_logging.logger import Originator
from steppers.utils.stepper_utils import build_pre_transition_data


class SelfPlayStepper(EpisodeStepper):

    def __init__(self, args, logger):
        """
        Stepper which is passing actions of two policies (home and opponent) into the same environment instead of
        hading one teams action selection over to an scripted AI.
        The stepper steps the environment and creates two batches of episode data one per policy.
        The resulting batches as well as the final env_info are returned per run()-cycle and can be
        served to its corresponding learner and replay buffer. The Self-Play Stepper returns both batches although
        in Self-Play Training only the "home" multi-agent should learn to prevent a non-stationary environment in case
         the "away" batch is used for learning and updating the underlying "away" Multi-Agent.

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
        home_actions_taken = []
        away_actions_taken = []
        while not terminated:
            home_pre_transition_data, away_pre_transition_data = build_pre_transition_data(self.env)

            self.home_batch.update(home_pre_transition_data, ts=self.t)
            self.away_batch.update(away_pre_transition_data, ts=self.t)

            home_actions, h_is_greedy = self.home_mac.select_actions(self.home_batch, t_ep=self.t, t_env=self.t_env,
                                                        test_mode=test_mode)
            away_actions, a_is_greedy = self.away_mac.select_actions(self.away_batch, t_ep=self.t, t_env=self.t_env,
                                                        test_mode=test_mode)

            home_actions_taken.append(th.stack([home_actions, h_is_greedy]))
            away_actions_taken.append(th.stack([away_actions, a_is_greedy]))

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

        home_last_data, away_last_data = build_pre_transition_data(self.env)

        self.home_batch.update(home_last_data, ts=self.t)
        self.away_batch.update(away_last_data, ts=self.t)

        # Select actions in the last stored state
        home_actions, h_is_greedy = self.home_mac.select_actions(self.home_batch, t_ep=self.t, t_env=self.t_env, test_mode=test_mode)
        self.home_batch.update({"actions": home_actions}, ts=self.t)

        away_actions, a_is_greedy = self.away_mac.select_actions(self.away_batch, t_ep=self.t, t_env=self.t_env, test_mode=test_mode)
        self.away_batch.update({"actions": away_actions}, ts=self.t)

        home_actions_taken.append(th.stack([home_actions, h_is_greedy]))
        away_actions_taken.append(th.stack([away_actions, a_is_greedy]))

        if not test_mode:
            self.t_env += self.t

        home_actions_taken = th.squeeze(th.stack(home_actions_taken))
        away_actions_taken = th.squeeze(th.stack(away_actions_taken))

        #
        # Stats and Logging for two learners
        #
        # Send data collected during the episode - this data needs further processing
        self.logger.collect(Collectibles.RETURN, home_episode_return, origin=Originator.HOME)
        self.logger.collect(Collectibles.RETURN, away_episode_return, origin=Originator.AWAY)
        self.logger.collect(Collectibles.ACTIONS_TAKEN, home_actions_taken, origin=Originator.HOME)
        self.logger.collect(Collectibles.ACTIONS_TAKEN, away_actions_taken, origin=Originator.AWAY)
        self.logger.collect(Collectibles.WON, env_info["battle_won"][0], origin=Originator.HOME)
        self.logger.collect(Collectibles.WON, env_info["battle_won"][1], origin=Originator.AWAY)
        self.logger.collect(Collectibles.DRAW, env_info["draw"])
        self.logger.collect(Collectibles.EPISODE, self.t)
        # Log epsilon from mac directly
        self.logger.log_stat("home_epsilon", self.epsilons[0], self.t)
        self.logger.log_stat("away_epsilon", self.epsilons[1], self.t)
        # Log collectibles if conditions suffice
        self.logger.log(self.t_env)

        return self.home_batch, self.away_batch, env_info