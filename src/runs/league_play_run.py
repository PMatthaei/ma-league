from learners.learner import Learner
from runs.normal_play_run import NormalPlayRun

from learners import REGISTRY as le_REGISTRY
from controllers import REGISTRY as mac_REGISTRY
from components.episode_buffer import ReplayBuffer
from steppers import SELF_REGISTRY as self_steppers_REGISTRY
import torch as th


class LeaguePlayRun(NormalPlayRun):

    def __init__(self, args, logger, finish_callback=None, episode_callback=None):
        """
        LeaguePlay performs training of single multi-agent against a checkpointed agent in the same environment,
        thus causing inter-non-stationarity between two agents since the opposite agent becomes part of the environment.
        :param args:
        :param logger:
        :param finish_callback:
        :param episode_callback:
        """
        super().__init__(args, logger)
        self.finish_callback = finish_callback
        self.episode_callback = episode_callback
        self.away_mac = None
        self.away_learner = None

    def set_away_learner(self, away: Learner):
        self.away_learner = away
        self.away_mac = away.mac
        self.away_learner.name = "away"
        self.learners.append(self.away_learner)

    def _set_scheme_meta(self):
        super()._set_scheme_meta()
        # Override number of agents with per agent value
        self.args.n_agents = int(self.env_info["n_agents"] / 2)  # TODO: assuming same team size and two teams

    def _build_stepper(self):
        self.stepper = self_steppers_REGISTRY[self.args.runner](args=self.args, logger=self.logger)

    def _init_stepper(self):
        # Give runner the scheme
        self.stepper.initialize(scheme=self.scheme, groups=self.groups, preprocess=self.preprocess,
                                home_mac=self.home_mac, away_mac=self.away_mac)

    def _finish(self):
        super()._finish()
        if self.finish_callback is not None:
            self.finish_callback()

    def _train_episode(self, episode_num, callback=None):
        # Run for a whole episode at a time
        home_batch, _, last_env_info = self.stepper.run(test_mode=False)
        if self.episode_callback is not None:
            self.episode_callback(last_env_info)

        self.home_buffer.insert_episode_batch(home_batch)

        # Sample batch from buffer if possible
        batch_size = self.args.batch_size
        if self.home_buffer.can_sample(batch_size):
            home_sample = self.home_buffer.sample(batch_size)

            # Truncate batch to only filled timesteps
            max_ep_t_h = home_sample.max_t_filled()
            home_sample = home_sample[:, :max_ep_t_h]

            device = self.args.device
            if home_sample.device != device:
                home_sample.to(device)

            self.home_learner.train(home_sample, self.stepper.t_env, episode_num)

            if callback:
                callback(self.learners)

    def _test(self, n_test_runs):
        self.last_test_T = self.stepper.t_env
        pass  # Skip tests in league
