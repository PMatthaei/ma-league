from runs.run import NormalPlayRun

from learners import REGISTRY as le_REGISTRY
from controllers import REGISTRY as mac_REGISTRY
from components.episode_buffer import ReplayBuffer
from steppers import SELF_REGISTRY as self_steppers_REGISTRY


class SelfPlayRun(NormalPlayRun):

    def __init__(self, args, logger, finish_callback=None, episode_callback=None):
        super().__init__(args, logger)
        self.finish_callback = finish_callback
        self.episode_callback = episode_callback

    def _set_scheme_meta(self):
        super()._set_scheme_meta()
        # Override number of agents with per agent value
        self.args.n_agents = int(self.env_info["n_agents"] / 2)  # TODO: assuming same team size

    def _build_learners(self):
        super()._build_learners()
        self.away_buffer = ReplayBuffer(self.scheme, self.groups, self.args.buffer_size,
                                        self.env_info["episode_limit"] + 1,
                                        preprocess=self.preprocess,
                                        device="cpu" if self.args.buffer_cpu_only else self.args.device)
        # Setup multi-agent controller here
        self.away_mac = mac_REGISTRY[self.args.mac](self.away_buffer.scheme, self.groups, self.args)
        # Learners
        self.away_learner = le_REGISTRY[self.args.learner](self.away_mac, self.scheme, self.logger, self.args,
                                                           name="away")
        self.learners.append(self.away_learner)

    def _build_stepper(self):
        self.stepper = self_steppers_REGISTRY[self.args.runner](args=self.args, logger=self.logger)

    def _init_stepper(self):
        # Give runner the scheme
        self.stepper.initialize(scheme=self.scheme, groups=self.groups, preprocess=self.preprocess,
                                home_mac=self.home_mac,
                                opponent_mac=self.away_mac)

    def _finish(self):
        super()._finish()
        if self.finish_callback is not None:
            self.finish_callback()

    def _train_episode(self, episode_num):
        # Run for a whole episode at a time
        home_batch, opponent_batch, last_env_info = self.stepper.run(test_mode=False)
        if self.episode_callback is not None:
            self.episode_callback(last_env_info)

        self.home_buffer.insert_episode_batch(home_batch)
        self.away_buffer.insert_episode_batch(opponent_batch)

        # Sample batch from buffer if possible
        batch_size = self.args.batch_size
        if self.home_buffer.can_sample(batch_size) and self.away_buffer.can_sample(batch_size):
            home_sample = self.home_buffer.sample(batch_size)
            opponent_sample = self.away_buffer.sample(batch_size)

            # Truncate batch to only filled timesteps
            max_ep_t_h = home_sample.max_t_filled()
            max_ep_t_o = opponent_sample.max_t_filled()
            home_sample = home_sample[:, :max_ep_t_h]
            opponent_sample = opponent_sample[:, :max_ep_t_o]

            device = self.args.device
            if home_sample.device != device:
                home_sample.to(device)

            if opponent_sample.device != device:
                opponent_sample.to(device)

            self.home_learner.train(home_sample, self.stepper.t_env, episode_num)
            self.away_learner.train(opponent_sample, self.stepper.t_env, episode_num)
