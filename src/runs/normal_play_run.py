import time
from types import SimpleNamespace
from typing import Dict

import torch as th

from runs.experiment_run import ExperimentRun
from utils.checkpoint_manager import CheckpointManager
from utils.timehelper import time_left, time_str

from learners import REGISTRY as le_REGISTRY
from controllers import REGISTRY as mac_REGISTRY
from components.episode_buffer import ReplayBuffer
from components.transforms import OneHot
from steppers import REGISTRY as stepper_REGISTRY


class NormalPlayRun(ExperimentRun):

    def __init__(self, args, logger, on_episode_end=None):
        """
        NormalPlay performs the standard way of training a single multi-agent against a static opponent.
        :param args:
        :param logger:
        """
        super().__init__(args, logger)
        self.last_test_T = -self.args.test_interval - 1
        self.last_log_T = 0
        self.model_save_time = 0
        self.learners = []
        self.start_time = time.time()
        self.last_time = self.start_time
        self.episode_callback = on_episode_end

        # Init stepper so we can get env info
        self._build_stepper()

        # Get env info from stepper
        self.env_info = self.stepper.get_env_info()

        # Retrieve important data from the env and set in args
        shapes = self._update_shapes()

        self.logger.update_shapes(shapes)

        # Default/Base scheme- call AFTER extracting env info
        self.groups, self.preprocess, self.scheme = self._build_schemes()

        self._build_learners()

        # Activate CUDA mode if supported
        if self.args.use_cuda:
            [learner.cuda() for learner in self.learners]

        self.checkpoint_manager = CheckpointManager(args=self.args, logger=self.logger)

    def _build_learners(self):
        # Buffers
        self.home_buffer = ReplayBuffer(self.scheme, self.groups, self.args.buffer_size,
                                        self.env_info["episode_limit"] + 1,
                                        preprocess=self.preprocess,
                                        device="cpu" if self.args.buffer_cpu_only else self.args.device)
        # Setup multi-agent controller here
        self.home_mac = mac_REGISTRY[self.args.mac](self.home_buffer.scheme, self.groups, self.args)
        # Learners
        self.home_learner = le_REGISTRY[self.args.learner](self.home_mac, self.home_buffer.scheme, self.logger,
                                                           self.args,
                                                           name="home")
        # Register in list of learners
        self.learners.append(self.home_learner)

    def _update_shapes(self) -> Dict:
        shapes = {
            "n_agents": int(self.env_info["n_agents"]),
            "n_actions": int(self.env_info["n_actions"]),
            "state_shape": int(self.env_info["state_shape"])
        }
        self.args = SimpleNamespace(**{**vars(self.args), **shapes})
        return shapes

    def _build_schemes(self):
        scheme = {
            "state": {"vshape": self.env_info["state_shape"]},
            "obs": {"vshape": self.env_info["obs_shape"], "group": "agents"},
            "actions": {"vshape": (1,), "group": "agents", "dtype": th.long},
            "avail_actions": {"vshape": (self.env_info["n_actions"],), "group": "agents", "dtype": th.int},
            "reward": {"vshape": (1,)},
            "terminated": {"vshape": (1,), "dtype": th.uint8},
        }
        groups = {
            "agents": self.args.n_agents
        }
        preprocess = {
            "actions": ("actions_onehot", [OneHot(out_dim=self.args.n_actions)])
        }
        return groups, preprocess, scheme

    def _build_stepper(self):
        # Give runner the scheme
        self.stepper = stepper_REGISTRY[self.args.runner](args=self.args, logger=self.logger)

    def _init_stepper(self):
        self.stepper.initialize(scheme=self.scheme, groups=self.groups, preprocess=self.preprocess,
                                home_mac=self.home_mac)

    @property
    def _has_not_reached_t_max(self):
        return self._play_time is None and (self.stepper.t_env <= self.args.t_max)

    @property
    def _has_not_reached_time_limit(self):
        return self._play_time is not None and ((self._end_time - self._start_time) <= self._play_time)

    def start(self, play_time=None, train_callback=None):
        """

        :param play_time: Play the run for a certain time in seconds.
        :return:
        """
        self._play_time = play_time
        self._init_stepper()

        if self.args.checkpoint_path != "":
            self.load_learners()

            if self.args.evaluate or self.args.save_replay:
                self._evaluate_sequential()
                return

        # start training
        episode = 0
        if play_time:
            self.logger.info("Beginning training for {} seconds.".format(play_time))
        else:
            self.logger.info("Beginning training for {} timesteps.".format(self.args.t_max))

        self._start_time = time.time()
        self._end_time = time.time()

        while self._has_not_reached_time_limit or self._has_not_reached_t_max:

            # Run for a whole episode at a time
            self._train_episode(episode_num=episode, after_train=train_callback)

            # Execute test runs once in a while
            n_test_runs = max(1, self.args.test_nepisode // self.stepper.batch_size)
            if (self.stepper.t_env - self.last_test_T) / self.args.test_interval >= 1.0:
                self.logger.info("t_env: {} / {}".format(self.stepper.t_env, self.args.t_max))
                self.logger.info("Estimated time left: {}. Time passed: {}".format(
                    time_left(self.last_time, self.last_test_T, self.stepper.t_env, self.args.t_max),
                    time_str(time.time() - self.start_time)))
                self.last_time = time.time()
                self._test(n_test_runs)

            # Save model if configured
            save_interval_reached = (self.stepper.t_env - self.model_save_time) >= self.args.save_model_interval
            if self.args.save_model and (save_interval_reached or self.model_save_time == 0):
                self.save_learners()

            # Update episode counter with number of episodes run in the batch
            episode += self.args.batch_size_run

            # Log metrics and learner stats once in a while
            if (self.stepper.t_env - self.last_log_T) >= self.args.log_interval:
                self.logger.log_stat("episode", episode, self.stepper.t_env)
                self.logger.log_console()
                self.last_log_T = self.stepper.t_env

            self._end_time = time.time()
        # Finish and clean up
        self._finish()

    def load_learners(self, checkpoint_path=None):
        timestep_to_load = self.checkpoint_manager.load(learners=self.learners, load_step=self.args.load_step)
        self.stepper.t_env = timestep_to_load

    def save_learners(self, identifier=None):
        self.model_save_time = self.stepper.t_env
        out_path = self.checkpoint_manager.save(learners=self.learners, t_env=self.model_save_time,
                                                identifier=identifier)
        self.logger.info("Saving models to {}".format(out_path))
        return out_path

    def _finish(self):
        self.stepper.close_env()
        self.logger.info("Finished.")

    def _train_episode(self, episode_num, after_train=None):
        episode_batch, env_info = self.stepper.run(test_mode=False)
        if self.episode_callback is not None:
            self.episode_callback(env_info)

        self.home_buffer.insert_episode_batch(episode_batch)

        if self.home_buffer.can_sample(self.args.batch_size):
            episode_sample = self.home_buffer.sample(self.args.batch_size)

            # Truncate batch to only filled timesteps
            max_ep_t = episode_sample.max_t_filled()
            episode_sample = episode_sample[:, :max_ep_t]

            if episode_sample.device != self.args.device:
                episode_sample.to(self.args.device)

            self.home_learner.train(episode_sample, self.stepper.t_env, episode_num)

            if after_train:
                after_train(self.learners)

    def _test(self, n_test_runs):
        self.last_test_T = self.stepper.t_env
        for _ in range(n_test_runs):
            self.stepper.run(test_mode=True)

    def _evaluate_sequential(self):
        self.logger.info("Evaluate for {} steps.".format(self.args.test_nepisode))
        for _ in range(self.args.test_nepisode):
            self.stepper.run(test_mode=True)

        if self.args.save_replay:
            self.stepper.save_replay()

        self.stepper.close_env()
