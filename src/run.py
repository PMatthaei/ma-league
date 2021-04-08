import time

import torch as th

from utils.checkpoint_manager import CheckpointManager
from utils.timehelper import time_left, time_str

from learners import REGISTRY as le_REGISTRY
from steppers import EpisodeStepper
from controllers import REGISTRY as mac_REGISTRY
from components.episode_buffer import ReplayBuffer
from components.transforms import OneHot


class NormalPlayRun:

    def __init__(self, args, logger):
        self.args = args
        self.logger = logger

        # Init runner so we can get env info
        self.stepper: EpisodeStepper = EpisodeStepper(args=args, logger=logger)

        # Set up schemes and groups here
        env_info = self.stepper.get_env_info()
        self.args.n_agents = int(env_info["n_agents"])
        self.args.n_actions = env_info["n_actions"]
        self.args.state_shape = env_info["state_shape"]

        # Default/Base scheme- call AFTER extracting env info
        groups, preprocess, scheme = self._build_schemes()

        # Buffers
        self.buffer = ReplayBuffer(scheme, groups, self.args.buffer_size, env_info["episode_limit"] + 1,
                                   preprocess=preprocess,
                                   device="cpu" if self.args.buffer_cpu_only else self.args.device)

        # Setup multi-agent controller here
        self.mac = mac_REGISTRY[args.mac](self.buffer.scheme, groups, self.args)

        # Give runner the scheme
        self.stepper.initialize(scheme=scheme, groups=groups, preprocess=preprocess, mac=self.mac)

        # Learners
        self.learner = le_REGISTRY[self.args.learner](self.mac, self.buffer.scheme, logger, self.args, name="home")

        # Activate CUDA mode if supported
        if self.args.use_cuda:
            self.learner.cuda()

        self.learners = [self.learner]
        self.checkpoint_manager = CheckpointManager(args=self.args, logger=self.logger)

        self.last_test_T = -self.args.test_interval - 1
        self.last_log_T = 0
        self.model_save_time = 0

        self.start_time = time.time()
        self.last_time = self.start_time

    def _build_schemes(self):
        env_info = self.stepper.get_env_info()
        scheme = {
            "state": {"vshape": env_info["state_shape"]},
            "obs": {"vshape": env_info["obs_shape"], "group": "agents"},
            "actions": {"vshape": (1,), "group": "agents", "dtype": th.long},
            "avail_actions": {"vshape": (env_info["n_actions"],), "group": "agents", "dtype": th.int},
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

    def start(self):
        if self.args.checkpoint_path != "":
            timestep_to_load = self.checkpoint_manager.load(learners=self.learners)
            self.stepper.t_env = timestep_to_load

            if self.args.evaluate or self.args.save_replay:
                self.evaluate_sequential()
                return

        # start training
        episode = 0

        self.logger.console_logger.info("Beginning training for {} timesteps".format(self.args.t_max))

        while self.stepper.t_env <= self.args.t_max:

            # Run for a whole episode at a time
            self._train_episode(episode_num=episode)

            # Execute test runs once in a while
            n_test_runs = max(1, self.args.test_nepisode // self.stepper.batch_size)
            if (self.stepper.t_env - self.last_test_T) / self.args.test_interval >= 1.0:
                self._test(n_test_runs)

            # Save model if configured
            if self.args.save_model and (self.stepper.t_env - self.model_save_time >= self.args.save_model_interval or self.model_save_time == 0):
                self.model_save_time = self.stepper.t_env
                out_path = self.checkpoint_manager.save(self.model_save_time, learners=self.learners)
                self.logger.console_logger.info("Saving models to {}".format(out_path))

            # Update episode counter with number of episodes run in the batch
            episode += self.args.batch_size_run

            # Log metrics and learner stats once in a while
            if (self.stepper.t_env - self.last_log_T) >= self.args.log_interval:
                self.logger.add_stat("episode", episode, self.stepper.t_env)
                self.logger.log_recent_stats()
                self.last_log_T = self.stepper.t_env

        # Finish and clean up
        self.stepper.close_env()
        self.logger.console_logger.info("Finished Training")

    def _train_episode(self, episode_num):
        episode_batch = self.stepper.run(test_mode=False)
        self.buffer.insert_episode_batch(episode_batch)
        if self.buffer.can_sample(self.args.batch_size):
            episode_sample = self.buffer.sample(self.args.batch_size)

            # Truncate batch to only filled timesteps
            max_ep_t = episode_sample.max_t_filled()
            episode_sample = episode_sample[:, :max_ep_t]

            if episode_sample.device != self.args.device:
                episode_sample.to(self.args.device)

            self.learner.train(episode_sample, self.stepper.t_env, episode_num)

    def _test(self, n_test_runs):
        self.logger.console_logger.info("t_env: {} / {}".format(self.stepper.t_env, self.args.t_max))
        self.logger.console_logger.info("Estimated time left: {}. Time passed: {}".format(
            time_left(self.last_time, self.last_test_T, self.stepper.t_env, self.args.t_max),
            time_str(time.time() - self.start_time)))
        self.last_time = time.time()
        self.last_test_T = self.stepper.t_env
        for _ in range(n_test_runs):
            self.stepper.run(test_mode=True)

    def evaluate_sequential(self):
        self.logger.console_logger.info("Evaluate for {} steps.".format(self.args.test_nepisode))
        for _ in range(self.args.test_nepisode):
            self.stepper.run(test_mode=True)

        if self.args.save_replay:
            self.stepper.save_replay()

        self.stepper.close_env()
