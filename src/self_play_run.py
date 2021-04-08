import datetime
import os
import pprint
import time
import threading

import torch as th
from types import SimpleNamespace as SN

from runners.self_play_runner import SelfPlayRunner
from utils.checkpoint_manager import CheckpointManager
from utils.logging import LeagueLogger
from utils.run_utils import args_sanity_check
from utils.timehelper import time_left, time_str
from os.path import dirname, abspath

from learners import REGISTRY as le_REGISTRY
from controllers import REGISTRY as mac_REGISTRY
from components.episode_buffer import ReplayBuffer
from components.transforms import OneHot


def run(_run, _config, _log):
    # check args sanity
    _config = args_sanity_check(_config, _log)

    args = SN(**_config)
    args.device = "cuda" if args.use_cuda else "cpu"

    # setup loggers
    logger = LeagueLogger(_log)

    _log.info("Experiment Parameters:")
    experiment_params = pprint.pformat(_config,
                                       indent=4,
                                       width=1)
    _log.info("\n\n" + experiment_params + "\n")

    # configure tensorboard logger
    unique_token = "{}__{}".format(args.name, datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S"))
    args.unique_token = unique_token
    if args.use_tensorboard:
        tb_logs_direc = os.path.join(dirname(dirname(abspath(__file__))), "results", "tb_logs")
        tb_exp_direc = os.path.join(tb_logs_direc, "{}").format(unique_token)
        logger.setup_tensorboard(tb_exp_direc)

    # sacred is on by default
    logger.setup_sacred(_run)

    # Run and train
    selfplay = SelfPlayRun(args=args, logger=logger)
    selfplay.run()

    # Clean up after finishing
    print("Exiting Main")

    print("Stopping all threads")
    for t in threading.enumerate():
        if t.name != "MainThread":
            print("Thread {} is alive! Is daemon: {}".format(t.name, t.daemon))
            t.join(timeout=1)
            print("Thread joined")

    print("Exiting script")

    # Making sure framework really exits
    os._exit(os.EX_OK)


class SelfPlayRun:

    def __init__(self, args, logger):
        self.args = args
        self.logger = logger

        # Init runner so we can get env info
        self.runner = SelfPlayRunner(args=args, logger=logger)

        # Set up schemes and groups here
        env_info = self.runner.get_env_info()
        args.n_agents = int(env_info["n_agents"] / 2)  # TODO: assuming same team size
        args.n_actions = env_info["n_actions"]
        args.state_shape = env_info["state_shape"]

        # Default/Base scheme
        scheme = {
            "state": {"vshape": env_info["state_shape"]},
            "obs": {"vshape": env_info["obs_shape"], "group": "agents"},
            "actions": {"vshape": (1,), "group": "agents", "dtype": th.long},
            "avail_actions": {"vshape": (env_info["n_actions"],), "group": "agents", "dtype": th.int},
            "reward": {"vshape": (1,)},
            "terminated": {"vshape": (1,), "dtype": th.uint8},
        }
        groups = {
            "agents": args.n_agents
        }
        preprocess = {
            "actions": ("actions_onehot", [OneHot(out_dim=args.n_actions)])
        }

        # Buffers
        self.opponent_buffer = ReplayBuffer(scheme, groups, args.buffer_size, env_info["episode_limit"] + 1,
                                            preprocess=preprocess,
                                            device="cpu" if args.buffer_cpu_only else args.device)

        self.home_buffer = ReplayBuffer(scheme, groups, args.buffer_size, env_info["episode_limit"] + 1,
                                        preprocess=preprocess,
                                        device="cpu" if args.buffer_cpu_only else args.device)

        # Setup multi-agent controller here
        self.home_mac = mac_REGISTRY[args.mac](self.home_buffer.scheme, groups, args)
        self.opponent_mac = mac_REGISTRY[args.mac](self.opponent_buffer.scheme, groups, args)

        # Give runner the scheme
        self.runner.setup(scheme=scheme, groups=groups, preprocess=preprocess, home_mac=self.home_mac,
                          opponent_mac=self.opponent_mac)

        # Learners
        self.home_learner = le_REGISTRY[args.learner](self.home_mac, self.home_buffer.scheme, logger, args, name="home")
        self.opponent_learner = le_REGISTRY[args.learner](self.opponent_mac, self.opponent_buffer.scheme, logger, args,
                                                          name="opponent")

        # Activate CUDA mode if supported
        if args.use_cuda:
            self.home_learner.cuda()
            self.opponent_learner.cuda()

        self.checkpoint_manager = CheckpointManager(args=args, logger=logger)

    def evaluate_sequential(self):
        for _ in range(self.args.test_nepisode):
            self.runner.run(test_mode=True)

        if self.args.save_replay:
            self.runner.save_replay()

        self.runner.close_env()

    def run(self):
        if self.args.checkpoint_path != "":
            timestep_to_load = self.checkpoint_manager.load(learners=[self.home_learner, self.opponent_learner])
            self.runner.t_env = timestep_to_load

            if self.args.evaluate or self.args.save_replay:
                self.evaluate_sequential()
                return

        # start training
        episode = 0
        last_test_T = -self.args.test_interval - 1
        last_log_T = 0
        model_save_time = 0

        start_time = time.time()
        last_time = start_time

        self.logger.console_logger.info("Beginning training for {} timesteps".format(self.args.t_max))

        #
        #
        # Main Loop
        #
        #
        while self.runner.t_env <= self.args.t_max:

            # Run for a whole episode at a time
            home_batch, opponent_batch = self.runner.run(test_mode=False)

            self.home_buffer.insert_episode_batch(home_batch)
            self.opponent_buffer.insert_episode_batch(opponent_batch)

            # Sample batch from buffer if possible
            batch_size = self.args.batch_size
            if self.home_buffer.can_sample(batch_size) and self.opponent_buffer.can_sample(batch_size):
                home_sample = self.home_buffer.sample(batch_size)
                opponent_sample = self.opponent_buffer.sample(batch_size)

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

                self.home_learner.train(home_sample, self.runner.t_env, episode)
                self.opponent_learner.train(opponent_sample, self.runner.t_env, episode)

            #
            # Execute test runs once in a while
            #
            n_test_runs = max(1, self.args.test_nepisode // self.runner.batch_size)
            if (self.runner.t_env - last_test_T) / self.args.test_interval >= 1.0:

                self.logger.console_logger.info("t_env: {} / {}".format(self.runner.t_env, self.args.t_max))
                self.logger.console_logger.info("Estimated time left: {}. Time passed: {}".format(
                    time_left(last_time, last_test_T, self.runner.t_env, self.args.t_max),
                    time_str(time.time() - start_time)))
                last_time = time.time()

                last_test_T = self.runner.t_env
                for _ in range(n_test_runs):
                    self.runner.run(test_mode=True)

            # Model saving
            if self.args.save_model and (self.runner.t_env - model_save_time >= self.args.save_model_interval or model_save_time == 0):
                model_save_time = self.runner.t_env
                self.checkpoint_manager.save(model_save_time, learners=[self.home_learner, self.opponent_learner])

            # Batch size == how many episodes are run -> add on top of episode counter
            episode += self.args.batch_size_run

            # Log
            if (self.runner.t_env - last_log_T) >= self.args.log_interval:
                self.logger.add_stat("episode", episode, self.runner.t_env)
                self.logger.log_recent_stats()
                last_log_T = self.runner.t_env
        #
        #
        #
        #
        #

        self.runner.close_env()
        self.logger.console_logger.info("Finished Training")
