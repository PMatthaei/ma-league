import random
from logging import warning
from multiprocessing.dummy import Process

import os
import time
from multiprocessing.connection import Connection

import torch as th
from types import SimpleNamespace

from league.roles.players import Player
from learners.learner import Learner
from runners.self_play_runner import SelfPlayRunner
from utils.logging import LeagueLogger
from utils.timehelper import time_left, time_str

from learners import REGISTRY as le_REGISTRY
from controllers import REGISTRY as mac_REGISTRY
from components.episode_buffer import ReplayBuffer
from components.transforms import OneHot


class LeagueProcess(Process):
    def __init__(self, home: Player, conn: Connection, args: SimpleNamespace, logger: LeagueLogger):
        super().__init__()
        self.home = home
        self.conn = conn
        self.args = args
        self.logger = logger

        self.away = None
        self.terminated = False

    def run(self):
        # Init runner so we can get env info
        runner = SelfPlayRunner(args=self.args, logger=self.logger)

        # Set up schemes and groups here
        env_info = runner.get_env_info()
        self.args.n_agents = int(env_info["n_agents"] / 2)  # TODO: assuming same team size
        self.args.n_actions = env_info["n_actions"]
        self.args.state_shape = env_info["state_shape"]

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
            "agents": self.args.n_agents
        }
        preprocess = {
            "actions": ("actions_onehot", [OneHot(out_dim=self.args.n_actions)])
        }

        # Buffers
        away_buffer = ReplayBuffer(scheme, groups, self.args.buffer_size, env_info["episode_limit"] + 1,
                                   preprocess=preprocess,
                                   device="cpu" if self.args.buffer_cpu_only else self.args.device)

        home_buffer = ReplayBuffer(scheme, groups, self.args.buffer_size, env_info["episode_limit"] + 1,
                                   preprocess=preprocess,
                                   device="cpu" if self.args.buffer_cpu_only else self.args.device)

        # Setup multi-agent controller here
        home_mac = mac_REGISTRY[self.args.mac](home_buffer.scheme, groups, self.args)
        away_mac = mac_REGISTRY[self.args.mac](away_buffer.scheme, groups, self.args)

        # Give runner the scheme
        runner.setup(scheme=scheme, groups=groups, preprocess=preprocess, home_mac=home_mac, opponent_mac=away_mac)

        # Learners
        home_learner = le_REGISTRY[self.args.learner](home_mac, home_buffer.scheme, self.logger, self.args, name="home")
        away_learner = le_REGISTRY[self.args.learner](away_mac, away_buffer.scheme, self.logger, self.args, name="away")

        # Activate CUDA mode if supported
        if self.args.use_cuda:
            home_learner.cuda()
            away_learner.cuda()

        while not self.terminated:  # TODO end condition
            # Generate new opponent to train against and load his current checkpoint
            self.away, flag = self.home.get_match()

            if self.away is None:
                warning("Opponent was none")
                continue

            self.load_checkpoint(self.away, away_learner, runner)

            self.logger.console_logger.info(self._get_match_str())

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
            while runner.t_env <= self.args.t_max:

                # Run for a whole episode at a time
                home_batch, opponent_batch = runner.run(test_mode=False)

                # Fake episode play with sleep and fake result TODO
                result = random.choice(["win", "draw", "loss"])
                self.conn.send({"result": (self.home.player_id, self.away.player_id, result)})

                home_buffer.insert_episode_batch(home_batch)
                away_buffer.insert_episode_batch(opponent_batch)

                # Sample batch from buffer if possible
                if home_buffer.can_sample(self.args.batch_size) and away_buffer.can_sample(self.args.batch_size):
                    home_sample = home_buffer.sample(self.args.batch_size)
                    opponent_sample = away_buffer.sample(self.args.batch_size)

                    # Truncate batch to only filled timesteps
                    max_ep_t_h = home_sample.max_t_filled()
                    max_ep_t_o = opponent_sample.max_t_filled()
                    home_sample = home_sample[:, :max_ep_t_h]
                    opponent_sample = opponent_sample[:, :max_ep_t_o]

                    if home_sample.device != self.args.device:
                        home_sample.to(self.args.device)

                    if opponent_sample.device != self.args.device:
                        opponent_sample.to(self.args.device)

                    home_learner.train(home_sample, runner.t_env, episode)
                    away_learner.train(opponent_sample, runner.t_env, episode)

                #
                # Execute test runs once in a while
                #
                n_test_runs = max(1, self.args.test_nepisode // runner.batch_size)
                if (runner.t_env - last_test_T) / self.args.test_interval >= 1.0:

                    self.logger.console_logger.info("t_env: {} / {}".format(runner.t_env, self.args.t_max))
                    self.logger.console_logger.info("Estimated time left: {}. Time passed: {}".format(
                        time_left(last_time, last_test_T, runner.t_env, self.args.t_max),
                        time_str(time.time() - start_time)))
                    last_time = time.time()

                    last_test_T = runner.t_env
                    for _ in range(n_test_runs):
                        runner.run(test_mode=True)

                # Model saving
                if self.args.save_model and (
                        runner.t_env - model_save_time >= self.args.save_model_interval or model_save_time == 0):
                    model_save_time = runner.t_env
                    save_path = os.path.join(self.args.local_results_path, "models", self.args.unique_token,
                                             str(runner.t_env))
                    # "results/models/{}".format(unique_token)
                    os.makedirs(save_path, exist_ok=True)
                    self.logger.console_logger.info("Saving models to {}".format(save_path))

                    # learner should handle saving/loading -- delegate actor save/load to mac,
                    # use appropriate filenames to do critics, optimizer states
                    home_learner.save_models(save_path)

                # Batch size == how many episodes are run -> add on top of episode counter
                episode += self.args.batch_size_run

                # Log
                if (runner.t_env - last_log_T) >= self.args.log_interval:
                    self.logger.add_stat("episode", episode, runner.t_env)
                    self.logger.log_recent_stats()
                    last_log_T = runner.t_env
            #
            #
            #
            #
            #

        runner.close_env()
        self.logger.console_logger.info("Finished Training")
        self.conn.send({"close": self.home.player_id})
        self.conn.close()

    def load_checkpoint(self, player: Player, learner: Learner, runner: SelfPlayRunner):
        if player.checkpoint_path != "":  # TODO move to agent code

            timesteps = []
            timestep_to_load = 0

            if not os.path.isdir(player.checkpoint_path):
                self.logger.console_logger.info(
                    "Checkpoint directory {} doesn't exist".format(self.args.checkpoint_path))
                return

            # Go through all files in self.args.checkpoint_path
            for name in os.listdir(player.checkpoint_path):
                full_name = os.path.join(player.checkpoint_path, name)
                # Check if they are dirs the names of which are numbers
                if os.path.isdir(full_name) and name.isdigit():
                    timesteps.append(int(name))

            if self.args.load_step == 0:
                # choose the max timestep
                timestep_to_load = max(timesteps)
            else:
                # choose the timestep closest to load_step
                timestep_to_load = min(timesteps, key=lambda x: abs(x - self.args.load_step))

            model_path = os.path.join(player.checkpoint_path, str(timestep_to_load))

            self.logger.console_logger.info("Loading model from {}".format(model_path))
            learner.load_models(model_path)
            runner.t_env = timestep_to_load

    def _get_match_str(self):
        player_str = f"{type(self.home).__name__} {self.home.player_id}"
        opponent_str = f"{type(self.away).__name__} {self.away.player_id} "
        return f"{player_str} playing against opponent {opponent_str} in Process {self.home.player_id}"
