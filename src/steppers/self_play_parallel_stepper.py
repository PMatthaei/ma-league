from gym.vector.utils import CloudpickleWrapper

from envs import REGISTRY as env_REGISTRY
from functools import partial
from components.episode_buffer import EpisodeBatch
from multiprocessing import Pipe
import numpy as np

from steppers.env_worker_process import EnvWorker
import torch as th

from utils.logging import Originator


class SelfPlayParallelStepper:
    def __init__(self, args, logger):
        """
        Based (very) heavily on SubprocVecEnv from OpenAI Baselines
        https://github.com/openai/baselines/blob/master/baselines/common/vec_env/subproc_vec_env.py
        Runs multiple environments in parallel to play and collects episode batches to feed into a single learner.
        :param args:
        :param logger:
        """
        self.args = args
        self.logger = logger
        self.batch_size = self.args.batch_size_run
        self.policy_team_id = 0

        # Make subprocesses for the envs
        self.parent_conns, self.worker_conns = zip(*[Pipe() for _ in range(self.batch_size)])
        env_fn = env_REGISTRY[self.args.env]
        self.workers = [
            EnvWorker(worker_conn, CloudpickleWrapper(partial(env_fn, **self.args.env_args)), self.policy_team_id)
            for worker_conn in self.worker_conns
        ]

        for worker in self.workers:
            worker.daemon = True
            worker.start()

        self.parent_conns[0].send(("get_env_info", None))
        self.env_info = self.parent_conns[0].recv()
        self.episode_limit = self.env_info["episode_limit"]

        self.t = 0

        self.t_env = 0

        self.train_returns = []
        self.test_returns = []
        self.train_stats = {}
        self.test_stats = {}

        self.log_train_stats_t = -100000
        self.new_batch_fn = None
        self.scheme = None
        self.groups = None
        self.preprocess = None
        self.env_steps_this_run = 0

        self.home_mac = None
        self.away_mac = None
        self.home_batch = None
        self.away_batch = None

    def initialize(self, scheme, groups, preprocess, home_mac, opponent_mac):
        self.new_batch_fn = partial(EpisodeBatch, scheme, groups, self.batch_size, self.episode_limit + 1,
                                    preprocess=preprocess, device=self.args.device)
        self.scheme = scheme
        self.groups = groups
        self.preprocess = preprocess
        self.home_mac = home_mac
        self.away_mac = opponent_mac

    @property
    def epsilons(self):
        return getattr(self.home_mac.action_selector, "epsilon", None), \
               getattr(self.away_mac.action_selector, "epsilon", None)

    def get_env_info(self):
        return self.env_info

    def save_replay(self):
        pass

    def close_env(self):
        for parent_conn in self.parent_conns:
            parent_conn.send(("close", None))

    def reset(self):
        self.home_batch = self.new_batch_fn()
        self.away_batch = self.new_batch_fn()

        # Reset the envs
        for parent_conn in self.parent_conns:
            parent_conn.send(("reset", None))
        # Pre transition data
        home_ptd = {
            "state": [],
            "avail_actions": [],
            "obs": []
        }
        away_ptd = home_ptd.copy()

        # Get the obs, state and avail_actions back
        for parent_conn in self.parent_conns:
            data = parent_conn.recv()
            away_ptd, home_ptd = self._append_pre_transition_data(away_ptd, home_ptd, data)

        self.home_batch.update(home_ptd, ts=0)
        self.away_batch.update(away_ptd, ts=0)
        self.t = 0
        self.env_steps_this_run = 0

    def _append_pre_transition_data(self, away_pre_transition_data, home_pre_transition_data, data):
        state = data["state"]
        actions = data["avail_actions"]
        obs = data["obs"]
        # TODO: only supports same team sizes!
        home_avail_actions = actions[:len(actions) // 2]
        home_obs = obs[:len(obs) // 2]
        home_pre_transition_data["state"].append(state)
        home_pre_transition_data["avail_actions"].append(home_avail_actions)
        home_pre_transition_data["obs"].append(home_obs)

        away_avail_actions = actions[len(actions) // 2:]
        away_obs = obs[len(obs) // 2:]
        away_pre_transition_data["state"].append(state)
        away_pre_transition_data["avail_actions"].append(away_avail_actions)
        away_pre_transition_data["obs"].append(away_obs)
        return away_pre_transition_data, home_pre_transition_data

    def run(self, test_mode=False):
        self.reset()

        all_terminated = False
        home_episode_returns = [0 for _ in range(self.batch_size)]
        away_episode_returns = [0 for _ in range(self.batch_size)]
        episode_lengths = [0 for _ in range(self.batch_size)]

        self.home_mac.init_hidden(batch_size=self.batch_size)
        self.away_mac.init_hidden(batch_size=self.batch_size)

        terminated = [False for _ in range(self.batch_size)]
        envs_not_terminated = [b_idx for b_idx, termed in enumerate(terminated) if not termed]
        final_env_infos = []  # may store extra stats like battle won. this is filled in ORDER OF TERMINATION

        while True:

            # Pass the entire batch of experiences up till now to the agents
            # Receive the actions for each agent at this timestep in a batch for each un-terminated env
            home_actions = self.home_mac.select_actions(self.home_batch, t_ep=self.t, t_env=self.t_env,
                                                        test_mode=test_mode)
            away_actions = self.away_mac.select_actions(self.away_batch, t_ep=self.t, t_env=self.t_env,
                                                        test_mode=test_mode)
            actions = th.cat((home_actions[0], away_actions[0]))

            cpu_actions = actions.to("cpu").numpy()

            # Update the actions taken
            home_actions_chosen = {
                "actions": home_actions.unsqueeze(1)
            }
            away_actions_chosen = {
                "actions": away_actions.unsqueeze(1)
            }
            self.home_batch.update(home_actions_chosen, bs=envs_not_terminated, ts=self.t, mark_filled=False)
            self.away_batch.update(away_actions_chosen, bs=envs_not_terminated, ts=self.t, mark_filled=False)

            # Send actions to each env
            action_idx = 0
            for idx, parent_conn in enumerate(self.parent_conns):
                if idx in envs_not_terminated:  # We produced actions for this env
                    if not terminated[idx]:  # Only send the actions to the env if it hasn't terminated
                        parent_conn.send(("step", cpu_actions[action_idx]))
                    action_idx += 1  # actions is not a list over every env

            # Update envs_not_terminated
            envs_not_terminated = [b_idx for b_idx, termed in enumerate(terminated) if not termed]
            all_terminated = all(terminated)
            if all_terminated:
                break

            # Post step data we will insert for the current timestep
            home_post_transition_data = {
                "reward": [],
                "terminated": []
            }
            away_post_transition_data = home_post_transition_data.copy()

            # Data for the next step we will insert in order to select an action
            home_pre_transition_data = {
                "state": [],
                "avail_actions": [],
                "obs": []
            }
            away_pre_transition_data = home_pre_transition_data.copy()

            # Receive data back for each unterminated env
            for idx, parent_conn in enumerate(self.parent_conns):
                if not terminated[idx]:
                    data = parent_conn.recv()
                    # Remaining data for this current timestep
                    home_post_transition_data["reward"].append((data["reward"][0],))
                    away_post_transition_data["reward"].append((data["reward"][0],))

                    home_episode_returns[idx] += data["reward"][0]
                    away_episode_returns[idx] += data["reward"][1]
                    episode_lengths[idx] += 1

                    if not test_mode:
                        self.env_steps_this_run += 1

                    env_terminated = False
                    if data["terminated"]:
                        final_env_infos.append(data["info"])
                    if data["terminated"] and not data["info"].get("episode_limit", False):
                        env_terminated = True
                    terminated[idx] = data["terminated"]
                    home_post_transition_data["terminated"].append((env_terminated,))
                    away_post_transition_data["terminated"].append((env_terminated,))

                    # Data for the next timestep needed to select an action
                    self._append_pre_transition_data(away_pre_transition_data, home_pre_transition_data, data)

            # Add post_transiton data into the batch
            self.home_batch.update(home_post_transition_data, bs=envs_not_terminated, ts=self.t, mark_filled=False)
            self.away_batch.update(away_post_transition_data, bs=envs_not_terminated, ts=self.t, mark_filled=False)

            # Move onto the next timestep
            self.t += 1

            # Add the pre-transition data
            self.home_batch.update(home_pre_transition_data, bs=envs_not_terminated, ts=self.t, mark_filled=False)
            self.away_batch.update(away_pre_transition_data, bs=envs_not_terminated, ts=self.t, mark_filled=False)

        if not test_mode:
            self.t_env += self.env_steps_this_run

        # Get stats back for each env
        for parent_conn in self.parent_conns:
            parent_conn.send(("get_stats", None))

        env_stats = []
        for parent_conn in self.parent_conns:
            env_stat = parent_conn.recv()
            env_stats.append(env_stat)

        self.logger.collect_episode_returns(home_episode_returns, parallel=True)
        self.logger.collect_episode_returns(away_episode_returns, org=Originator.AWAY, parallel=True)
        self.logger.collect_episode_stats(final_env_infos, self.t, parallel=True, batch_size=self.batch_size,
                                          ep_lens=episode_lengths)
        self.logger.add_stats(self.t_env, epsilons=self.epsilons)

        return self.home_batch, self.away_batch
