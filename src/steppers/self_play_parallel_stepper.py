from functools import partial
from components.episode_buffer import EpisodeBatch

from steppers import ParallelStepper
import torch as th

from steppers.utils.transition_parser import append_pre_transition_data
from utils.logging import Originator


class SelfPlayParallelStepper(ParallelStepper):
    def __init__(self, args, logger):
        """
        Based (very) heavily on SubprocVecEnv from OpenAI Baselines
        https://github.com/openai/baselines/blob/master/baselines/common/vec_env/subproc_vec_env.py
        Runs multiple environments in parallel to play and collects episode batches to feed into a single learner.
        :param args:
        :param logger:
        """
        super().__init__(args, logger)
        self.away_mac = None
        self.away_batch = None

    def initialize(self, scheme, groups, preprocess, home_mac, away_mac=None):
        self.new_batch_fn = partial(EpisodeBatch, scheme, groups, self.batch_size, self.episode_limit + 1,
                                    preprocess=preprocess, device=self.args.device)
        self.scheme = scheme
        self.groups = groups
        self.preprocess = preprocess
        self.home_mac = home_mac
        self.away_mac = away_mac

    @property
    def epsilons(self):
        return getattr(self.home_mac.action_selector, "epsilon", None), \
               getattr(self.away_mac.action_selector, "epsilon", None)

    def save_replay(self):
        pass

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
        away_ptd = {
            "state": [],
            "avail_actions": [],
            "obs": []
        }

        # Get the obs, state and avail_actions back
        for parent_conn in self.parent_conns:
            data = parent_conn.recv()
            append_pre_transition_data(away_ptd, home_ptd, data)

        self.home_batch.update(home_ptd, ts=0)
        self.away_batch.update(away_ptd, ts=0)
        self.t = 0
        self.env_steps_this_run = 0

    def run(self, test_mode=False):
        """
        Run a single episode with multiple environments in parallel while performing self-play
        :param test_mode:
        :return:
        """
        self.reset()

        self.logger.test_mode = test_mode
        self.logger.test_n_episode = self.args.test_nepisode
        self.logger.runner_log_interval = self.args.runner_log_interval

        all_terminated = False
        home_episode_returns = [0 for _ in range(self.batch_size)]
        away_episode_returns = [0 for _ in range(self.batch_size)]
        ep_lens = [0 for _ in range(self.batch_size)]

        self.home_mac.init_hidden(batch_size=self.batch_size)
        self.away_mac.init_hidden(batch_size=self.batch_size)

        terminateds = [False for _ in range(self.batch_size)]
        running_envs = [idx for idx, terminated in enumerate(terminateds) if not terminated]
        env_infos = []  # may store extra stats like battle won. this is filled in ORDER OF TERMINATION

        while True:

            # Pass the entire batch of experiences up till now to the agents
            # Receive the actions for each agent at this timestep in a batch for each un-terminateds env
            home_actions = self.home_mac.select_actions(self.home_batch, t_ep=self.t, t_env=self.t_env, bs=running_envs,
                                                        test_mode=test_mode)
            away_actions = self.away_mac.select_actions(self.away_batch, t_ep=self.t, t_env=self.t_env, bs=running_envs,
                                                        test_mode=test_mode)

            actions = th.cat((home_actions, away_actions), dim=1)

            cpu_actions = actions.to("cpu").numpy()

            # Update the actions taken
            home_actions_chosen = {
                "actions": home_actions.unsqueeze(1)
            }
            away_actions_chosen = {
                "actions": away_actions.unsqueeze(1)
            }
            self.home_batch.update(home_actions_chosen, bs=running_envs, ts=self.t, mark_filled=False)
            self.away_batch.update(away_actions_chosen, bs=running_envs, ts=self.t, mark_filled=False)

            # Send actions to each env
            action_idx = 0
            for idx, parent_conn in enumerate(self.parent_conns):
                if idx in running_envs:  # We produced actions for this env
                    if not terminateds[idx]:  # Only send the actions to the env if it hasn't terminateds
                        parent_conn.send(("step", cpu_actions[action_idx]))
                        # parent_conn.send(("step", actions[action_idx])) # TODO only with torch/CUDA multiprocessing
                    action_idx += 1  # actions is not a list over every env

            # Update running_envs
            running_envs = [idx for idx, terminated in enumerate(terminateds) if not terminated]
            all_terminated = all(terminateds)
            if all_terminated:
                break

            # Post step data we will insert for the current timestep
            home_post_transition_data = {
                "reward": [],
                "terminated": []
            }
            away_post_transition_data = {
                "reward": [],
                "terminated": []
            }
            # Data for the next step we will insert in order to select an action
            home_pre_transition_data = {
                "state": [],
                "avail_actions": [],
                "obs": []
            }
            away_pre_transition_data = {
                "state": [],
                "avail_actions": [],
                "obs": []
            }
            # Receive data back for each unterminated env
            for idx, parent_conn in enumerate(self.parent_conns):
                if not terminateds[idx]:
                    data = parent_conn.recv()
                    home_reward, away_reward = data["reward"]
                    # Remaining data for this current timestep
                    home_post_transition_data["reward"].append((home_reward,))
                    away_post_transition_data["reward"].append((away_reward,))

                    home_episode_returns[idx] += home_reward
                    away_episode_returns[idx] += away_reward
                    ep_lens[idx] += 1

                    if not test_mode:
                        self.env_steps_this_run += 1

                    env_terminateds = data["terminated"]  # list of done booleans per team
                    env_terminated = any(env_terminateds)
                    if env_terminated:  # if any team is done
                        env_infos.append(data["info"])
                    terminateds[idx] = env_terminated
                    home_post_transition_data["terminated"].append((env_terminated,))
                    away_post_transition_data["terminated"].append((env_terminated,))  # TODO Correct to feed every agent the global termination?

                    # Data for the next timestep needed to select an action
                    append_pre_transition_data(away_pre_transition_data, home_pre_transition_data, data)

            # Add post_transiton data into the batch
            self.home_batch.update(home_post_transition_data, bs=running_envs, ts=self.t, mark_filled=False)
            self.away_batch.update(away_post_transition_data, bs=running_envs, ts=self.t, mark_filled=False)

            # Move onto the next timestep
            self.t += 1

            # Add the pre-transition data
            self.home_batch.update(home_pre_transition_data, bs=running_envs, ts=self.t, mark_filled=True)
            self.away_batch.update(away_pre_transition_data, bs=running_envs, ts=self.t, mark_filled=True)

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
        self.logger.collect_episode_stats(env_infos, self.t, parallel=True, batch_size=self.batch_size, ep_lens=ep_lens)
        self.logger.add_stats(self.t_env, epsilons=self.epsilons)

        return self.home_batch, self.away_batch, env_infos
