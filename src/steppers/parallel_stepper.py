import torch
from gym.vector.utils import CloudpickleWrapper

from envs import REGISTRY as env_REGISTRY
from functools import partial
from components.episode_buffer import EpisodeBatch
from multiprocessing import Pipe

from steppers.utils.env_worker_process import EnvWorker


class ParallelStepper:
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
        # Find id of the first policy team - Only supported for one policy team in the build plan
        teams = args.env_args["match_build_plan"]
        self.policy_team_id = teams.index(next(filter(lambda x: not x["is_scripted"], teams), None))

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
        self.home_batch = None

    def initialize(self, scheme, groups, preprocess, home_mac, away_mac=None):
        self.new_batch_fn = partial(EpisodeBatch, scheme, groups, self.batch_size, self.episode_limit + 1,
                                    preprocess=preprocess, device=self.args.device)
        self.home_mac = home_mac
        self.scheme = scheme
        self.groups = groups
        self.preprocess = preprocess

    def get_env_info(self):
        return self.env_info

    def save_replay(self):
        pass

    def close_env(self):
        for parent_conn in self.parent_conns:
            parent_conn.send(("close", None))

    def reset(self):
        self.home_batch = self.new_batch_fn()

        # Reset the envs
        for parent_conn in self.parent_conns:
            parent_conn.send(("reset", None))

        pre_transition_data = {
            "state": [],
            "avail_actions": [],
            "obs": []
        }
        # Get the obs, state and avail_actions back
        for parent_conn in self.parent_conns:
            data = parent_conn.recv()
            pre_transition_data["state"].append(data["state"])
            pre_transition_data["avail_actions"].append(data["avail_actions"])
            pre_transition_data["obs"].append(data["obs"])

        self.home_batch.update(pre_transition_data, ts=0)

        self.t = 0
        self.env_steps_this_run = 0

    def run(self, test_mode=False):
        """
        Run a single episode with multiple environments in parallel
        :param test_mode:
        :return:
        """
        self.reset()

        self.logger.test_mode = test_mode
        self.logger.test_n_episode = self.args.test_nepisode
        self.logger.runner_log_interval = self.args.runner_log_interval

        all_terminated = False
        episode_returns = [0 for _ in range(self.batch_size)]
        eps = [0 for _ in range(self.batch_size)]
        self.home_mac.init_hidden(batch_size=self.batch_size)
        # bools to determine finished envs
        terminateds = [False for _ in range(self.batch_size)]
        # IDs of running envs
        running_envs = [idx for idx, terminated in enumerate(terminateds) if not terminated]
        env_infos = []  # may store extra stats like battle won. this is filled in ORDER OF TERMINATION

        self.actions = torch.zeros((self.batch_size, self.args.n_agents))
        while True:

            # Pass the entire batch of experiences up till now to the agents
            # Receive the actions for each agent at this timestep in a batch for each un-terminated env
            actions = self.home_mac.select_actions(self.home_batch, t_ep=self.t, t_env=self.t_env, bs=running_envs,
                                                   test_mode=test_mode)

            # Update the actions taken
            actions_chosen = {
                "actions": actions.unsqueeze(1)
            }
            self.home_batch.update(actions_chosen, bs=running_envs, ts=self.t, mark_filled=False)

            # Send actions to each running env
            action_idx = 0
            for idx, parent_conn in enumerate(self.parent_conns):
                if idx in running_envs:  # We produced actions for this env
                    if not terminateds[idx]:  # Only send the actions to the env if it hasn't terminated
                        self.actions[action_idx] = actions[action_idx]
                        parent_conn.send(("step", self.actions[action_idx]))
                    action_idx += 1  # actions is not a list over every env

            # Update running envs
            running_envs = [idx for idx, terminated in enumerate(terminateds) if not terminated]
            if all(terminateds):
                break

            # Post step data we will insert for the current timestep
            post_transition_data = {
                "reward": [],
                "terminated": []
            }
            # Data for the next step we will insert in order to select an action
            pre_transition_data = {
                "state": [],
                "avail_actions": [],
                "obs": []
            }

            # Receive step data back for each unterminated env
            for idx, parent_conn in enumerate(self.parent_conns):
                if not terminateds[idx]:
                    data = parent_conn.recv()
                    # Remaining data for this current timestep
                    policy_team_reward = data["reward"][0]  # ! Only supported if one policy team is playing
                    post_transition_data["reward"].append((policy_team_reward,))

                    episode_returns[idx] += policy_team_reward
                    eps[idx] += 1

                    if not test_mode:
                        self.env_steps_this_run += 1

                    done_n = data["terminated"]  # list of done booleans per team
                    terminated = any(done_n)
                    if terminated:  # if any team is done -> env terminated
                        env_infos.append(data["info"])
                    terminateds[idx] = terminated
                    post_transition_data["terminated"].append((done_n[self.policy_team_id],))

                    # Data for the next timestep needed to select an action
                    pre_transition_data["state"].append(data["state"])
                    pre_transition_data["avail_actions"].append(data["avail_actions"])
                    pre_transition_data["obs"].append(data["obs"])

            # Add post_transition data into the batch
            self.home_batch.update(post_transition_data, bs=running_envs, ts=self.t, mark_filled=False)

            # Move onto the next timestep
            self.t += 1

            # Add the pre-transition data
            self.home_batch.update(pre_transition_data, bs=running_envs, ts=self.t, mark_filled=True)

        if not test_mode:
            self.t_env += self.env_steps_this_run

        self.logger.collect_episode_returns(episode_returns, parallel=True)
        self.logger.collect_episode_stats(env_infos, self.t, parallel=True, batch_size=self.batch_size, ep_lens=eps)
        self.logger.add_stats(self.t_env, epsilons=self.home_mac.action_selector.epsilon)

        return self.home_batch
