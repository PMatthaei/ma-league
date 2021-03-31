from bin.controls.headless_controls import HeadlessControls

from envs import REGISTRY as env_REGISTRY
from functools import partial
from components.episode_buffer import EpisodeBatch
from exceptions.runner_exceptions import RunnerMACNotInitialized


class EpisodeRunner:

    def __init__(self, args, logger):
        """
        Runs a single episode and returns the gathered step data as episode batch to feed into a single learner.
        This runner is only supported one training agent against a AI/environment.
        :param args: args passed from main
        :param logger: logger
        """
        self.args = args
        self.logger = logger
        self.batch_size = self.args.batch_size_run
        assert self.batch_size == 1

        self.env = env_REGISTRY[self.args.env](**self.args.env_args)
        self.policy_team_id = 0
        controls = HeadlessControls(env=self.env)
        controls.daemon = True
        controls.start()

        self.episode_limit = self.env.episode_limit
        self.t = 0  # current time step within the episode

        self.t_env = 0  # total time steps for this runner in the provided environment across multiple episodes

        self.batch = None
        self.mac = None
        self.new_batch_fn = None

    def initialize(self, scheme, groups, preprocess, mac):
        self.new_batch_fn = partial(EpisodeBatch, scheme, groups, self.batch_size, self.episode_limit + 1,  # last step
                                    preprocess=preprocess, device=self.args.device)
        self.mac = mac

    def get_env_info(self):
        return self.env.get_env_info()

    def save_replay(self):
        raise NotImplementedError()

    def close_env(self):
        self.env.close()

    def reset(self):
        self.batch = self.new_batch_fn()
        self.env.reset()
        self.t = 0

    @property
    def epsilon(self):
        return getattr(self.mac.action_selector, "epsilon", None)

    def run(self, test_mode=False):

        self.reset()

        if self.mac is None:
            raise RunnerMACNotInitialized()

        terminated = False
        episode_return = 0

        self.mac.init_hidden(batch_size=self.batch_size)

        self.logger.test_mode = test_mode
        self.logger.test_n_episode = self.args.test_nepisode
        self.logger.runner_log_interval = self.args.runner_log_interval

        self.env.render()

        env_info = {}

        while not terminated:
            pre_transition_data = {
                "state": [self.env.get_state()],
                "avail_actions": [self.env.get_avail_actions()],
                "obs": [self.env.get_obs()]
            }

            self.batch.update(pre_transition_data, ts=self.t)

            # Pass the entire batch of experiences up till now to the agents
            # Receive the actions for each agent at this timestep in a batch of size 1
            actions = self.mac.select_actions(self.batch, t_ep=self.t, t_env=self.t_env, test_mode=test_mode)

            obs, reward, done_n, env_info = self.env.step(actions[0])
            self.env.render()

            episode_return += reward[self.policy_team_id]

            post_transition_data = {
                "actions": actions,
                "reward": [(reward[self.policy_team_id],)],
                "terminated": [(done_n[self.policy_team_id],)],
            }

            self.batch.update(post_transition_data, ts=self.t)
            # Termination is dependent on all team-wise terminations - AI or policy controlled teams
            terminated = any(done_n)

            self.t += 1

        last_data = {
            "state": [self.env.get_state()],
            "avail_actions": [self.env.get_avail_actions()],
            "obs": [self.env.get_obs()]
        }

        self.batch.update(last_data, ts=self.t)

        # Select actions in the last stored state
        actions = self.mac.select_actions(self.batch, t_ep=self.t, t_env=self.t_env, test_mode=test_mode)
        self.batch.update({"actions": actions}, ts=self.t)

        if not test_mode:
            self.t_env += self.t

        self.logger.collect_episode_returns(episode_return)
        self.logger.collect_episode_stats(env_info, self.t)
        self.logger.add_stats(self.t_env, epsilons=self.epsilon)

        return self.batch
