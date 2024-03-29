import torch as th
from bin.controls.headless_controls import HeadlessControls

from custom_logging.logger import MainLogger, Collectibles
from custom_logging.utils.enums import Originator
from envs import REGISTRY as env_REGISTRY
from functools import partial
from marl.components.episode_batch import EpisodeBatch
from exceptions.runner_exceptions import MultiAgentControllerNotInitialized
from steppers.env_stepper import EnvStepper
from steppers.utils.stepper_utils import get_policy_team_id
from marl.components.feature_functions import REGISTRY as feature_func_REGISTRY, FeatureFunction


class EpisodeStepper(EnvStepper):

    def __init__(self, args, logger: MainLogger, log_start_t=0):
        """
        Runs a single episode and returns the gathered step data as episode batch to feed into a single learner.
        This runner is only supported one training agent against a AI/environment.
        :param args: args passed from main
        :param logger: logger
        """

        super().__init__(args, logger)
        self.batch_size = self.args.batch_size_run
        assert self.batch_size == 1

        self.env = env_REGISTRY[self.args.env](**self.args.env_args)
        # Find id of the first policy team - Only supported for one policy team in the build plan
        teams = args.env_args["match_build_plan"]
        self.policy_team_id = get_policy_team_id(teams)
        if self.args.headless_controls:
            controls = HeadlessControls(env=self.env)
            controls.daemon = True
            controls.start()

        self.episode_limit = self.env.episode_limit
        self.t = 0  # current time step within the episode
        self.log_start_t = log_start_t # timestep to start logging from
        self.t_env = 0  # total time steps for this runner in the provided environment across multiple episodes
        self.phi: FeatureFunction = feature_func_REGISTRY[self.args.sfs] if self.args.sfs else None
        self.home_batch = None
        self.home_mac = None
        self.new_batch_fn = None

    def initialize(self, scheme, groups, preprocess, home_mac, away_mac=None):
        self.new_batch_fn = partial(
            EpisodeBatch,
            scheme,
            groups,
            self.batch_size,
            self.episode_limit + 1,  # last step
            preprocess=preprocess,
            device=self.args.device
        )
        self.home_mac = home_mac
        self.is_initalized = True

    def get_env_info(self):
        return self.env.get_env_info()

    def save_replay(self):
        raise NotImplementedError()

    def close_env(self):
        self.env.close()

    def reset(self):
        self.home_batch = self.new_batch_fn()
        self.env.reset()
        self.t = 0

    def rebuild_env(self, env_args):
        self.env.close()
        self.env = env_REGISTRY[self.args.env](**env_args)

    @property
    def epsilon(self):
        return getattr(self.home_mac.action_selector, "epsilon", None)

    @property
    def log_t(self):
        return self.log_start_t + self.t_env

    def run(self, test_mode=False):
        """
        Run a single episode
        :param test_mode:
        :return:
        """
        self.reset()

        if self.home_mac is None:
            raise MultiAgentControllerNotInitialized()

        terminated = False
        episode_return = 0
        actions_taken = []

        self.logger.test_mode = test_mode

        self.home_mac.init_hidden(batch_size=self.batch_size)

        self.env.render()

        env_info = {}

        while not terminated:
            pre_transition_data = self.perform_pre_transition_step()
            actions, is_greedy = self.home_mac.select_actions(
                self.home_batch,
                t_ep=self.t,
                t_env=self.t_env,
                test_mode=test_mode
            )
            if is_greedy is not None:
                actions_taken.append(th.stack([actions, is_greedy]))

            obs, reward, done_n, env_info = self.env.step(actions[0])
            terminated = any(done_n)

            self.env.render()

            episode_return += reward[0]  # WARN! Only supported if one policy team is playing
            post_transition_data = {
                "actions": actions,
                "reward": [(reward[0],)],  # WARN! Only supported if one policy team is playing
                "terminated": [(terminated,)],
            }

            if self.phi is not None:  # Calculate features based on (o, a, o')
                self.add_features(actions, obs, pre_transition_data)

            self.home_batch.update(post_transition_data, ts=self.t)
            # Termination is dependent on all team-wise terminations - AI or policy controlled teams

            self.t += 1

        _ = self.perform_pre_transition_step()
        actions, is_greedy = self.home_mac.select_actions(
            self.home_batch,
            t_ep=self.t,
            t_env=self.t_env,
            test_mode=test_mode
        )

        if is_greedy is not None:
            actions_taken.append(th.stack([actions, is_greedy]))

        self.home_batch.update({"actions": actions}, ts=self.t)

        if not test_mode:
            self.t_env += self.t

        if len(actions_taken) > 0:
            actions_taken = th.squeeze(th.stack(actions_taken))
            self.logger.collect(Collectibles.ACTIONS_TAKEN, actions_taken, origin=Originator.HOME)

        # Send data collected during the episode - this data needs further processing
        self.logger.collect(Collectibles.RETURN, episode_return, origin=Originator.HOME)
        self.logger.collect(Collectibles.WON, env_info["battle_won"][0], origin=Originator.HOME)
        self.logger.collect(Collectibles.WON, env_info["battle_won"][1], origin=Originator.AWAY)
        self.logger.collect(Collectibles.DRAW, env_info["draw"])
        self.logger.collect(Collectibles.STEPS, self.t)
        # Log epsilon from mac directly
        self.logger.log_stat("home_epsilon", self.epsilon, self.log_t)
        # Log collectibles if conditions suffice
        self.logger.log(self.log_t)

        return self.home_batch, env_info

    def add_features(self, actions, obs_next, pre_transition_data):
        obs = pre_transition_data["obs"][0]
        self.home_batch.update({"features": self.phi(obs, actions, obs_next)}, ts=self.t)

    def perform_pre_transition_step(self):
        # Build
        pre_transition_data = {
            "state": [self.env.get_state()],
            "avail_actions": [self.env.get_avail_actions()],
            "obs": [self.env.get_obs()]
        }
        # Add to episode batch
        self.home_batch.update(pre_transition_data, ts=self.t)
        return pre_transition_data
