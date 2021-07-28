import pprint
from copy import copy
from types import SimpleNamespace
from typing import Dict

import torch as th

from components.replay_buffers.replay_buffer import ReplayBuffer
from components.transforms import OneHot
from controllers import EnsembleMAC
from runs.experiment_run import ExperimentRun
from steppers import REGISTRY as stepper_REGISTRY
from steppers.episode_stepper import EnvStepper
from utils.asset_manager import AssetManager

# Config TODO: Pack into args
MODEL_COLLECTION_BASE_PATH = "/home/pmatthaei/Projects/ma-league-results/models/"

# POLICY_TEAM = MODEL_COLLECTION_BASE_PATH + "4/qmix__2021-07-09_12-49-37_team_0"
POLICY_TEAM = MODEL_COLLECTION_BASE_PATH + "2/qmix__2021-07-07_12-52-06_team_0"

# POLICY_TEAM = MODEL_COLLECTION_BASE_PATH + "3/qmix__2021-07-08_22-23-56_team_0"
# POLICY_TEAM = MODEL_COLLECTION_BASE_PATH + "2/qmix__2021-07-07_12-52-06_team_0"
POLICY_TEAM_ID = 1


class ReplayGenerationRun(ExperimentRun):

    def __init__(self, args, logger):
        """
        Load a saved model into the Multi-Agent Controller and perform inference for a given amount of time.
        Episodes run within this time are captured as video for later replay and visual understanding of the loaded
        model/policy.
        :param args:
        :param logger:
        """
        super().__init__(args, logger)
        self.asset_manager = AssetManager(args=self.args, logger=self.logger)

        self.home_mac, self.away_mac = None, None

        # Init stepper so we can get env info
        self.stepper = self._build_stepper()

        # Get env info from stepper
        self.env_info = self.stepper.get_env_info()

        # Retrieve important data from the env and set in args - call BEFORE scheme building
        env_scheme = self._update_args()

        self.logger.update_scheme(env_scheme)

        # Default/Base scheme - call AFTER extracting env info
        self.groups, self.preprocess, self.scheme = self._build_schemes()

        self.home_buffer = ReplayBuffer(self.scheme, self.groups, self.args.buffer_size,
                                        self.env_info["episode_limit"] + 1,
                                        preprocess=self.preprocess,
                                        device="cpu" if self.args.buffer_cpu_only else self.args.device)
        # Setup multi-agent controller here
        self.home_mac = EnsembleMAC(self.home_buffer.scheme, self.groups, self.args)
        self.away_mac = EnsembleMAC(self.home_buffer.scheme, self.groups, self.args)

    def _update_args(self) -> Dict:
        shapes = {
            "n_agents": int(self.env_info["n_agents"]),
            "n_actions": int(self.env_info["n_actions"]),
            "state_shape": int(self.env_info["state_shape"])
        }
        self.args = SimpleNamespace(**{**vars(self.args), **shapes})  # Update
        return shapes  # return delta to re-use

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
            "agents": int(self.env_info["n_agents"])
        }
        preprocess = {
            "actions": ("actions_onehot", [OneHot(out_dim=int(self.env_info["n_actions"]))])
        }
        return groups, preprocess, scheme

    def _build_stepper(self) -> EnvStepper:
        self._configure_environment_args()

        return stepper_REGISTRY[self.args.runner](args=self.args, logger=self.logger)

    def _configure_environment_args(self):
        policy_team_id = POLICY_TEAM_ID
        ai_team_id = 1 - policy_team_id

        self.args.env_args["record"] = True
        self.args.env_args["debug_range"] = True
        self.args.env_args["stochastic_spawns"] = False
        home_team = self.asset_manager.load_team(path=POLICY_TEAM)
        self.args.env_args['match_build_plan'][policy_team_id] = home_team
        self.args.env_args['match_build_plan'][ai_team_id] = copy(home_team)
        self.args.env_args['match_build_plan'][policy_team_id]['tid'] = policy_team_id
        self.args.env_args['match_build_plan'][ai_team_id]['tid'] = ai_team_id
        self.args.env_args['match_build_plan'][ai_team_id]['is_scripted'] = True

    def _init_stepper(self):
        if not self.stepper.is_initalized:
            self.stepper.initialize(scheme=self.scheme, groups=self.groups, preprocess=self.preprocess,
                                    home_mac=self.home_mac)

    def load_agents(self):
        state = self.asset_manager.load_state(path=POLICY_TEAM, component="agent")
        self.home_mac.load_state_dict(agent=state)

    def start(self, play_time_seconds=None):
        """
        :param play_time_seconds: Play the run for a certain time in seconds.
        :return:
        """
        self.logger.info("Experiment Parameters:")
        experiment_params = pprint.pformat(self.args.__dict__, indent=4, width=1)
        self.logger.info("\n\n" + experiment_params + "\n")

        self.load_agents()

        self._init_stepper()

        # start training
        self.logger.info("Beginning inference for {} episodes.".format(self.args.test_nepisode))
        episode = 0

        while episode < self.args.test_nepisode:
            # Run for a whole episode at a time
            _, _ = self.stepper.run(test_mode=True)
            self.logger.log_stat("episode", episode, self.stepper.t_env)

            episode += 1

        self.logger.log_report()

        # Finish and clean up
        self._finish()

    def _finish(self):
        self.stepper.close_env()
        self.logger.info("Finished.")

    def _build_learners(self):
        pass  # we do not need learners since...

    def _train_episode(self, episode_num):
        pass  # ... we do not train, just infer
