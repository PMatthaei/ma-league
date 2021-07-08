import pprint
import time
from copy import copy
from types import SimpleNamespace
from typing import Dict

import torch as th
from maenv.core import RoleTypes, UnitAttackTypes

from modules.agents import Agent
from runs.experiment_run import ExperimentRun
from steppers.episode_stepper import EnvStepper
from utils.checkpoint_manager import CheckpointManager
from utils.timehelper import time_left, time_str

from learners import REGISTRY as le_REGISTRY
from controllers import REGISTRY as mac_REGISTRY, EnsembleInferenceMAC
from components.episode_buffer import ReplayBuffer
from components.transforms import OneHot
from steppers import REGISTRY as stepper_REGISTRY

MODEL_COLLECTION_BASE_PATH = "/home/pmatthaei/Projects/ma-league-results/models/"
H1_T4 = {
    "tid": 0,
    "is_scripted": False,
    "units": [  # Team 1
        {
            "uid": 0,
            "role": RoleTypes.TANK,
            "attack_type": UnitAttackTypes.RANGED
        },
        {
            "uid": 0,
            "role": RoleTypes.TANK,
            "attack_type": UnitAttackTypes.RANGED
        },
        {
            "uid": 1,
            "role": RoleTypes.HEALER,
            "attack_type": UnitAttackTypes.RANGED
        },
        {
            "uid": 0,
            "role": RoleTypes.TANK,
            "attack_type": UnitAttackTypes.RANGED
        },
        {
            "uid": 0,
            "role": RoleTypes.TANK,
            "attack_type": UnitAttackTypes.RANGED
        },
    ]
}

H1_A4 = {
    "tid": 1,
    "is_scripted": False,
    "units": [  # Team 1
        {
            "uid": 2,
            "role": RoleTypes.ADC,
            "attack_type": UnitAttackTypes.RANGED
        },
        {
            "uid": 2,
            "role": RoleTypes.ADC,
            "attack_type": UnitAttackTypes.RANGED
        },
        {
            "uid": 1,
            "role": RoleTypes.HEALER,
            "attack_type": UnitAttackTypes.RANGED
        },
        {
            "uid": 2,
            "role": RoleTypes.ADC,
            "attack_type": UnitAttackTypes.RANGED
        },
        {
            "uid": 2,
            "role": RoleTypes.ADC,
            "attack_type": UnitAttackTypes.RANGED
        },
    ]
}

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
        self.home_mac, self.away_mac = None, None

        # Init stepper so we can get env info
        self.stepper = self._build_stepper()

        # Get env info from stepper
        self.env_info = self.stepper.get_env_info()

        # Retrieve important data from the env and set in args
        shapes = self._update_shapes()

        self.logger.update_shapes(shapes)

        # Default/Base scheme- call AFTER extracting env info
        self.groups, self.preprocess, self.scheme = self._build_schemes()

        self.home_buffer = ReplayBuffer(self.scheme, self.groups, self.args.buffer_size,
                                        self.env_info["episode_limit"] + 1,
                                        preprocess=self.preprocess,
                                        device="cpu" if self.args.buffer_cpu_only else self.args.device)
        # Setup multi-agent controller here
        self.home_mac = EnsembleInferenceMAC(self.home_buffer.scheme, self.groups, self.args)
        self.away_mac = EnsembleInferenceMAC(self.home_buffer.scheme, self.groups, self.args)

        self.checkpoint_manager = CheckpointManager(args=self.args, logger=self.logger)

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

    def _build_stepper(self) -> EnvStepper:
        self.args.env_args["record"] = True
        self.args.env_args["debug_range"] = True
        self.args.env_args['match_build_plan'][0] = copy(H1_A4)
        self.args.env_args['match_build_plan'][1] = copy(H1_A4)

        self.args.env_args['match_build_plan'][0] = copy(H1_T4)
        self.args.env_args['match_build_plan'][1] = copy(H1_T4)

        self.args.env_args['match_build_plan'][0]['tid'] = 0
        self.args.env_args['match_build_plan'][1]['tid'] = 1
        self.args.env_args['match_build_plan'][1]['is_scripted'] = True
        return stepper_REGISTRY[self.args.runner](args=self.args, logger=self.logger)

    def _init_stepper(self):
        if not self.stepper.is_initalized:
            self.stepper.initialize(scheme=self.scheme, groups=self.groups, preprocess=self.preprocess,
                                    home_mac=self.home_mac)

    def load_agents(self):
        path1 = MODEL_COLLECTION_BASE_PATH + "2/qmix__2021-07-07_12-52-06_team_1/500040"
        path2 = MODEL_COLLECTION_BASE_PATH + "2/qmix__2021-07-07_12-52-06_team_0/500053"
        name = "home_qlearner_"
        state_dict1 = th.load("{}/{}agent.th".format(path1, name), map_location=lambda storage, loc: storage)
        state_dict2 = th.load("{}/{}agent.th".format(path2, name), map_location=lambda storage, loc: storage)
        # self.home_mac.load_state_dict(agent=state_dict2, ensemble={2: state_dict1})
        self.home_mac.load_state_dict(agent=state_dict2)

    def start(self, play_time_seconds=None):
        """
        :param play_time_seconds: Play the run for a certain time in seconds.
        :return:
        """
        self.logger.info("Experiment Parameters:")
        experiment_params = pprint.pformat(self.args.__dict__, indent=4, width=1)
        self.logger.info("\n\n" + experiment_params + "\n")

        self._init_stepper()

        self.load_agents()

        # start training
        self.logger.info("Beginning inference for {} episodes.".format(200))
        episode = 0

        while episode < 200:
            # Run for a whole episode at a time
            episode_batch, env_info = self.stepper.run(test_mode=True)
            self.logger.log_stat("episode", episode, self.stepper.t_env)

            episode += 1

        self.logger.log_console()

        # Finish and clean up
        self._finish()

    def _finish(self):
        self.stepper.close_env()
        self.logger.info("Finished.")
