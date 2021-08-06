import pprint
from copy import copy
from types import SimpleNamespace
from typing import Dict

import torch as th

from marl.components.replay_buffers import ReplayBuffer
from runs.experiment_run import ExperimentRun
from runs.train.ma_experiment import MultiAgentExperiment
from steppers.episode_stepper import EnvStepper

from marl.controllers import REGISTRY as mac_REGISTRY

from marl.components.transforms import OneHot
from steppers import REGISTRY as stepper_REGISTRY, SelfPlayStepper

# Config TODO: Pack into args
from utils.asset_manager import AssetManager

POLICY_TEAM_1 = "/home/pmatthaei/Projects/ma-league-results/saba/results/league_2021-08-05_19-02-10/instance_0/models/qmix"
POLICY_TEAM_1_ID = 0
POLICY_TEAM_1_STEP = 1250140

#SELF_PLAY = False
SELF_PLAY = True

POLICY_TEAM_2 = "/home/pmatthaei/Projects/ma-league-results/saba/results/league_2021-08-05_19-02-10/instance_1/models/qmix"
POLICY_TEAM_2_STEP = 1750135


class ReplayGenerationRun(MultiAgentExperiment):

    def __init__(self, args, logger):
        """
        Load a saved model into the Multi-Agent Controller and perform inference for a given amount of time.
        Episodes run within this time are captured as video for later replay and visual understanding of the loaded
        model/policy.
        :param args:
        :param logger:
        """
        super().__init__(args, logger)

    def _integrate_env_info(self):
        total_n_agents = self.env_info["n_agents"]
        # Since the AI is replaced with fixed policy adversary agents, we need to re-calculate
        # the amount of the learning agents for the scheme
        if SELF_PLAY:
            assert total_n_agents % 2 == 0, f"A total of {total_n_agents} agents in the env do not fit in the symmetric two-team scenario. " \
                                            f"Ensure the Self-Play scenario has two team set to is_scripted=False"
            per_team_n_agents = int(total_n_agents / 2)
        env_scheme = {
            "n_agents": per_team_n_agents if SELF_PLAY else total_n_agents,
            "n_actions": int(self.env_info["n_actions"]),
            "state_shape": int(self.env_info["state_shape"]),
            "total_n_agents": total_n_agents
        }
        self._update_args(env_scheme)
        return env_scheme

    def _build_learners(self):
        super(ReplayGenerationRun, self)._build_learners()
        if SELF_PLAY:
            self.away_mac = mac_REGISTRY[self.args.mac](self.home_buffer.scheme, self.groups, self.args)

    def _build_stepper(self, log_start_t: int = 0) -> EnvStepper:
        self._configure_environment_args()
        if SELF_PLAY:
            return SelfPlayStepper(args=self.args, logger=self.logger)
        return stepper_REGISTRY[self.args.runner](args=self.args, logger=self.logger)

    def _configure_environment_args(self):
        policy_team_id = POLICY_TEAM_1_ID
        other_team_id = 1 - policy_team_id

        self.args.env_args["headless"] = False
        self.args.env_args["debug_range"] = True
        self.args.env_args["stochastic_spawns"] = True
        home_team = self.asset_manager.load_team(path=POLICY_TEAM_1)
        self.args.env_args['match_build_plan'][policy_team_id] = home_team
        if SELF_PLAY:
            away_team = self.asset_manager.load_team(path=POLICY_TEAM_2)
            self.args.env_args['match_build_plan'][other_team_id] = away_team
        else:
            self.args.env_args['match_build_plan'][other_team_id] = copy(home_team)
        self.args.env_args['match_build_plan'][policy_team_id]['tid'] = policy_team_id
        self.args.env_args['match_build_plan'][other_team_id]['tid'] = other_team_id
        self.args.env_args['match_build_plan'][other_team_id]['is_scripted'] = not SELF_PLAY

    def _init_stepper(self):
        if not self.stepper.is_initalized:
            if isinstance(self.stepper, SelfPlayStepper):
                self.stepper.initialize(
                    scheme=self.scheme,
                    groups=self.groups,
                    preprocess=self.preprocess,
                    home_mac=self.home_mac,
                    away_mac=self.away_mac
                )
            else:
                self.stepper.initialize(
                    scheme=self.scheme,
                    groups=self.groups,
                    preprocess=self.preprocess,
                    home_mac=self.home_mac
                )

    def load_agents(self):
        state = self.asset_manager.load_state(path=POLICY_TEAM_1, component="agent", load_step=POLICY_TEAM_1_STEP)
        self.home_mac.load_state_dict(agent=state)

        if SELF_PLAY:
            state = self.asset_manager.load_state(path=POLICY_TEAM_2, component="agent", load_step=POLICY_TEAM_1_STEP)
            self.away_mac.load_state_dict(agent=state)

    def start(self, play_time_seconds=None, on_train=None) -> int:
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
            self.stepper.run(test_mode=True)
            self.logger.log_stat("episode", episode, self.stepper.t_env)

            episode += 1

        self.logger.log_report()

        # Finish and clean up
        self._finish()

    def _finish(self):
        self.stepper.close_env()
        self.logger.info("Finished.")

    def _train_episode(self, episode_num):
        pass  # ... we do not train, just infer
