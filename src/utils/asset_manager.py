import os
from typing import List, OrderedDict, Dict, Union

from maenv.utils.enums import EnumEncoder, as_enum

from exceptions.checkpoint_exceptions import NoLearnersProvided
from league.utils.team_composer import Team
from learners.learner import Learner
from main import results_path
from utils.run_utils import find_latest_model_path


class AssetManager:

    def __init__(self, args, logger):
        """
        Saves and loads various assets needed for league or normal training such as loading models or team constellations
        from files.
        :param args:
        :param logger:
        """
        self.logger = logger
        self.args = args
        self.unique_token = args.unique_token
        self.checkpoint_path = args.checkpoint_path

    def save_learner(self, learners: List[Learner], t_env, identifier=None) -> str:
        """
        :param identifier:
        :param learners: list of learners to save
        :param t_env: timestep at which this save was issued
        :return:
        """
        if not learners:
            raise NoLearnersProvided()
        identifier = identifier if identifier is not None else ""
        save_path = os.path.join(results_path, "models", self.unique_token, identifier, str(t_env))
        os.makedirs(save_path, exist_ok=True)
        # Save all provided learners in the same path
        [learner.save_models(save_path, learner.name) for learner in learners]
        self.logger.info("Saving learner components to {}".format(save_path))

        return save_path

    def load_learner(self, learners: List[Learner], load_step) -> int:
        """
        :param learners: learners to load existing checkpoints into
        :param load_step: step at which to load
        :param checkpoint_path:
        :return:
        """

        if not os.path.isdir(self.checkpoint_path):
            self.logger.info("Checkpoint directory {} doesn't exist".format(self.checkpoint_path))
            return -1

        model_path, timestep_to_load = find_latest_model_path(path=self.checkpoint_path, load_step=load_step)

        self.logger.info("Loading learner components from {}".format(model_path))
        [learner.load_models(model_path) for learner in learners]

        return timestep_to_load

    def load_state(self, path: str, component: str) -> OrderedDict:
        """
        Loads a network state dict from a .th file.
        :param path:
        :param component:
        :return:
        """
        import torch as th
        latest, _ = find_latest_model_path(path)
        name = "home_qlearner_" # TODO Adapt if more learners used
        return th.load(f"{latest}/{name}{component}.th", map_location=lambda storage, loc: storage)

    def load_team(self, path: str, as_team=False) -> Union[Dict, Team]:
        """
        Loads a team constellation from a *_team.json file
        :param as_team: convert return to Team class
        :param path:
        :return:
        """
        import glob
        import json
        for file_path in glob.glob(f'{path}/*team.json'):
            with open(file_path, 'r') as f:
                plan = json.load(fp=f, object_hook=as_enum)
                f.close()
                return plan if not as_team else Team(**plan)
