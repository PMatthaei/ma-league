import os
from typing import List

from exceptions.checkpoint_exceptions import NoLearnersProvided
from learners.learner import Learner
from main import results_path


class CheckpointManager:

    def __init__(self, args, logger):
        self.logger = logger
        self.args = args
        self.unique_token = args.unique_token
        self.checkpoint_path = args.checkpoint_path

    def save(self, learners: List[Learner], t_env, identifier=None) -> str:
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
        return save_path

    def load(self, learners: List[Learner], load_step) -> int:
        """
        :param learners: learners to load existing checkpoints into
        :param load_step: step at which to load
        :param checkpoint_path:
        :return:
        """
        timesteps = []
        timestep_to_load = 0

        if not os.path.isdir(self.checkpoint_path):
            self.logger.info("Checkpoint directory {} doesn't exist".format(self.checkpoint_path))
            return -1

        # Go through all files in args.checkpoint_path
        for name in os.listdir(self.checkpoint_path):
            full_name = os.path.join(self.checkpoint_path, name)
            # Check if they are dirs the names of which are numbers
            if os.path.isdir(full_name) and name.isdigit():
                timesteps.append(int(name))

        if load_step == 0:
            # choose the max timestep
            timestep_to_load = max(timesteps)
        else:
            # choose the timestep closest to load_step
            timestep_to_load = min(timesteps, key=lambda x: abs(x - load_step))

        model_path = os.path.join(self.checkpoint_path, str(timestep_to_load))

        self.logger.info("Loading model from {}".format(model_path))
        [learner.load_models(model_path) for learner in learners]

        return timestep_to_load