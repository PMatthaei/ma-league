import os
from typing import List

from exceptions.checkpoint_exceptions import NoLearnersProvided
from learners.learner import Learner
from main import results_path


class CheckpointManager:

    def __init__(self, args, logger):
        self.unique_token = args.unique_token
        self.checkpoint_path = args.checkpoint_path
        self.load_step = args.load_step
        self.logger = logger
        pass

    def save(self, t_env, learners: List[Learner]) -> str:
        if len(learners) == 0:
            raise NoLearnersProvided()
        save_path = os.path.join(results_path, "models", self.unique_token, str(t_env))
        os.makedirs(save_path, exist_ok=True)
        # Save all provided learners in the same path
        [learner.save_models(save_path, learner.name) for learner in learners]
        return save_path

    def load(self, learners: List[Learner], checkpoint_path=None) -> int:
        path = checkpoint_path if checkpoint_path is not None else self.checkpoint_path
        timesteps = []
        timestep_to_load = 0

        if not os.path.isdir(path):
            self.logger.console_logger.info("Checkpoint directory {} doesn't exist".format(path))
            return -1

        # Go through all files in args.checkpoint_path
        for name in os.listdir(path):
            full_name = os.path.join(path, name)
            # Check if they are dirs the names of which are numbers
            if os.path.isdir(full_name) and name.isdigit():
                timesteps.append(int(name))

        if self.load_step == 0:
            # choose the max timestep
            timestep_to_load = max(timesteps)
        else:
            # choose the timestep closest to load_step
            timestep_to_load = min(timesteps, key=lambda x: abs(x - self.load_step))

        model_path = os.path.join(path, str(timestep_to_load))

        self.logger.console_logger.info("Loading model from {}".format(model_path))
        [learner.load_models(model_path, learner.name) for learner in learners]

        return timestep_to_load

