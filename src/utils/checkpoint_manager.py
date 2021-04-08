import os
from typing import List

from learners.learner import Learner


class CheckpointManager:

    def __init__(self, args, logger):
        self.local_results_path = args.local_results_path
        self.unique_token = args.unique_token
        self.checkpoint_path = args.checkpoint_path
        self.load_step = args.load_step
        self.logger = logger
        pass

    def save(self, t_env, learners: List[Learner]):
        save_path = os.path.join(self.local_results_path, "models", self.unique_token, str(t_env))
        os.makedirs(save_path, exist_ok=True)
        self.logger.console_logger.info("Saving models to {}".format(save_path))

        [learner.save_models(save_path, learner.name) for learner in learners]

    def load(self, learners: List[Learner]) -> int:
        timesteps = []
        timestep_to_load = 0

        if not os.path.isdir(self.checkpoint_path):
            self.logger.console_logger.info("Checkpoint directory {} doesn't exist".format(self.checkpoint_path))
            return -1

        # Go through all files in args.checkpoint_path
        for name in os.listdir(self.checkpoint_path):
            full_name = os.path.join(self.checkpoint_path, name)
            # Check if they are dirs the names of which are numbers
            if os.path.isdir(full_name) and name.isdigit():
                timesteps.append(int(name))

        if self.load_step == 0:
            # choose the max timestep
            timestep_to_load = max(timesteps)
        else:
            # choose the timestep closest to load_step
            timestep_to_load = min(timesteps, key=lambda x: abs(x - self.load_step))

        model_path = os.path.join(self.checkpoint_path, str(timestep_to_load))

        self.logger.console_logger.info("Loading model from {}".format(model_path))
        [learner.save_models(model_path, learner.name) for learner in learners]

        return timestep_to_load

