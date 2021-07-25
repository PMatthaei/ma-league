from types import SimpleNamespace
from typing import Dict


class ExperimentRun:
    def __init__(self, args, logger):
        """
        Interface for all experiment runs. An experiment run describes the training of one or more multi-agents.
        When inheriting this class the subclass needs to handle:
        1. Setup of one or more learners.
        2. Setup of the environment stepper
        3. Define how training is performed
        4. Define how the experiment should start and finish.
        :param args:
        :param logger:
        """
        self.args = args
        self.logger = logger

    def _build_learners(self):
        raise NotImplementedError()

    def _build_stepper(self):
        raise NotImplementedError()

    def _init_stepper(self):
        raise NotImplementedError()

    def _train_episode(self, episode_num):
        raise NotImplementedError()

    def start(self, play_time=None):
        raise NotImplementedError()

    def _finish(self):
        raise NotImplementedError()

    def _integrate_env_info(self):
        raise NotImplementedError()

    def _update_args(self, update: Dict):
        self.args = SimpleNamespace(**{**vars(self.args), **update})
