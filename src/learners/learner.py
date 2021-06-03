from components.episode_buffer import EpisodeBatch
from controllers.multi_agent_controller import MultiAgentController


class Learner:
    def __init__(self, mac: MultiAgentController, scheme, logger, args, name=None):
        self.mac = mac
        self.scheme = scheme
        self.logger = logger
        self.args = args
        self.name = "" if name is None else name

    def train(self, batch: EpisodeBatch, t_env: int, episode_num: int) -> None:
        """
        Define training procedure based on a batch of episode data, the current timestep in the environment
        and the current episode number.
        :param batch:
        :param t_env:
        :param episode_num:
        :return:
        """
        raise NotImplementedError()

    def cuda(self) -> None:
        """
        Move all components to GPU.
        :return:
        """
        raise NotImplementedError()

    def save_models(self, path, name):
        """
        Define to save a learner model. Consider all required components and parameters.
        :param path:
        :param name:
        :return:
        """
        raise NotImplementedError()

    def load_models(self, path):
        """
        Define to load a learner model. Consider all required components and parameters.
        :param path:
        :return:
        """
        raise NotImplementedError()
