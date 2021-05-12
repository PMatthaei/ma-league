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
        raise NotImplementedError()

    def cuda(self) -> None:
        raise NotImplementedError()

    def save_models(self, path, name):
        raise NotImplementedError()

    def load_models(self, path):
        raise NotImplementedError()

    def get_current_step(self) -> int:
        return 0  # TODO
