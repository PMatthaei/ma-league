from components.episode_buffer import EpisodeBatch


class Learner:
    def __init__(self, mac=None, scheme=None, logger=None, args=None, name=None):
        self.mac = mac
        self.scheme = scheme
        self.logger = logger
        self.args = args
        self.name = "" if name is None else name

    def train(self, batch: EpisodeBatch, t_env: int, episode_num: int):
        raise NotImplementedError()

    def cuda(self):
        raise NotImplementedError()

    def save_models(self, path, name):
        raise NotImplementedError()

    def load_models(self, path, name):
        raise NotImplementedError()

    def get_current_step(self) -> int:
        raise NotImplementedError()
