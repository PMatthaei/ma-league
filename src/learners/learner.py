from components.episode_buffer import EpisodeBatch


class Learner:

    def train(self, batch: EpisodeBatch, t_env: int, episode_num: int):
        raise NotImplementedError()

    def cuda(self):
        raise NotImplementedError()

    def save_models(self, path):
        raise NotImplementedError()

    def load_models(self, path):
        raise NotImplementedError()

    def get_current_step(self) -> int:
        raise NotImplementedError()
