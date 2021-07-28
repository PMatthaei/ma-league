from torch.optim import RMSprop

from components.episode_batch import EpisodeBatch
from controllers.multi_agent_controller import MultiAgentController


class Learner:
    def __init__(self, mac: MultiAgentController, scheme, logger, args, name=None):
        """
        Learners update parameters and networks provided via the Multi-Agent Controller
        :param mac:
        :param scheme:
        :param logger:
        :param args:
        :param name:
        """
        self.mac = mac
        self.scheme = scheme
        self.logger = logger
        self.args = args
        self.name = f'{"" if name is None else name}_{self.__class__.__name__.lower()}_'
        self.log_stats_t = -self.args.learner_log_interval - 1
        # Receive params from the agent from Multi-Agent Controller
        self.params = list(mac.parameters())
        self.optimiser = None

    def build_optimizer(self):
        self.optimiser = RMSprop(params=self.params, lr=self.args.lr, alpha=self.args.optim_alpha,
                                 eps=self.args.optim_eps)

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
