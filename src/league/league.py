from torch import Tensor
from typing import Tuple, List

from torch.multiprocessing.queue import Queue

from league.components import Team


class League(object):

    def __init__(self,
                 teams: List[Team],
                 payoff: Tensor,
                 communication: Tuple[int, Tuple[Queue, Queue]],
                 main_agents_n=1):
        self._payoff = payoff
        self._learning_agents = {}
        self._main_agents_n = main_agents_n
        self._comm = communication

        # Setup initial learning agents
        self._setup(teams)

    def __getitem__(self, item):
        raise NotImplementedError()

    def _setup(self, initial_agents):
        raise NotImplementedError()

    def roles_per_initial_agent(self) -> int:
        raise NotImplementedError()

    @property
    def size(self) -> int:
        return len(self._learning_agents)
