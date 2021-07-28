from multiprocessing.synchronize import Barrier

from torch import Tensor
from typing import  List

from league.components import Team
from league.processes.agent_pool_instance import AgentPoolInstance
from utils.config_builder import ConfigBuilder


class League(object):

    def __init__(self,
                 teams: List[Team],
                 payoff: Tensor,
                 agent_pool: AgentPoolInstance,
                 main_agents_n=1):
        self._teams = teams
        self._payoff = payoff
        self._matchmakers = {}
        self._main_agents_n = main_agents_n
        self._agent_pool = agent_pool

    def __getitem__(self, item):
        raise NotImplementedError()

    def setup(self, sync: Barrier, config_builder: ConfigBuilder):
        raise NotImplementedError()

    def roles_per_initial_agent(self) -> int:
        raise NotImplementedError()

    @property
    def size(self) -> int:
        return len(self._matchmakers)
