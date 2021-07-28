from multiprocessing.synchronize import Barrier
from typing import List

from torch import Tensor

from league.components import Team
from league.league import League
from league.processes.agent_pool_instance import AgentPoolInstance
from league.rolebased.simple import SimplePlayer
from utils.config_builder import ConfigBuilder


class SimpleLeague(League):

    def __init__(self, teams: List[Team], payoff: Tensor, agent_pool: AgentPoolInstance):
        super().__init__(teams, payoff, agent_pool)

    def __getitem__(self, idx: int):
        return self._matchmakers[idx]

    def setup(self, sync: Barrier, config_builder: ConfigBuilder):
        for idx, team in enumerate(self._teams): # Every team is mapped to an simple player
            matchmaker = SimplePlayer(
                idx,
                payoff=self._payoff,
                teams=teams,
                communication=self._agent_pool.register()
            )
            self._matchmakers.append(matchmaker)

    def roles_per_initial_agent(self) -> int:
        return self._main_agents_n
