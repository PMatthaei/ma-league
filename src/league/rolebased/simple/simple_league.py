from multiprocessing.synchronize import Barrier
from typing import List

from torch import Tensor

from league.components import Team
from league.processes import RoleBasedLeagueInstance
from league.processes.interfaces.league_experiment_process import LeagueExperimentInstance
from league.rolebased.league import League
from league.processes.agent_pool_instance import AgentPoolInstance
from league.rolebased.players import Player
from league.rolebased.simple import SimplePlayer
from utils.config_builder import ConfigBuilder


class SimpleLeague(League):

    def __init__(self, teams: List[Team], payoff: Tensor, agent_pool: AgentPoolInstance, sync, config_builder):
        super().__init__(teams, payoff, agent_pool)
        self.sync = sync
        self.config_builder = config_builder
        self.matchmakers: List[Player] = []

    def __getitem__(self, idx: int):
        return self._matchmakers[idx]

    def start(self) -> List[LeagueExperimentInstance]:
        #
        # Start experiment instances
        #
        procs = []  # All running processes representing an agent playing in the league
        for idx, team in enumerate(self._teams):
            player = SimplePlayer(
                idx,
                payoff=self._payoff,
                teams=self._teams,
                communication=self._agent_pool.register()
            )
            proc = RoleBasedLeagueInstance(
                idx=idx,
                experiment_config=self.config_builder.build(idx),
                home_team=team,  # TODO replace with teams[idx] from matchmaker
                matchmaker=player,
                communication=self._agent_pool.register(),
                sync_barrier=self.sync
            )
            procs.append(proc)
            self.matchmakers.append(player)

        self._agent_pool.start()

        [p.start() for p in procs]

        return procs

    def disconnect(self):
        [matchmaker.disconnect() for matchmaker in self.matchmakers]

    def roles_per_initial_agent(self) -> int:
        return self._main_agents_n
