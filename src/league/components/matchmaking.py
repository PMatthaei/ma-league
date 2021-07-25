from typing import Tuple, Dict, Union, List

import torch as th
from torch import Tensor

from league.components import PayoffEntry
from league.components.agent_pool import AgentPool
from league.components.self_play import OpponentSampling
from league.utils.team_composer import Team
from modules.agents import AgentNetwork


def dd():
    return 0


class Matchmaking:

    def __init__(self, agent_pool: AgentPool, teams: List[Team], payoff: Tensor, allocation: Dict[int, int],
                 sampling_strategy: OpponentSampling = None):
        self._agent_pool = agent_pool
        self._allocation = allocation
        self._payoff = payoff
        self._teams = teams
        self._sampling_strategy = sampling_strategy

    def get_instance_id(self, team: Team) -> int:
        return self._allocation[team.id_]

    def get_team(self, instance_id: int) -> Team:
        _inv_allocation = {v: k for k, v in self._allocation.items()}
        tid = _inv_allocation[instance_id]
        return next((team for team in self._teams if team.id_ == tid), None)

    def get_match(self, home_team: Team) -> Union[None, Tuple[Team, AgentNetwork]]:
        raise NotImplementedError()


class PlayEvenlyMatchmaking(Matchmaking):
    def __init__(self, agent_pool: AgentPool, allocation: Dict[int, int], payoff: Tensor, teams: List[Team]):
        super().__init__(agent_pool, teams, payoff, allocation)

    def get_match(self, home_team: Team) -> Union[None, Tuple[Team, AgentNetwork]]:
        idx = self.get_instance_id(home_team)
        games = self._payoff[idx, :, PayoffEntry.GAMES]
        # games = games[games != idx] # Remove play against one self
        match_idx = th.argmin(games).item()  # Get adversary we played the least
        team = self.get_team(match_idx)
        return team, self._agent_pool[team]


class RandomMatchmaking(Matchmaking):

    def __init__(self, agent_pool: AgentPool, allocation: Dict[int, int], teams: List[Team], round_limit: int = 5,
                 payoff: Tensor = None):
        """
        Matchmaking to return a random adversary team agent bound to a round limit.
        :param agent_pool:
        :param round_limit:
        :param time_limit:
        """
        super().__init__(agent_pool, teams, payoff, allocation)
        self.round_limit = round_limit
        self.current_round = 0

    def get_match(self, home_team: Team) -> Union[None, Tuple[Team, AgentNetwork]]:
        """
        Find a opponent for the given team using various methods.
        :param home_team:
        :return:
        """
        if self.current_round >= self.round_limit:
            return None

        if not self._agent_pool.can_sample():
            return home_team, self._agent_pool[home_team]  # Self-Play if no one available
        self.current_round += 1
        return self._agent_pool.sample()
