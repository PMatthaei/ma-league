import random
from collections import defaultdict
from typing import Tuple, Dict

from torch import Tensor

from league.components.agent_pool import AgentPool
from league.components.payoff_matchmaking import MatchmakingPayoff
from league.components.self_play import OpponentSampling
from league.utils.team_composer import Team
from modules.agents import AgentNetwork


def dd():
    return 0


class Matchmaking:

    def __init__(self, agent_pool: AgentPool, payoff: Tensor, sampling_strategy: OpponentSampling = None):
        self._agent_pool = agent_pool
        self._payoff = payoff
        self._sampling_strategy = sampling_strategy

    def get_match(self, home_team: Team) -> Tuple[Team, AgentNetwork]:
        raise NotImplementedError()

    def get_ensemble(self, home_team: Team) -> Dict[int, AgentNetwork]:
        raise NotImplementedError()


class IteratingMatchmaking(Matchmaking):
    def __init__(self, agent_pool: AgentPool, payoff: Tensor = None):
        super().__init__(agent_pool, payoff)
        self.current_match = defaultdict(dd)

    def get_match(self, home_team: Team) -> Tuple[Team, AgentNetwork]:
        if self.current_match[home_team] >= len(self._agent_pool.collected_teams):
            return None

        team = self._agent_pool.collected_teams[self.current_match[home_team]]
        if team == home_team:  # If the selected team is the own jump to the next
            self.current_match[home_team] += 1
            if self.current_match[home_team] >= len(self._agent_pool.collected_teams):
                return None

        team = self._agent_pool.collected_teams[self.current_match[home_team]]
        self.current_match[home_team] += 1
        return team, self._agent_pool[team]

    def get_ensemble(self, home_team: Team) -> Dict[int, AgentNetwork]:
        pass


class RandomMatchmaking(Matchmaking):

    def __init__(self, agent_pool: AgentPool, round_limit: int = 5, payoff: Tensor = None):
        """
        Matchmaking to return a random adversary team agent bound to a round limit.
        :param agent_pool:
        :param round_limit:
        :param time_limit:
        """
        super().__init__(agent_pool, payoff)
        self.round_limit = round_limit
        self.current_round = 0

    def get_match(self, home_team: Team) -> Tuple[Team, AgentNetwork]:
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

    def get_ensemble(self, home_team: Team) -> Dict[int, AgentNetwork]:
        pass
