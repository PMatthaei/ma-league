from typing import Tuple

from league.components.agent_pool import AgentPool
from league.components.payoff_matchmaking import MatchmakingPayoff
from league.components.self_play import OpponentSampling
from league.utils.team_composer import Team
from modules.agents import Agent


class Matchmaking:

    def __init__(self, agent_pool: AgentPool, payoff: MatchmakingPayoff = None, sampling_strategy: OpponentSampling = None):
        self._agent_pool = agent_pool
        self._payoff = payoff
        self._sampling_strategy = sampling_strategy
        pass

    def get_match(self, home_team: Team) -> Tuple[Team, Agent]:
        """
        Find a opponent for the given team using various methods.
        :param home_team:
        :return:
        """
        teams = self._agent_pool.collected_teams

        if not self._agent_pool.can_sample():
            return home_team, self._agent_pool[home_team]  # Self-Play if no one available

        return self._agent_pool.sample()
