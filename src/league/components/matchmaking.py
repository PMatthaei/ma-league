from league.components.agent_pool import AgentPool
from league.components.payoff_v2 import PayoffV2
from league.components.self_play import OpponentSampling
from league.utils.team_composer import Team


class Matchmaking:

    def __init__(self, agent_pool: AgentPool, payoff: PayoffV2 = None, sampling_strategy: OpponentSampling = None):
        self._agent_pool = agent_pool
        self._payoff = payoff
        self._sampling_strategy = sampling_strategy
        pass

    def get_match(self, team: Team):
        """
        Find a opponent for the given team using various methods.
        :param team:
        :return:
        """
        if not self._agent_pool.can_sample():
            return team  # Self-Play if no one available

        return self._agent_pool.sample()
