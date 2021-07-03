import random
from typing import Dict, Tuple, List

from league.utils.team_composer import Team
from modules.agents import Agent


class AgentPool:

    def __init__(self, agents_dict: Dict[Team, Agent]):
        """
        Manages the current set of trained multi-agents which themselves are linked to their underlying team.
        :param agents_dict:
        """
        self._agents_dict = agents_dict

    def __setitem__(self, team: Team, value: Agent):
        self._agents_dict[team] = value

    def __getitem__(self, team: Team):
        return self._agents_dict[team]

    @property
    def collected_teams(self) -> List[Team]:
        """
        :return: List of teams for which an agent exists in the pool
        """
        return list(self._agents_dict.keys())

    def sample(self) -> Tuple[Team, Agent]:
        """
        :return: Sample a random (team, agent) tuple
        """
        return random.choice(self._agents_dict.items())

    def can_sample(self) -> bool:
        return len(self._agents_dict) > 0
