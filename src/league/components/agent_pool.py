import random
from typing import Dict, Tuple, List, ItemsView

from league.utils.team_composer import Team
from modules.agents import AgentNetwork


class AgentPool:

    def __init__(self, agents_dict: Dict[Team, AgentNetwork]):
        """
        Manages the current set of trained multi-agents which themselves are linked to their underlying team.
        :param agents_dict:
        """
        self._agents_dict = agents_dict

    def __setitem__(self, team: Team, value: AgentNetwork):
        self._agents_dict[team] = value

    def __getitem__(self, team: Team) -> AgentNetwork:
        return self._agents_dict[team] # ! WARN ! Does not work on CUDA and multiprocessing

    def as_list(self):
        return self._agents_dict.items()

    @property
    def teams(self) -> List[Team]:
        """
        :return: List of teams for which an agent exists in the pool
        """
        return list(self._agents_dict.keys())

    @property
    def agents(self) -> List[AgentNetwork]:
        """
        :return: List of teams for which an agent exists in the pool
        """
        return list(self._agents_dict.values())

    def sample(self) -> Tuple[Team, AgentNetwork]:
        """
        :return: Sample a random (team, agent) tuple
        """
        return random.choice(self._agents_dict.items())

    def can_sample(self) -> bool:
        return len(self._agents_dict) > 0
