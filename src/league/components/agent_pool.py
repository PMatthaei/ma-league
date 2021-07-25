import random
from typing import Dict, Tuple, List, ItemsView, OrderedDict

from league.utils.team_composer import Team
from modules.agents import AgentNetwork


class AgentPool:

    def __init__(self, shared_storage: Dict[Team, OrderedDict]):
        """
        Manages the current set of trained multi-agents which themselves are linked to their underlying team.
        Only parameters of agents are stored to prevent storing the same network architecture redundantly.
        ! WARN ! This will only work as long as all agent networks share the same architecture.
        :param shared_storage:
        """
        self._shared_storage = shared_storage

    def __setitem__(self, team: Team, value: OrderedDict):
        self._shared_storage[team] = value

    def __getitem__(self, team: Team) -> OrderedDict:
        return self._shared_storage[team] # ! WARN ! Does not work on CUDA and multiprocessing

    def as_list(self):
        return self._shared_storage.items()

    @property
    def teams(self) -> List[Team]:
        """
        :return: List of teams for which an agent exists in the pool
        """
        return list(self._shared_storage.keys())

    @property
    def agents(self) -> List[OrderedDict]:
        """
        :return: List of teams for which an agent exists in the pool
        """
        return list(self._shared_storage.values())

    def sample(self) -> Tuple[Team, OrderedDict]:
        """
        :return: Sample a random (team, agent) tuple
        """
        return random.choice(self._shared_storage.items())

    def can_sample(self) -> bool:
        return len(self._shared_storage) > 0
