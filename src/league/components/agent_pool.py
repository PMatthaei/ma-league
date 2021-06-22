import random
from typing import Dict, Tuple

from league.utils.team_composer import Team
from modules.agents import Agent


class AgentPool:

    def __init__(self, agents_dict: Dict[Team, Agent]):
        self._agents_dict = agents_dict

    def __setitem__(self, team: Team, value: Agent):
        self._agents_dict[team] = value

    def __getitem__(self, team: Team):
        return self._agents_dict[team]

    def can_sample(self) -> bool:
        return len(self._agents_dict) > 0

    def sample(self) -> Tuple[Team, Agent]:
        return random.choice(self._agents_dict.items())

    def get_neighbors(self, team):
        # TODO: get agents of teams which have a similar constellation
        pass