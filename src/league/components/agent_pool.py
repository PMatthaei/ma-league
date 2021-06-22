from typing import Dict, List

from modules.agents import Agent


class AgentPool:

    def __init__(self, agents_dict: Dict[int, List[Agent]]):
        self._agents_dict = agents_dict

    def __getitem__(self, team_id: int):
        return self._agents_dict[team_id]

    def get_neighbors(self, team):
        # TODO: get agents of teams which have a similar constellation
        pass