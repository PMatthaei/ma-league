from typing import List, Tuple

from torch import Tensor
from torch.multiprocessing.queue import Queue

from league.league import League
from league.rolebased.simple import SimplePlayer


class SimpleLeague(League):

    def __init__(self, teams, payoff: Tensor, communication: Tuple[int, Tuple[Queue, Queue]]):
        super().__init__(teams, payoff, communication)

    def __getitem__(self, idx: int):
        return self._learning_agents[idx]

    def _setup(self, initial_agents):
        player_id = 0
        for _, team_plan in enumerate(initial_agents):
            for _ in range(self._main_agents_n):
                main_agent = SimplePlayer(player_id, payoff=self._payoff, teams=team_plan)
                self._learning_agents[player_id] = main_agent
                player_id += 1

    def roles_per_initial_agent(self) -> int:
        return self._main_agents_n
