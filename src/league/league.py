from typing import Tuple

from league.roles.players import Player


class League(object):

    def __init__(self,
                 initial_agents,
                 payoff,
                 main_agents_n=1):
        self._payoff = payoff
        self._learning_agents = {}
        self._main_agents_n = main_agents_n

        # Setup initial learning agents
        self._setup(initial_agents)

    def _setup(self, initial_agents):
        raise NotImplementedError()

    def roles_per_initial_agent(self) -> int:
        raise NotImplementedError()

    def update(self, home: int, away: int, result: str) -> Tuple[Player, Player]:
        return self._payoff.update(home, away, result)

    def get_player(self, idx: int) -> Player:
        return self._payoff.get_player(idx)

    def add_player(self, player: Player):
        self._payoff.add_player(player)

    @property
    def size(self) -> int:
        return len(self._learning_agents)
