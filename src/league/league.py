from typing import Tuple

from league.roles.exploiters import MainExploiter, LeagueExploiter
from league.roles.players import MainPlayer, Player
from learners.learner import Learner


class League(object):

    def __init__(self,
                 initial_agents,
                 payoff,
                 main_agents_n=1,
                 main_exploiters_n=1,
                 league_exploiters_n=2):
        self._payoff = payoff
        self._learning_agents = {}
        self._main_agents_n = main_agents_n
        self._main_exploiters_n = main_exploiters_n
        self._league_exploiters_n = league_exploiters_n

        player_id = 0
        # Setup initial learning agents
        for i, plan in enumerate(initial_agents):
            for _ in range(self._main_agents_n):
                main_agent = MainPlayer(player_id, payoff=self._payoff)
                self._learning_agents[player_id] = main_agent
                player_id += 1

            for _ in range(self._main_exploiters_n):
                exploiter = MainExploiter(player_id, payoff=self._payoff)
                self._learning_agents[player_id] = exploiter
                player_id += 1

            for _ in range(self._league_exploiters_n):
                league_exploiter = LeagueExploiter(player_id, payoff=self._payoff)
                self._learning_agents[player_id] = league_exploiter
                player_id += 1

        for player in self._learning_agents.values():
            self._payoff.add_player(player)

    def roles_per_initial_agent(self) -> int:
        return self._main_agents_n + self._main_exploiters_n + self._league_exploiters_n

    def update(self, home: int, away: int, result: str) -> Tuple[Player, Player]:
        return self._payoff.update(home, away, result)

    def get_player(self, idx: int) -> Player:
        return self._learning_agents[idx]

    def add_player(self, player: Player):
        self._payoff.add_player(player)

    def provide_learner(self, player_id: int, learner: Learner):
        self._payoff.get_player(player_id).learner = learner

    def print_payoff(self):
        player_ids = list(range(self.size))
        for player_id in player_ids:
            print(f"Win rates for player {player_id}:")
            print(self._payoff[player_id, player_ids])

    @property
    def size(self) -> int:
        return len(self._learning_agents)
