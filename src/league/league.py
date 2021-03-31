from typing import Tuple

from league.roles.agent import Agent
from league.roles.exploiters import MainExploiter, LeagueExploiter
from league.roles.players import MainPlayer, Player


class League(object):

    def __init__(self,
                 initial_agents,
                 main_agents_n=1,
                 main_exploiters_n=1,
                 league_exploiters_n=2,
                 payoff=None):
        self._payoff = payoff
        self._learning_agents = {}
        self._main_agents_n = main_agents_n
        self._main_exploiters_n = main_exploiters_n
        self._league_exploiters_n = league_exploiters_n

        player_id = 0
        # Setup initial learning agents
        for i, plan in enumerate(initial_agents):
            # TODO: how to build a set of initial agents so that initial_agents[plan] makes sense?
            for _ in range(self._main_agents_n):
                main_agent = MainPlayer(player_id, team_plan=plan, agent=Agent(plan, 0), payoff=self._payoff)
                self._learning_agents[player_id] = main_agent
                self._payoff.add_player(main_agent.checkpoint())
                player_id += 1

            for _ in range(self._main_exploiters_n):
                exploiter = MainExploiter(player_id, team_plan=plan, agent=Agent(plan, 0), payoff=self._payoff)
                self._learning_agents[player_id] = exploiter
                player_id += 1

            for _ in range(self._league_exploiters_n):
                league_exploiter = LeagueExploiter(player_id, team_plan=plan, agent=Agent(plan, 0), payoff=self._payoff)
                self._learning_agents[player_id] = league_exploiter
                player_id += 1

        for player_id, player in self._learning_agents.items():
            self._payoff.add_player(player)

    def roles_per_initial_agent(self):
        return self._main_agents_n + self._main_exploiters_n + self._league_exploiters_n

    def update(self, home: int, away: int, result: str) -> Tuple[Player, Player]:
        return self._payoff.update(home, away, result)

    def get_player(self, idx) -> Player:
        return self._learning_agents[idx]

    def add_player(self, player: Player):
        self._payoff.add_player(player)
