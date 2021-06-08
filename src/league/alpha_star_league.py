from league.league import League
from league.roles.alphastar.exploiters import MainExploiter, LeagueExploiter
from league.roles.alphastar.main_player import MainPlayer


class AlphaStarLeague(League):

    def __init__(self, initial_agents, payoff, main_agents_n=1, main_exploiters_n=1, league_exploiters_n=2):
        self._main_exploiters_n = main_exploiters_n
        self._league_exploiters_n = league_exploiters_n
        super().__init__(initial_agents, payoff, main_agents_n)

    def _setup(self, initial_agents):
        player_id = 0
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

