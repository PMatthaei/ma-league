from league.league import League
from league.roles.simple.simple_player import SimplePlayer


class SimpleLeague(League):

    def __init__(self, initial_agents, payoff, main_agents_n=2):
        """
        Simple Example League
        :param initial_agents:
        :param payoff:
        :param main_agents_n:
        """
        super().__init__(initial_agents, payoff, main_agents_n)

    def _setup(self, initial_agents):
        player_id = 0
        for i, plan in enumerate(initial_agents):
            for _ in range(self._main_agents_n):
                main_agent = SimplePlayer(player_id, payoff=self._payoff, team=plan)
                self._learning_agents[player_id] = main_agent
                player_id += 1

        for player in self._learning_agents.values():
            self._payoff.add_player(player)

    def roles_per_initial_agent(self) -> int:
        return self._main_agents_n

