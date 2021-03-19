from league.payoff import Payoff
from league.roles.agent import Agent
from league.roles.exploiters import MainExploiter, LeagueExploiter
from league.roles.players import MainPlayer


class League(object):

    def __init__(self,
                 initial_agents,
                 main_agents_n=1,
                 main_exploiters_n=1,
                 league_exploiters_n=2):
        self._payoff = Payoff()
        self._learning_agents = []
        self._main_agents_n = main_agents_n
        self._main_exploiters_n = main_exploiters_n
        self._league_exploiters_n = league_exploiters_n
        # Setup initial learning agents
        for i, plan in enumerate(initial_agents):
            # TODO: how to build a set of initial agents so that initial_agents[plan] makes sense?
            for _ in range(self._main_agents_n):
                main_agent = MainPlayer(team_plan=plan, agent=Agent(plan, 0), payoff=self._payoff)
                self._learning_agents.append(main_agent)
                self._payoff.add_player(main_agent.checkpoint())

            for _ in range(self._main_exploiters_n):
                exploiter = MainExploiter(team_plan=plan, agent=Agent(plan, 0), payoff=self._payoff)
                self._learning_agents.append(exploiter)

            for _ in range(self._league_exploiters_n):
                league_exploiter = LeagueExploiter(team_plan=plan, agent=Agent(plan, 0), payoff=self._payoff)
                self._learning_agents.append(league_exploiter)

        for player in self._learning_agents:
            self._payoff.add_player(player)

    def roles_per_initial_agent(self):
        return self._main_agents_n + self._main_exploiters_n + self._league_exploiters_n

    def update(self, home, away, result):
        return self._payoff.update(home, away, result)

    def get_player(self, idx):
        return self._learning_agents[idx]

    def add_player(self, player):
        self._payoff.add_player(player)
