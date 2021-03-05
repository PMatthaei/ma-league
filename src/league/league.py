from league.payoff import Payoff
from league.roles.exploiters import MainExploiter, LeagueExploiter
from league.roles.players import MainPlayer, Player


class Agent(object):
    """Demonstrates agent interface.

    In practice, this needs to be instantiated with the right neural network
    architecture.
    """

    def __init__(self, team_plan, initial_weights):
        self.team_plan = team_plan
        self.steps = 0
        self.weights = initial_weights

    def initial_state(self):
        """Returns the hidden state of the agent for the start of an episode."""
        # Network details elided.
        return initial_state

    def set_weights(self, weights):
        self.weights = weights

    def get_steps(self):
        """How many agent steps the agent has been trained for."""
        return self.steps

    def step(self, observation, last_state):
        """Performs inference on the observation, given hidden state last_state."""
        # We are omitting the details of network inference here.
        # ...
        return action, policy_logits, new_state

    def unroll(self, trajectory):
        """Unrolls the network over the trajectory.

        The actions taken by the agent and the initial state of the unroll are
        dictated by trajectory.
        """
        # We omit the details of network inference here.
        return policy_logits, baselines


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
        for plan in initial_agents:
            # TODO: how to build a set of initial agents so that initial_agents[plan] makes sense?
            for _ in range(self._main_agents_n):
                main_agent = MainPlayer(team_plan=plan, agent=initial_agents[plan], payoff=self._payoff)
                self._learning_agents.append(main_agent)
                self._payoff.add_player(main_agent.checkpoint())

            for _ in range(self._main_exploiters_n):
                exploiter = MainExploiter(team_plan=plan, agent=initial_agents[plan], payoff=self._payoff)
                self._learning_agents.append(exploiter)
            for _ in range(self._league_exploiters_n):
                league_exploiter = LeagueExploiter(team_plan=plan, agent=initial_agents[plan], payoff=self._payoff)
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
