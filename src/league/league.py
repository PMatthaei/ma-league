"""Library for multiagent concerns."""
import collections

import numpy as np

from league.roles.exploiters import MainExploiter, LeagueExploiter
from league.roles.players import MainPlayer, Player


class Agent(object):
    """Demonstrates agent interface.

    In practice, this needs to be instantiated with the right neural network
    architecture.
    """

    def __init__(self, race, initial_weights):
        self.race = race # TODO: replace race with team composition he resides in?
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


class Payoff:

    def __init__(self):
        self._players = []
        self._wins = collections.defaultdict(lambda: 0)
        self._draws = collections.defaultdict(lambda: 0)
        self._losses = collections.defaultdict(lambda: 0)
        self._games = collections.defaultdict(lambda: 0)
        self._decay = 0.99

    def _win_rate(self, _home, _away):
        if self._games[_home, _away] == 0:
            return 0.5

        return (self._wins[_home, _away] +
                0.5 * self._draws[_home, _away]) / self._games[_home, _away]

    def __getitem__(self, match):
        home, away = match

        if isinstance(home, Player):
            home = [home]
        if isinstance(away, Player):
            away = [away]

        win_rates = np.array([[self._win_rate(h, a) for a in away] for h in home])
        if win_rates.shape[0] == 1 or win_rates.shape[1] == 1:
            win_rates = win_rates.reshape(-1)

        return win_rates

    def update(self, home, away, result):
        for stats in (self._games, self._wins, self._draws, self._losses):
            stats[home, away] *= self._decay
            stats[away, home] *= self._decay

        self._games[home, away] += 1
        self._games[away, home] += 1
        if result == "win":
            self._wins[home, away] += 1
            self._losses[away, home] += 1
        elif result == "draw":
            self._draws[home, away] += 1
            self._draws[away, home] += 1
        else:
            self._wins[away, home] += 1
            self._losses[home, away] += 1

    def add_player(self, player):
        self._players.append(player)

    @property
    def players(self):
        return self._players


class League(object):

    def __init__(self,
                 initial_agents,
                 main_agents=1,
                 main_exploiters=1,
                 league_exploiters=2):
        self._payoff = Payoff()
        self._learning_agents = []
        # TODO: This will change since we do not consider races but team compositions in terms of agent/unit selection
        for race in initial_agents:
            for _ in range(main_agents):
                main_agent = MainPlayer(race, initial_agents[race], self._payoff)
                self._learning_agents.append(main_agent)
                self._payoff.add_player(main_agent.checkpoint())

            for _ in range(main_exploiters):
                self._learning_agents.append(
                    MainExploiter(race, initial_agents[race], self._payoff))
            for _ in range(league_exploiters):
                self._learning_agents.append(
                    LeagueExploiter(race, initial_agents[race], self._payoff))

        for player in self._learning_agents:
            self._payoff.add_player(player)

    def update(self, home, away, result):
        return self._payoff.update(home, away, result)

    def get_player(self, idx):
        return self._learning_agents[idx]

    def add_player(self, player):
        self._payoff.add_player(player)
