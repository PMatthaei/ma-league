import collections

import numpy as np

from league.roles.players import Player


class Payoff:

    def __init__(self):
        self.players = []
        self.wins = collections.defaultdict(lambda: 0)
        self.draws = collections.defaultdict(lambda: 0)
        self.losses = collections.defaultdict(lambda: 0)
        self.games = collections.defaultdict(lambda: 0)
        self.decay = 0.99

    def _win_rate(self, _home: Player, _away: Player):
        if self.games[_home, _away] == 0:
            return 0.5

        return (self.wins[_home, _away] +
                0.5 * self.draws[_home, _away]) / self.games[_home, _away]

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

    def update(self, home: Player, away: Player, result: str):
        """
        Integrate the result into its corresponding payoff matrix cell defined by home and away.
        :param home:
        :param away:
        :param result:
        :return:
        """
        for stats in (self.games, self.wins, self.draws, self.losses):
            stats[home, away] *= self.decay
            stats[away, home] *= self.decay

        self.games[home, away] += 1
        self.games[away, home] += 1
        if result == "win":
            self.wins[home, away] += 1
            self.losses[away, home] += 1
        elif result == "draw":
            self.draws[home, away] += 1
            self.draws[away, home] += 1
        else:
            self.wins[away, home] += 1
            self.losses[home, away] += 1

    def add_player(self, player: Player):
        self.players.append(player)
