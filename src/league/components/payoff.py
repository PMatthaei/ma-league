from __future__ import annotations

import collections
from enum import Enum

import numpy as np


class MatchResult(Enum):
    WIN = 0,
    LOSS = 1,
    DRAW = 2,


def dd():
    return 0


class Payoff:

    def __init__(self, p_matrix, players):
        self._players = players
        self.p_matrix = p_matrix
        self.decay = 0.99
        self._wins = collections.defaultdict(dd)
        self._draws = collections.defaultdict(dd)
        self._losses = collections.defaultdict(dd)
        self._games = collections.defaultdict(dd)
        self._decay = 0.99

    def _win_rate(self, _home, _away):
        if self._games[_home, _away] == 0:
            return 0.5

        return (self._wins[_home, _away] +
                0.5 * self._draws[_home, _away]) / self._games[_home, _away]

    def __getitem__(self, match):
        home, away = match

        from league.roles.players import Player
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

    def add_player(self, player) -> None:
        self._players.append(player)

    def get_player(self, player_id: int):
        player = self._players[player_id]
        assert player.id_ == player_id, "ID mismatch."
        return player

    def get_players_of_type(self, cls):
        players = [
            player.id_ for player in self._players
            if isinstance(player, cls)
        ]
        if len(players) == 0:
            raise Exception(f"No opponent of Type: {cls} found.")

        return players

    @property
    def players(self):
        return self._players
