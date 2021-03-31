from __future__ import annotations

import numpy as np


class Payoff:

    def __init__(self, p_matrix, players):
        self.players = players
        self.p_matrix = p_matrix
        self.decay = 0.99

    def _win_rate(self, _home: int, _away: int):
        if (_home, _away) not in self.p_matrix:
            self.p_matrix[_home, _away] = 0

        if self.p_matrix[_home, _away] == 0:
            return 0.5

        return (self.p_matrix[_home, _away, 'win'] +
                0.5 * self.p_matrix[_home, _away, 'draw']) / self.p_matrix[_home, _away]

    def __getitem__(self, match: tuple):
        home, away = match

        if isinstance(home, int):
            home = [home]

        if isinstance(away, int):
            away = [away]

        win_rates = np.array([[self._win_rate(h, a) for a in away] for h in home])
        if win_rates.shape[0] == 1 or win_rates.shape[1] == 1:
            win_rates = win_rates.reshape(-1)

        return win_rates

    def update(self, home: int, away: int, result: str):
        if not self.has_entries(home, away):
            self._init_p_matrix_entries(home, away)

        self._apply_decay(home, away)

        self._update_episodes_played(home, away)
        self._update_episodes_played(away, home)

        if result == "win":
            self._update_result(home, away, "win")
            self._update_result(away, home, "loss")
        elif result == "draw":
            self._update_result(home, away, result)
            self._update_result(away, home, result)
        else:
            self._update_result(home, away, "loss")
            self._update_result(away, home, "loss")

    def _apply_decay(self, home: int, away: int):
        self.p_matrix[home, away, "win"] *= self.decay
        self.p_matrix[away, home, "win"] *= self.decay
        self.p_matrix[home, away, "loss"] *= self.decay
        self.p_matrix[away, home, "loss"] *= self.decay
        self.p_matrix[home, away, "draw"] *= self.decay
        self.p_matrix[away, home, "draw"] *= self.decay
        self.p_matrix[away, home] *= self.decay

    def add_player(self, player):
        self.players.append(player)

    def _update_result(self, home: int, away: int, result):
        if (home, away, result) in self.p_matrix:
            self.p_matrix[home, away, result] += 1
        else:
            self.p_matrix[home, away, result] = 1

    def _update_episodes_played(self, home: int, away: int):
        if (home, away) in self.p_matrix:
            self.p_matrix[home, away] += 1
        else:
            self.p_matrix[home, away] = 1

    def has_entries(self, home: int, away: int):
        keys = []
        for result in ["win", "loss", "draw"]:
            keys.append((home, away, result))
            keys.append((away, home, result))
        keys.append((home, away))
        keys.append((away, home))

        return all([k in self.p_matrix for k in keys])

    def _init_p_matrix_entries(self, home: int, away: int):
        self.p_matrix[home, away, "win"] = 0
        self.p_matrix[away, home, "win"] = 0
        self.p_matrix[home, away, "loss"] = 0
        self.p_matrix[away, home, "loss"] = 0
        self.p_matrix[home, away, "draw"] = 0
        self.p_matrix[away, home, "draw"] = 0
        self.p_matrix[away, home] = 0
