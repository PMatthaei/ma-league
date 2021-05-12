from __future__ import annotations

from enum import Enum
from typing import Tuple, Union, List

import numpy as np


class MatchResult(Enum):
    WIN = 0,
    LOSS = 1,
    DRAW = 2,


class Payoff:

    def __init__(self, p_matrix, players):
        self.players = players
        self.p_matrix = p_matrix
        self.decay = 0.99

    def _win_rate(self, _home: int, _away: int):
        """
        Calculates the win rate of the home team against the away team.
        Draws are weighted have as much as wins.
        :param _home:
        :param _away:
        :return:
        """
        if (_home, _away) not in self.p_matrix:
            self.p_matrix[_home, _away] = 0

        if self.p_matrix[_home, _away] == 0:
            return 0.5

        return (self.p_matrix[_home, _away, MatchResult.WIN] +
                0.5 * self.p_matrix[_home, _away, MatchResult.DRAW]) / self.p_matrix[_home, _away]

    def __getitem__(self, match: Union[Tuple[int, List[int]], Tuple[int, int]]):
        """
        Get the win rates of the home player against one or more away teams.
        Away teams can be passed as list.
        :param match:
        :return:
        """
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
        """
        Update the statistic of a certain match with a new result
        :param home:
        :param away:
        :param result:
        :return:
        """
        if not self.has_entries(home, away):  # init if new match
            self._init_p_matrix_entries(home, away)

        self._apply_decay(home, away)

        self.p_matrix[home, away] += 1
        self.p_matrix[away, home] += 1

        if result == MatchResult.WIN:
            self.p_matrix[home, away, MatchResult.WIN] += 1
            self.p_matrix[away, home, MatchResult.LOSS] += 1
        elif result == MatchResult.DRAW:
            self.p_matrix[home, away, MatchResult.DRAW] += 1
            self.p_matrix[away, home, MatchResult.DRAW] += 1
        else:
            self.p_matrix[home, away, MatchResult.LOSS] += 1
            self.p_matrix[away, home, MatchResult.WIN] += 1

        return self.players[home], self.players[away]

    def _apply_decay(self, home: int, away: int):
        self.p_matrix[home, away, MatchResult.WIN] *= self.decay
        self.p_matrix[away, home, MatchResult.WIN] *= self.decay
        self.p_matrix[home, away, MatchResult.LOSS] *= self.decay
        self.p_matrix[away, home, MatchResult.LOSS] *= self.decay
        self.p_matrix[home, away, MatchResult.DRAW] *= self.decay
        self.p_matrix[away, home, MatchResult.DRAW] *= self.decay
        self.p_matrix[home, away] *= self.decay
        self.p_matrix[away, home] *= self.decay

    def add_player(self, player) -> None:
        self.players.append(player)

    def get_player(self, player_id: int):
        player = self.players[player_id]
        assert player.id_ == player_id, "ID mismatch."
        return player

    def has_entries(self, home: int, away: int):
        """
        Collect all keys which are needed to capture statistics of the given match
        and test if they exist for later updates.
        :param home:
        :param away:
        :return:
        """
        keys = []
        for result in [MatchResult.WIN, MatchResult.LOSS, MatchResult.DRAW]:
            keys.append((home, away, result))
            keys.append((away, home, result))
        keys.append((home, away))
        keys.append((away, home))

        return all([k in self.p_matrix for k in keys])

    def _init_p_matrix_entries(self, home: int, away: int):
        """
        Initialize all necessary dict entries to allow for later updates on these statistics.
        :param home:
        :param away:
        :return:
        """
        self.p_matrix[home, away, MatchResult.WIN] = 0
        self.p_matrix[away, home, MatchResult.WIN] = 0
        self.p_matrix[home, away, MatchResult.LOSS] = 0
        self.p_matrix[away, home, MatchResult.LOSS] = 0
        self.p_matrix[home, away, MatchResult.DRAW] = 0
        self.p_matrix[away, home, MatchResult.DRAW] = 0
        self.p_matrix[home, away] = 0
        self.p_matrix[away, home] = 0
