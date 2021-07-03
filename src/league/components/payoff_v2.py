from __future__ import annotations

from enum import Enum
from typing import List, Union

import numpy as np


class PayoffEntry(Enum):
    GAMES = 0,
    WIN = 1,
    LOSS = 2,
    DRAW = 3,


class PayoffV2:

    def __init__(self, payoff_dict: dict):
        self.payoff_dict = payoff_dict
        self.decay = 0.99

    def _win_rate(self, home: int, away: int):
        """
        Calculates the win rate of the home team against the away team.
        Draws are weighted have as much as wins.
        :param home:
        :param away:
        :return:
        """
        if self.payoff_dict[home, away, PayoffEntry.GAMES] == 0:
            return 0.5

        return (self.payoff_dict[home, away, PayoffEntry.WIN] +
                0.5 * self.payoff_dict[home, away, PayoffEntry.DRAW]) / self.payoff_dict[
                   home, away, PayoffEntry.GAMES]

    def __getitem__(self, match):
        """
        Get the win rates of the home player against one or more away teams.
        Away teams can be passed as list.
        :param match:
        :return:
        """
        home, away = match
        away, home = map(lambda x: [x] if isinstance(x, int) else x, self._map_to_id(away, home))
        if not self.has_entries(home, away):  # init if new match
            self._init_entries(home, away)

        win_rates = np.array([[self._win_rate(h, a) for a in away] for h in home])
        if win_rates.shape[0] == 1 or win_rates.shape[1] == 1:
            win_rates = win_rates.reshape(-1)

        return win_rates

    def update(self, home, away, result: str):
        """
        Update the statistic of a certain match with a new result
        :param home:
        :param away:
        :param result:
        :return:
        """
        away, home = self._map_to_id(away, home)

        # On every update decay old data importance
        self._apply_decay(home, away)

        # Each result means a match between the two players finished
        self.payoff_dict[home, away, PayoffEntry.GAMES] += 1
        self.payoff_dict[away, home, PayoffEntry.GAMES] += 1

        if result == PayoffEntry.WIN:
            self.payoff_dict[home, away, PayoffEntry.WIN] += 1
            self.payoff_dict[away, home, PayoffEntry.LOSS] += 1
        elif result == PayoffEntry.DRAW:
            self.payoff_dict[home, away, PayoffEntry.DRAW] += 1
            self.payoff_dict[away, home, PayoffEntry.DRAW] += 1
        elif result == PayoffEntry.LOSS:
            self.payoff_dict[home, away, PayoffEntry.LOSS] += 1
            self.payoff_dict[away, home, PayoffEntry.WIN] += 1
        else:
            raise NotImplementedError("Payoff Update not implemented.")

    def has_played(self, home: int, away: int):
        key = (home, away, PayoffEntry.GAMES)
        return key in self.payoff_dict and self.payoff_dict[key] > 0

    def _apply_decay(self, home: int, away: int):
        for entry in PayoffEntry:
            self.payoff_dict[home, away, entry] *= self.decay
            self.payoff_dict[away, home, entry] *= self.decay

    def has_entries(self, home: Union[int, List[int]], away: Union[int, List[int]]):
        """
        Collect all keys which are needed to capture statistics of the given match
        and test if they exist for later updates. If even one key is missing return False
        :param home:
        :param away:
        :return:
        """
        keys = self._build_keys(away, home)
        return all([k in self.payoff_dict for k in keys])

    def _init_entries(self, home: Union[int, List[int]], away: Union[int, List[int]]):
        """
        Initialize all necessary dict entries to allow for later updates on these statistics.
        :param home:
        :param away:
        :return:
        """
        keys = self._build_keys(away, home)
        for k in keys:
            self.payoff_dict[k] = 0
            self.payoff_dict[k] = 0

    def _build_keys(self, away, home):
        keys = []
        if isinstance(away, int):
            away = [away]
        if isinstance(home, int):
            home = [home]
        for h in home:
            for a in away:
                for entry in PayoffEntry:
                    keys.append((h, a, entry))
                    keys.append((a, h, entry))
        return keys

    def _map_to_id(self, away, home):
        from league.roles.players import Player
        if isinstance(home, Player):
            home = home.id_
        if isinstance(away, Player):
            away = away.id_
        return away, home
