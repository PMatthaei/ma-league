from enum import IntEnum
from typing import List

from torch import Tensor


class PayoffEntry(IntEnum):
    GAMES = 0,  # How many episodes
    WIN = 1,  # How many episodes have been won
    LOSS = 2,  # How many episodes have been lost
    DRAW = 3,  # How many episodes ended in draw
    MATCHES = 4,  # How often has the corresponding instance been matched with all other instances


class PayoffWrapper:
    def __init__(self, payoff: Tensor):
        """
        Wraps around the shared payoff tensor to act as proxy for reoccurring operations on the payoff tensor.
        :param payoff:
        """
        self._p: Tensor = payoff

    def win_rates(self, idx, indices: List[int]=None):
        games = self._p[idx, :, PayoffEntry.GAMES] if indices is None else self._p[idx, indices, PayoffEntry.GAMES]
        no_game_mask = games == 0.0
        wins = self._p[idx, :, PayoffEntry.WIN] if indices is None else self._p[idx, indices, PayoffEntry.WIN]
        draws = self._p[idx, :, PayoffEntry.DRAW] if indices is None else self._p[idx, indices, PayoffEntry.DRAW]
        win_rates = (wins + 0.5 * draws) / games
        win_rates[no_game_mask] = .5  # If no games played we divided by 0 -> NaN -> replace with .5
        return win_rates

    def games(self, i):
        return self._p[i, :, PayoffEntry.WIN]

    def matches(self, i):
        return self._p[i, :, PayoffEntry.MATCHES]

    def win(self, i, j):
        self.increment(i, j, PayoffEntry.WIN)

    def draw(self, i, j):
        self.increment(i, j, PayoffEntry.DRAW)

    def loss(self, i, j):
        self.increment(i, j, PayoffEntry.LOSS)

    def match(self, i, j):
        self.increment(i, j, PayoffEntry.MATCHES)

    def increment(self, i, j, entry: PayoffEntry):
        self._p[i, j, entry] += 1
