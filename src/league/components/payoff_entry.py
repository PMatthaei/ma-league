from enum import IntEnum

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
        self._payoff_tensor: Tensor = payoff

    def _win_rates(self, idx):
        games = self._payoff_tensor[idx, :, PayoffEntry.GAMES]
        no_game_mask = games == 0.0
        wins = self._payoff_tensor[idx, :, PayoffEntry.WIN]
        draws = self._payoff_tensor[idx, :, PayoffEntry.DRAW]
        win_rates = (wins + 0.5 * draws) / games
        win_rates[no_game_mask] = .5  # If no games played we divided by 0 -> NaN -> replace with .5
        return win_rates

    def _increment(self, i, j, entry: PayoffEntry):
        self._payoff_tensor[i, j, entry] += 1
