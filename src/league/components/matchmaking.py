from league.components.payoff import Payoff
from league.components.self_play import OpponentSampling
from league.roles.players import Player


class Matchmaking:

    def __init__(self, payoff: Payoff, sampling_strategy: OpponentSampling):
        self._payoff = payoff
        self._sampling_strategy = sampling_strategy
        pass

    def get_match(self, player: Player):
        """
        Find a opponent for the given player using various methods.
        :param player:
        :return:
        """
        pass
