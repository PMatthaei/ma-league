from __future__ import annotations

import re
from copy import deepcopy
from typing import Tuple, Union

from league.components.payoff import Payoff

from modules.agents.agent import Agent


class Player(object):

    def __init__(self, player_id: int, payoff: Payoff, team):
        self.id_ = player_id
        self._payoff = payoff
        self.team = team
        self.agent: Union[Agent, None] = None

    def get_match(self) -> Player:
        raise NotImplementedError

    def ready_to_checkpoint(self) -> bool:
        return False

    def _create_checkpoint(self) -> HistoricalPlayer:
        return HistoricalPlayer(self.id_, self._payoff, deepcopy(self.agent))

    @property
    def payoff(self) -> Payoff:
        return self._payoff

    def checkpoint(self) -> HistoricalPlayer:
        self._checkpoint_step = self.agent.trained_steps
        return self._create_checkpoint()

    def __str__(self):
        return f"{type(self).__name__}_{self.id_}"

    def prettier(self):
        return re.sub(r'(?<!^)(?=[A-Z])', '_', str(self)).lower()


class HistoricalPlayer(Player):

    def __init__(self, player_id: int, payoff: Payoff, agent: Agent):
        """

        :param player_id:
        :param payoff:
        """

        super().__init__(player_id, payoff)
        self._parent = agent

    @property
    def parent(self) -> Agent:
        return self._parent

    def get_match(self) -> Tuple[Player, bool]:
        raise ValueError("Historical players should not request matches.")

    def checkpoint(self) -> HistoricalPlayer:
        raise NotImplementedError

    def ready_to_checkpoint(self) -> bool:
        return False
