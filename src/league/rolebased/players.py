from __future__ import annotations

import re
from copy import deepcopy
from typing import Tuple, List

from torch import Tensor
from torch.multiprocessing.queue import Queue

from league.components import Matchmaker
from league.components.team_composer import Team
from modules.agents.agent_network import AgentNetwork


class Player(Matchmaker):

    def __init__(self, player_id: int, communication: Tuple[int, Tuple[Queue, Queue]], teams: List[Team], payoff: Tensor):
        super().__init__(communication, teams, payoff)
        self.id_ = player_id
        self._payoff = payoff
        self.team = teams[player_id]
        self.trained_steps = 0
        self._checkpoint_step = None

    def get_match(self, team: Team) -> Player:
        raise NotImplementedError

    def ready_to_checkpoint(self) -> bool:
        return False

    def _create_checkpoint(self) -> HistoricalPlayer:
        return HistoricalPlayer(self.id_, self._payoff, deepcopy(self.agent))

    @property
    def payoff(self):
        return self._payoff

    def checkpoint(self) -> HistoricalPlayer:
        self._checkpoint_step = self.trained_steps
        return self._create_checkpoint()

    def __str__(self):
        return f"{type(self).__name__}_{self.id_}"

    def prettier(self):
        return re.sub(r'(?<!^)(?=[A-Z])', '_', str(self)).lower()


class HistoricalPlayer(Player):

    def __init__(self, player_id: int, payoff, agent: AgentNetwork):
        """

        :param player_id:
        :param payoff:
        """

        super().__init__(player_id, payoff)
        self._parent = agent

    @property
    def parent(self) -> AgentNetwork:
        return self._parent

    def get_match(self, team: Team) -> Tuple[Player, bool]:
        raise ValueError("Historical players should not request matches.")

    def checkpoint(self) -> HistoricalPlayer:
        raise NotImplementedError

    def ready_to_checkpoint(self) -> bool:
        return False
