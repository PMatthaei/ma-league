from __future__ import annotations

import re
from copy import deepcopy
from typing import Tuple, List, OrderedDict

from torch import Tensor
from torch.multiprocessing.queue import Queue

from league.components import Matchmaker
from league.components.payoff_entry import PayoffWrapper
from league.components.team_composer import Team
from modules.agents.agent_network import AgentNetwork


class Player(Matchmaker):

    def __init__(self, pid: int, communication: Tuple[int, Tuple[Queue, Queue]], teams: List[Team],
                 payoff: Tensor):
        super().__init__(communication, teams, payoff)
        self.pid = pid
        self.team = teams[pid]
        self.trained_steps = 0
        self._checkpoint_step = None

    def get_match(self, team=None) -> Player:
        raise NotImplementedError()

    def ready_to_checkpoint(self) -> bool:
        raise NotImplementedError()

    def is_main_player(self):
        raise NotImplementedError()

    def checkpoint(self) -> HistoricalPlayer:
        self._checkpoint_step = self.trained_steps
        return  # to send checkpoint cmd

    def __str__(self):
        return f"{type(self).__name__}_{self.pid}"

    def prettier(self):
        return re.sub(r'(?<!^)(?=[A-Z])', '_', str(self)).lower()


class HistoricalPlayer:

    def __init__(self, player_id: int, parent_id: int):
        """

        :param player_id:
        :param payoff:
        """

        super().__init__(player_id)
        self._parent_id = parent_id

    @property
    def parent(self) -> int:
        return self._parent_id

    def get_match(self, team: Team) -> Tuple[Player, bool]:
        raise ValueError("Historical players should not request matches.")

    def checkpoint(self) -> HistoricalPlayer:
        raise NotImplementedError

    def ready_to_checkpoint(self) -> bool:
        return False
