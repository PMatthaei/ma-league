from __future__ import annotations

from typing import Tuple, Union, Any, List, Dict

from torch import Tensor
from torch.multiprocessing.queue import Queue

from league.components.self_play import PFSPSampling

from league.components.team_composer import Team
from league.rolebased.players import Player


class SimplePlayer(Player):

    def __init__(self, pid: int, communication: Tuple[int, Tuple[Queue, Queue]], teams: List[Team],
                 payoff: Tensor):
        super().__init__(pid, communication, teams, payoff)
        self._pfsp = PFSPSampling()

    def get_match(self, team=None) -> Union[Tuple[Any, bool], Tuple[Player, bool]]:
        """
        Samples an SimplePlayer opponent using PFSP with win rates as prioritization.
        :param **kwargs:
        :return:
        """
        opponents: List[int] = self.payoff.get_players_of_type(SimplePlayer)
        win_rates = self.payoff.win_rates(self.pid, opponents)
        chosen = self._pfsp.sample(opponents, prio_measure=win_rates, weighting="squared")
        return chosen_idx, chosen_team, self.payoff.players[chosen]

    def ready_to_checkpoint(self) -> bool:
        """
        Checkpoint Logic - Checkpoint agent if more than 2e9 training steps passed
        :return:
        """
        steps_passed = self.trained_steps - self._checkpoint_step
        if steps_passed < 2e9:
            return False
        return True
