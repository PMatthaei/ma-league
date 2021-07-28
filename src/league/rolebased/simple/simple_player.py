from __future__ import annotations

from typing import Tuple, Union, Any, List, Dict

from torch import Tensor
from torch.multiprocessing.queue import Queue

from league.components.self_play import PFSPSampling

from league.components.team_composer import Team
from league.rolebased.players import Player


class SimplePlayer(Player):

    def __init__(self, player_id: int, communication: Tuple[int, Tuple[Queue, Queue]], teams: List[Team],
                 payoff: Tensor):
        super().__init__(player_id, communication, teams, payoff)
        self._pfsp = PFSPSampling()

    def get_match(self, home_team) -> Union[Tuple[Any, bool], Tuple[Player, bool]]:
        """
        Samples an SimplePlayer opponent using PFSP with win rates as prioritization.
        :return:
        """
        simple_players = self.payoff.get_players_of_type(SimplePlayer)
        win_rates = self.payoff[self, simple_players]
        chosen = self._pfsp.sample(simple_players, prio_measure=win_rates, weighting="squared")
        return self.payoff.players[chosen], True

    def ready_to_checkpoint(self) -> bool:
        """
        Checkpoint Logic - Checkpoint agent if more than 2e9 training steps passed
        :return:
        """
        steps_passed = self.trained_steps - self._checkpoint_step
        if steps_passed < 2e9:
            return False
        return True


