from __future__ import annotations

from typing import Tuple, Union, Any, List, Dict, OrderedDict

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

    def get_match(self, team=None) -> Union[None, Tuple[int, Team, OrderedDict]]:
        """
        Samples an SimplePlayer opponent using PFSP with win rates as prioritization.
        :param **kwargs:
        :return:
        """
        agents = self.get_agents()
        opponents_tids: List[int] = [agent for agent in agents if agent == SimplePlayer]
        win_rates = self.payoff.win_rates(self.pid, opponents_tids)
        chosen_tid = self._pfsp.sample(opponents_tids, prio_measure=win_rates, weighting="squared")
        chosen_team = self.get_team(tid=chosen_tid)
        chosen_idx = self._tid_to_instance[chosen_tid]
        return chosen_idx, chosen_team, agents[chosen_tid]

    def ready_to_checkpoint(self) -> bool:
        """
        Checkpoint Logic - Checkpoint agent if more than 2e9 training steps passed
        :return:
        """
        steps_passed = self.trained_steps - self._checkpoint_step
        if steps_passed < 2e9:
            return False
        return True
