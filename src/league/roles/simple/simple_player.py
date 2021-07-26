from __future__ import annotations

from typing import Tuple, Union, Any


from league.components.self_play import PrioritizedFictitiousSelfPlay

from league.roles.players import Player


class SimplePlayer(Player):
    def __init__(self, player_id: int, payoff, team):
        super().__init__(player_id, payoff, team)
        self._checkpoint_step = 0
        self._pfsp = PrioritizedFictitiousSelfPlay()

    def get_match(self) -> Union[Tuple[Any, bool], Tuple[Player, bool]]:
        """
        Samples an SimplePlayer opponent using PFSP with win rates as prioritization.
        :return:
        """
        simple_players = self.payoff.get_players_of_type(SimplePlayer)
        win_rates = self._payoff[self, simple_players]
        chosen = self._pfsp.sample(simple_players, prio_measure=win_rates, weighting="squared")
        return self._payoff.players[chosen], True

    def ready_to_checkpoint(self) -> bool:
        """
        Checkpoint Logic - Checkpoint agent if more than 2e9 training steps passed
        :return:
        """
        steps_passed = self.agent.trained_steps - self._checkpoint_step
        if steps_passed < 2e9:
            return False
        return True


