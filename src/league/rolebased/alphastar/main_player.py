from __future__ import annotations

from typing import Tuple, Union, Any

import numpy as np

from league.components.self_play import PFSPSampling
from league.rolebased.players import Player, HistoricalPlayer
from league.utils.helpers import remove_monotonic_suffix


class MainPlayer(Player):
    def __init__(self, pid: int, payoff, team):
        super().__init__(pid, payoff, team)
        self._checkpoint_step = 0
        self._pfsp = PFSPSampling()

    def get_match(self, team=None) -> Union[Tuple[Any, bool], Tuple[Player, bool]]:
        """
        Samples an HistoricalPlayer opponent using PFSP with probability 0.5.
        In other cases play against MainPlayers using SP or verify that no player was omitted.
        :return:
        """
        coin_toss = np.random.random()

        # Make sure you can beat the League via PFSP
        if coin_toss < 0.5:
            return self._pfsp_branch()

        main_agents = [
            player for player in self.payoff.players
            if isinstance(player, MainPlayer)
        ]
        opponent = np.random.choice(main_agents)

        # Verify if there are some rare players we omitted
        if coin_toss < 0.5 + 0.15:
            request = self._verification_branch(opponent)
            if request is not None:
                return request

        # Else play against yourself (MainPlayer)
        return self._selfplay_branch(opponent)

    def _pfsp_branch(self) -> Union[Tuple[Player, bool], Tuple[None, bool]]:
        """
        PFSP against historical players
        :return:
        """
        historical = [
            player.tid for player in self.payoff.players
            if isinstance(player, HistoricalPlayer)
        ]

        if len(historical) == 0:  # no new historical opponents found # TODO
            return None, False

        win_rates = self.payoff[self.tid, historical]
        chosen = self._pfsp.sample(historical, prio_measure=win_rates, weighting="squared")
        return self.payoff.players[chosen], True

    def _selfplay_branch(self, opponent: Player) -> Tuple[Player, bool]:
        """
        SP against main players, with exceptions if the opponent is too strong.
        :param opponent:
        :return:
        """
        # Play self-play match
        if self.payoff[self.tid, opponent.tid] > 0.3:
            return opponent, False

        # Opponent too strong -> use checkpoint of the opponent as curriculum
        historical = [
            player.tid for player in self.payoff.players
            if isinstance(player, HistoricalPlayer) and player.parent == opponent
        ]

        if len(historical) == 0:  # no new historical opponents found # TODO
            return opponent, False

        # PFSP on checkpoints of opponent
        win_rates = self.payoff[self.tid, historical]
        chosen = self._pfsp.sample(historical, prio_measure=win_rates, weighting="variance")
        return self.payoff.players[chosen], True

    def _verification_branch(self, opponent) -> Union[Tuple[None, None], Tuple[Player, bool]]:
        # Check exploitation
        from league.rolebased.alphastar.exploiters import MainExploiter

        exploiters = set([  # Get all exploiters
            player for player in self.payoff.players
            if isinstance(player, MainExploiter)
        ])
        exp_historical = [  # Get all historical players which originate from exploiters
            player.tid for player in self.payoff.players
            if isinstance(player, HistoricalPlayer) and player.parent in exploiters
        ]
        # If historical exploiters min. win rate is smaller threshold -> PFSP
        win_rates = self.payoff[self.tid, exp_historical]
        if len(win_rates) and win_rates.min() < 0.3:
            chosen = self._pfsp.sample(exp_historical, prio_measure=win_rates, weighting="squared")
            return self.payoff.players[chosen], True

        # Check forgetting
        historical = [
            player.tid for player in self.payoff.players
            if isinstance(player, HistoricalPlayer) and player.parent == opponent
        ]
        win_rates = self.payoff[self.tid, historical]
        win_rates, historical = remove_monotonic_suffix(win_rates, historical)
        if len(win_rates) and win_rates.min() < 0.7:
            chosen = self._pfsp.sample(historical, prio_measure=win_rates, weighting="squared")
            return self.payoff.players[chosen], True

        # TODO: when and why do we get here?
        return None, None

    def ready_to_checkpoint(self) -> bool:
        """
        Checkpoint Logic - AlphaStars Checkpointing Logic
        :return:
        """
        steps_passed = self.agent.trained_steps - self._checkpoint_step
        if steps_passed < 2e9:  # TODO make constant
            return False

        historical = [
            player.tid for player in self.payoff.players
            if isinstance(player, HistoricalPlayer)
        ]
        win_rates = self.payoff[self.tid, historical]
        return win_rates.min() > 0.7 or steps_passed > 4e9  # TODO make constant
