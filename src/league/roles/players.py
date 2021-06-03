from __future__ import annotations

import re
from copy import deepcopy
from typing import Tuple, Union, Any

from league.components.payoff import Payoff
from league.components.self_play import prioritized_fictitious_self_play
import numpy as np

from league.utils.various import remove_monotonic_suffix
from learners.learner import Learner
from modules.agents.agent import Agent


class Player(object):

    def __init__(self, player_id: int, payoff: Payoff):
        self.id_ = player_id
        self._payoff = payoff
        self.agent: Union[Agent, None] = None

    def get_match(self) -> Player:
        pass

    def get_current_step(self) -> int:
        pass

    def ready_to_checkpoint(self) -> bool:
        return False

    def _create_checkpoint(self) -> HistoricalPlayer:
        return HistoricalPlayer(self.id_, self._payoff, deepcopy(self.agent))

    @property
    def payoff(self) -> Payoff:
        return self._payoff

    def checkpoint(self) -> HistoricalPlayer:
        raise NotImplementedError

    def __str__(self):
        return f"{type(self).__name__}_{self.id_}"

    def prettier(self):
        return re.sub(r'(?<!^)(?=[A-Z])', '_', str(self)).lower()


class MainPlayer(Player):

    def __init__(self, player_id: int, payoff: Payoff):
        super().__init__(player_id, payoff)
        self._checkpoint_step = 0

    def _pfsp_branch(self) -> Union[Tuple[Player, bool], Tuple[None, bool]]:
        """

        :return:
        """
        historical = [
            player.id_ for player in self._payoff.players
            if isinstance(player, HistoricalPlayer)
        ]

        if len(historical) == 0:  # no new historical opponents found # TODO
            return None, False

        win_rates = self._payoff[self.id_, historical]
        chosen = np.random.choice(historical, p=prioritized_fictitious_self_play(win_rates, weighting="squared"))
        return self._payoff.players[chosen], True

    def _selfplay_branch(self, opponent: Player) -> Tuple[Player, bool]:
        """

        :param opponent:
        :return:
        """
        # Play self-play match
        if self._payoff[self.id_, opponent.id_] > 0.3:
            return opponent, False

        # If opponent is too strong, look for a checkpoint as curriculum
        historical = [
            player.id_ for player in self._payoff.players
            if isinstance(player, HistoricalPlayer) and player.parent == opponent
        ]

        if len(historical) == 0:  # no new historical opponents found # TODO
            return opponent, False

        win_rates = self._payoff[self.id_, historical]
        chosen = np.random.choice(historical, p=prioritized_fictitious_self_play(win_rates, weighting="variance"))
        return self._payoff.players[chosen], True

    def _verification_branch(self, opponent) -> Union[Tuple[None, None], Tuple[Player, bool]]:
        """

        :param opponent:
        :return:
        """
        # Check exploitation
        from league.roles.exploiters import MainExploiter

        exploiters = set([
            player for player in self._payoff.players
            if isinstance(player, MainExploiter)
        ])
        exp_historical = [
            player.id_ for player in self._payoff.players
            if isinstance(player, HistoricalPlayer) and player.parent in exploiters
        ]
        win_rates = self._payoff[self.id_, exp_historical]
        if len(win_rates) and win_rates.min() < 0.3:
            chosen = np.random.choice(exp_historical,
                                      p=prioritized_fictitious_self_play(win_rates, weighting="squared"))
            return self._payoff.players[chosen], True

        # Check forgetting
        historical = [
            player.id_ for player in self._payoff.players
            if isinstance(player, HistoricalPlayer) and player.parent == opponent
        ]
        win_rates = self._payoff[self.id_, historical]
        win_rates, historical = remove_monotonic_suffix(win_rates, historical)
        if len(win_rates) and win_rates.min() < 0.7:
            chosen = np.random.choice(historical, p=prioritized_fictitious_self_play(win_rates, weighting="squared"))
            return self._payoff.players[chosen], True

        # TODO: when and why do we get here?
        return None, None

    def get_match(self) -> Union[Tuple[Any, bool], Tuple[Player, bool]]:
        """

        :return:
        """
        coin_toss = np.random.random()

        # Make sure you can beat the League
        if coin_toss < 0.5:
            return self._pfsp_branch()
        main_agents = [
            player for player in self._payoff.players
            if isinstance(player, MainPlayer)
        ]
        opponent = np.random.choice(main_agents)

        # Verify if there are some rare players we omitted
        if coin_toss < 0.5 + 0.15:
            request = self._verification_branch(opponent)
            if request is not None:
                return request

        return self._selfplay_branch(opponent)

    def ready_to_checkpoint(self) -> bool:
        """

        :return:
        """
        steps_passed = self.agent.trained_steps - self._checkpoint_step
        if steps_passed < 2e9:  # TODO make constant
            return False

        historical = [
            player.id_ for player in self._payoff.players
            if isinstance(player, HistoricalPlayer)
        ]
        win_rates = self._payoff[self.id_, historical]
        return win_rates.min() > 0.7 or steps_passed > 4e9  # TODO make constant

    def checkpoint(self) -> HistoricalPlayer:
        """

        :return:
        """
        self._checkpoint_step = self.agent.trained_steps
        return self._create_checkpoint()


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
