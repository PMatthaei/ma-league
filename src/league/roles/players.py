from __future__ import annotations

from typing import Tuple, Union, Any

from league.components.payoff import Payoff
from league.roles.agent import Agent
from league.components.pfsp import prioritized_fictitious_self_play
import numpy as np

from league.utils.various import remove_monotonic_suffix


class Player(object):

    def __init__(self):
        """

        """
        self.player_id = None
        self.agent = None
        self._payoff = None
        self._team_plan = None

    def get_match(self) -> Player:
        pass

    def ready_to_checkpoint(self) -> bool:
        return False

    def _create_checkpoint(self) -> HistoricalPlayer:
        print("Saving checkpoint as HistoricalPlayer")
        return HistoricalPlayer(self.player_id, self.agent, self._payoff)

    @property
    def payoff(self) -> Payoff:
        return self._payoff

    @property
    def team_plan(self):
        return self._team_plan

    def checkpoint(self):
        raise NotImplementedError


class MainPlayer(Player):

    def __init__(self, player_id, team_plan, agent, payoff):
        """

        :param player_id:
        :param team_plan:
        :param agent:
        :param payoff:
        """
        super().__init__()
        self.player_id = player_id
        self.agent = Agent(team_plan, agent.get_weights())
        self._payoff = payoff
        self._team_plan = agent.team_plan
        self._checkpoint_step = 0

    def _pfsp_branch(self) -> Tuple[Player, bool]:
        """

        :return:
        """
        historical = [
            player.player_id for player in self._payoff.players
            if isinstance(player, HistoricalPlayer)
        ]
        win_rates = self._payoff[self.player_id, historical]
        chosen_id = np.random.choice(historical, p=prioritized_fictitious_self_play(win_rates, weighting="squared"))
        return self._payoff.players[chosen_id], True

    def _selfplay_branch(self, opponent: Player) -> Tuple[Player, bool]:
        """

        :param opponent:
        :return:
        """
        # Play self-play match
        if self._payoff[self.player_id, opponent.player_id] > 0.3:
            return opponent, False

        # If opponent is too strong, look for a checkpoint as curriculum
        historical = [
            player.player_id for player in self._payoff.players
            if isinstance(player, HistoricalPlayer) and player.parent == opponent
        ]

        if len(historical) == 0:  # no new historical opponents found # TODO
            return opponent, False

        win_rates = self._payoff[self.player_id, historical]
        chosen_id = np.random.choice(historical, p=prioritized_fictitious_self_play(win_rates, weighting="variance"))
        return self._payoff.players[chosen_id], True

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
            player.player_id for player in self._payoff.players
            if isinstance(player, HistoricalPlayer) and player.parent in exploiters
        ]
        win_rates = self._payoff[self.player_id, exp_historical]
        if len(win_rates) and win_rates.min() < 0.3:
            chosen_id = np.random.choice(exp_historical,
                                         p=prioritized_fictitious_self_play(win_rates, weighting="squared"))
            return self._payoff.players[chosen_id], True

        # Check forgetting
        historical = [
            player.player_id for player in self._payoff.players
            if isinstance(player, HistoricalPlayer) and player.parent == opponent
        ]
        win_rates = self._payoff[self.player_id, historical]
        win_rates, historical = remove_monotonic_suffix(win_rates, historical)
        if len(win_rates) and win_rates.min() < 0.7:
            chosen_id = np.random.choice(historical, p=prioritized_fictitious_self_play(win_rates, weighting="squared"))
            return self._payoff.players[chosen_id], True

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
        steps_passed = self.agent.get_steps() - self._checkpoint_step
        if steps_passed < 2e9:
            return False

        historical = [
            player.player_id for player in self._payoff.players
            if isinstance(player, HistoricalPlayer)
        ]
        win_rates = self._payoff[self.player_id, historical]
        return win_rates.min() > 0.7 or steps_passed > 4e9

    def checkpoint(self):
        """

        :return:
        """
        self._checkpoint_step = self.agent.get_steps()
        return self._create_checkpoint()


class HistoricalPlayer(Player):

    def __init__(self, player_id, agent, payoff):
        """

        :param player_id:
        :param agent:
        :param payoff:
        """
        super().__init__()
        self.player_id = player_id
        self._agent = Agent(agent.team_plan, agent.get_weights())
        self._payoff = payoff
        self._team_plan = agent.team_plan
        self._parent = agent

    @property
    def parent(self):
        return self._parent

    def get_match(self) -> Tuple[Player, bool]:
        raise ValueError("Historical players should not request matches")

    def checkpoint(self):
        raise NotImplementedError

    def ready_to_checkpoint(self) -> bool:
        return False
