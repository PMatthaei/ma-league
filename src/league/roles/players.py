from league.league import Agent
from league.roles.exploiters import MainExploiter
from league.utils.pfsp import prioritized_fictitious_self_play
import numpy as np

from league.utils.various import remove_monotonic_suffix


class Player(object):

    def __init__(self):
        self._payoff = None
        self._race = None

    def get_match(self):
        pass

    def ready_to_checkpoint(self):
        return False

    def _create_checkpoint(self):
        return HistoricalPlayer(self, self.payoff)

    @property
    def payoff(self):
        return self._payoff

    @property
    def race(self):
        return self._race

    def checkpoint(self):
        raise NotImplementedError


class MainPlayer(Player):

    def __init__(self, race, agent, payoff):
        super().__init__()
        self.agent = Agent(race, agent.get_weights())
        self._payoff = payoff
        self._race = agent.race
        self._checkpoint_step = 0

    def _pfsp_branch(self):
        historical = [
            player for player in self._payoff.players
            if isinstance(player, HistoricalPlayer)
        ]
        win_rates = self._payoff[self, historical]
        return np.random.choice(
            historical, p=prioritized_fictitious_self_play(win_rates, weighting="squared")), True

    def _selfplay_branch(self, opponent):
        # Play self-play match
        if self._payoff[self, opponent] > 0.3:
            return opponent, False

        # If opponent is too strong, look for a checkpoint
        # as curriculum
        historical = [
            player for player in self._payoff.players
            if isinstance(player, HistoricalPlayer) and player.parent == opponent
        ]
        win_rates = self._payoff[self, historical]
        return np.random.choice(
            historical, p=prioritized_fictitious_self_play(win_rates, weighting="variance")), True

    def _verification_branch(self, opponent):
        # Check exploitation
        exploiters = set([
            player for player in self._payoff.players
            if isinstance(player, MainExploiter)
        ])
        exp_historical = [
            player for player in self._payoff.players
            if isinstance(player, HistoricalPlayer) and player.parent in exploiters
        ]
        win_rates = self._payoff[self, exp_historical]
        if len(win_rates) and win_rates.min() < 0.3:
            return np.random.choice(
                exp_historical, p=prioritized_fictitious_self_play(win_rates, weighting="squared")), True

        # Check forgetting
        historical = [
            player for player in self._payoff.players
            if isinstance(player, HistoricalPlayer) and player.parent == opponent
        ]
        win_rates = self._payoff[self, historical]
        win_rates, historical = remove_monotonic_suffix(win_rates, historical)
        if len(win_rates) and win_rates.min() < 0.7:
            return np.random.choice(
                historical, p=prioritized_fictitious_self_play(win_rates, weighting="squared")), True

        return None

    def get_match(self):
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

    def ready_to_checkpoint(self):
        steps_passed = self.agent.get_steps() - self._checkpoint_step
        if steps_passed < 2e9:
            return False

        historical = [
            player for player in self._payoff.players
            if isinstance(player, HistoricalPlayer)
        ]
        win_rates = self._payoff[self, historical]
        return win_rates.min() > 0.7 or steps_passed > 4e9

    def checkpoint(self):
        self._checkpoint_step = self.agent.get_steps()
        return self._create_checkpoint()


class HistoricalPlayer(Player):

    def __init__(self, agent, payoff):
        super().__init__()
        self._agent = Agent(agent.race, agent.get_weights())
        self._payoff = payoff
        self._race = agent.race
        self._parent = agent

    @property
    def parent(self):
        return self._parent

    def get_match(self):
        raise ValueError("Historical players should not request matches")

    def checkpoint(self):
        raise NotImplementedError

    def ready_to_checkpoint(self):
        return False