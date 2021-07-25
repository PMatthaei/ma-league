from typing import Tuple, Dict, Union, List, OrderedDict

import torch as th
from torch import Tensor

from league.components import PayoffEntry
from league.components.agent_pool import AgentPool
from league.components.self_play import OpponentSampling, PrioritizedFictitiousSelfPlay
from league.utils.team_composer import Team
from modules.agents import AgentNetwork


def dd():
    return 0


class Matchmaking:

    def __init__(self, agent_pool: AgentPool, teams: List[Team], payoff: Tensor, allocation: Dict[int, int]):
        self._agent_pool = agent_pool
        self._allocation = allocation
        self._payoff = payoff
        self._teams = teams

    def get_instance_id(self, team: Team) -> int:
        return self._allocation[team.id_]

    def get_team(self, instance_id: int) -> Team:
        _inv_allocation = {v: k for k, v in self._allocation.items()}
        tid = _inv_allocation[instance_id]
        return next((team for team in self._teams if team.id_ == tid), None)

    def get_match(self, home_team: Team) -> Union[None, Tuple[Team, OrderedDict]]:
        raise NotImplementedError()


class PFSPMatchmaking(Matchmaking):
    def __init__(self, agent_pool: AgentPool, teams: List[Team], payoff: Tensor, allocation: Dict[int, int]):
        super().__init__(agent_pool, teams, payoff, allocation)
        self._sampling_strategy = PrioritizedFictitiousSelfPlay()

    def get_match(self, home_team: Team) -> Union[None, Tuple[Team, OrderedDict]]:
        idx = self.get_instance_id(home_team)
        opponents = self._agent_pool.teams
        games = self._payoff[idx, :, PayoffEntry.GAMES]
        no_game_mask = games == 0.0
        wins = self._payoff[idx, :, PayoffEntry.WIN]
        draws = self._payoff[idx, :, PayoffEntry.DRAW]
        win_rates = (wins + 0.5 * draws) / games
        win_rates[no_game_mask] = .5 # If no games played we divided by 0 -> NaN -> replace with .5
        chosen: Team = self._sampling_strategy.sample(opponents=opponents, prio_measure=win_rates)
        chosen_idx = self.get_instance_id(chosen)
        self._payoff[idx, chosen_idx, PayoffEntry.MATCHES] += 1
        return chosen, self._agent_pool[chosen]


class UniformMatchmaking(Matchmaking):
    def __init__(self, agent_pool: AgentPool, allocation: Dict[int, int], payoff: Tensor, teams: List[Team]):
        super().__init__(agent_pool, teams, payoff, allocation)

    def get_match(self, home_team: Team) -> Union[None, Tuple[Team, OrderedDict]]:
        idx = self.get_instance_id(home_team)
        matches = self._payoff[idx, :, PayoffEntry.MATCHES]
        # matches = matches[matches != idx] # Remove play against one self
        chosen_idx = th.argmin(matches).item()  # Get adversary we played the least
        self._payoff[idx, chosen_idx, PayoffEntry.MATCHES] += 1
        team = self.get_team(chosen_idx)
        return team, self._agent_pool[team]


class RandomMatchmaking(Matchmaking):

    def __init__(self, agent_pool: AgentPool, allocation: Dict[int, int], teams: List[Team], round_limit: int = 5,
                 payoff: Tensor = None):
        """
        Matchmaking to return a random adversary team agent bound to a round limit.
        :param agent_pool:
        :param round_limit:
        :param time_limit:
        """
        super().__init__(agent_pool, teams, payoff, allocation)
        self.round_limit = round_limit
        self.current_round = 0

    def get_match(self, home_team: Team) -> Union[None, Tuple[Team, OrderedDict]]:
        """
        Find a opponent for the given team using various methods.
        :param home_team:
        :return:
        """
        if self.current_round >= self.round_limit:
            return None

        if not self._agent_pool.can_sample():
            return home_team, self._agent_pool[home_team]  # Self-Play if no one available
        self.current_round += 1
        return self._agent_pool.sample()


REGISTRY = {
    "uniform": UniformMatchmaking,
    "random": RandomMatchmaking,
    "pfsp": PFSPMatchmaking
}
