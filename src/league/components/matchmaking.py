import random
from typing import Tuple, Dict, Union, List, OrderedDict
from torch.multiprocessing import Queue

import torch as th
from torch import Tensor

from league.components import PayoffEntry
from league.components.self_play import PrioritizedFictitiousSelfPlay, FictitiousSelfPlay
from league.utils.commands import AgentPoolGetCommand, CloseCommunicationCommand
from league.utils.team_composer import Team


def dd():
    return 0


class Matchmaking:

    def __init__(self, comm_id: int, communication: Tuple[Queue, Queue], teams: List[Team], payoff: Tensor,
                 allocation: Dict[int, int]):
        self._in_q, self._out_q = communication
        self._comm_id = comm_id
        self._allocation = allocation
        self._payoff = payoff
        self._teams = teams

    def get_agents(self):
        cmd = AgentPoolGetCommand(origin=self._comm_id)
        self._in_q.put(cmd)
        pool = self._out_q.get()
        return pool

    def disconnect(self):
        cmd = CloseCommunicationCommand(origin=self._comm_id)
        self._in_q.put(cmd)

    def get_instance_id(self, team: Team) -> int:
        return self._allocation[team.id_]

    def get_team(self, instance_id: int = None, tid=None) -> Team:
        if tid is None:
            _inv_allocation = {v: k for k, v in self._allocation.items()}
            tid = _inv_allocation[instance_id]
        return next((team for team in self._teams if team.id_ == tid), None)

    def get_match(self, home_team: Team) -> Union[None, Tuple[Team, OrderedDict]]:
        raise NotImplementedError()


class PFSPMatchmaking(Matchmaking):
    def __init__(self, comm_id: int, communication: Tuple[Queue, Queue], teams: List[Team], payoff: Tensor, allocation: Dict[int, int]):
        super().__init__(comm_id, communication, teams, payoff, allocation)
        self._sampling_strategy = PrioritizedFictitiousSelfPlay()

    def get_match(self, home_team: Team) -> Union[None, Tuple[Team, OrderedDict]]:
        idx = self.get_instance_id(home_team)
        opponents = self.get_agents()
        games = self._payoff[idx, :, PayoffEntry.GAMES]
        no_game_mask = games == 0.0
        wins = self._payoff[idx, :, PayoffEntry.WIN]
        draws = self._payoff[idx, :, PayoffEntry.DRAW]
        win_rates = (wins + 0.5 * draws) / games
        win_rates[no_game_mask] = .5  # If no games played we divided by 0 -> NaN -> replace with .5
        chosen: Team = self._sampling_strategy.sample(opponents=opponents, prio_measure=win_rates)
        chosen_idx = self.get_instance_id(chosen)
        self._payoff[idx, chosen_idx, PayoffEntry.MATCHES] += 1
        return chosen, opponents[chosen]


class FSPMatchmaking(Matchmaking):
    def __init__(self, comm_id: int, communication: Tuple[Queue, Queue], teams: List[Team], payoff: Tensor, allocation: Dict[int, int]):
        super().__init__(comm_id, communication, teams, payoff, allocation)
        self._sampling_strategy = FictitiousSelfPlay()

    def get_match(self, home_team: Team) -> Union[None, Tuple[Team, OrderedDict]]:
        idx = self.get_instance_id(home_team)
        opponents = self.get_agents()
        chosen: Team = self._sampling_strategy.sample(opponents=opponents)
        chosen_idx = self.get_instance_id(chosen)
        self._payoff[idx, chosen_idx, PayoffEntry.MATCHES] += 1
        return chosen, opponents[chosen]


class BalancedMatchmaking(Matchmaking):
    def __init__(self, comm_id: int, communication: Tuple[Queue, Queue], teams: List[Team], payoff: Tensor, allocation: Dict[int, int]):
        super().__init__(comm_id, communication, teams, payoff, allocation)

    def get_match(self, home_team: Team) -> Union[None, Tuple[Team, OrderedDict]]:
        idx = self.get_instance_id(home_team)
        agents = self.get_agents()
        matches = self._payoff[idx, :, PayoffEntry.MATCHES]
        # matches = matches[matches != idx] # Remove play against one self
        chosen_idx = th.argmin(matches).item()  # Get adversary we played the least
        self._payoff[idx, chosen_idx, PayoffEntry.MATCHES] += 1
        team = self.get_team(chosen_idx)
        return team, agents[team]


class RandomMatchmaking(Matchmaking):
    def __init__(self, comm_id: int, communication: Tuple[Queue, Queue], teams: List[Team], payoff: Tensor, allocation: Dict[int, int]):
        super().__init__(comm_id, communication, teams, payoff, allocation)

    def get_match(self, home_team: Team) -> Union[None, Tuple[Team, OrderedDict]]:
        idx = self.get_instance_id(home_team)

        agents = self.get_agents()
        ids = list(agents.keys())
        random_id = random.choice(ids)
        chosen_idx = self._allocation[random_id]

        self._payoff[idx, chosen_idx, PayoffEntry.MATCHES] += 1
        return self.get_team(tid=random_id), agents[random_id]


REGISTRY = {
    "uniform": BalancedMatchmaking,
    "pfsp": PFSPMatchmaking,
    "random": RandomMatchmaking,
    "fsp": FSPMatchmaking
}
