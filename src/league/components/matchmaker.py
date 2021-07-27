import random
from typing import Tuple, Dict, Union, List, OrderedDict
from torch.multiprocessing.queue import Queue

import torch as th
from torch import Tensor

from league.components import PayoffEntry
from league.components.self_play import PFSPSampling, FSPSampling
from league.components.commands import AgentPoolGetCommand, CloseCommunicationCommand
from league.components.team_composer import Team


def dd():
    return 0


class Matchmaker:

    def __init__(self, communication: Tuple[int, Tuple[Queue, Queue]], teams: List[Team], payoff: Tensor):
        self._in_q, self._out_q = communication[1]
        self._comm_id = communication[0]
        self._teams_dict = {team.id_: team for team in teams}
        self._instance_to_tid = {team.id_: idx for idx, team in enumerate(teams)}
        self._tid_to_instance = {v: k for k, v in self._instance_to_tid.items()}
        self.payoff = payoff

    def get_agents(self):
        cmd = AgentPoolGetCommand(origin=self._comm_id)
        self._in_q.put(cmd)
        pool = self._out_q.get()
        return pool

    def disconnect(self):
        cmd = CloseCommunicationCommand(origin=self._comm_id)
        self._in_q.put(cmd)
        ack = self._out_q.get()
        if ack is not None:
            raise Exception("Illegal ACK")

    def get_instance_id(self, team: Team) -> int:
        return self._instance_to_tid[team.id_]

    def get_team(self, instance_id: int = None, tid=None) -> Team:
        if tid is None:
            tid = self._tid_to_instance[instance_id]
        return self._teams_dict[tid]

    def get_match(self, home_team: Team) -> Union[None, Tuple[int, Team, OrderedDict]]:
        raise NotImplementedError()


class PFSPMatchmaking(Matchmaker):
    def __init__(self, communication: Tuple[int, Tuple[Queue, Queue]], teams: List[Team], payoff: Tensor):
        super().__init__(communication, teams, payoff)
        self._sampling_strategy = PFSPSampling()

    def get_match(self, home_team: Team) -> Union[None, Tuple[int, Team, OrderedDict]]:
        home_instance = self.get_instance_id(home_team)
        opponents = self.get_agents()
        win_rates = self._win_rates(home_instance)
        chosen_tid: int = self._sampling_strategy.sample(opponents=list(opponents.keys()), prio_measure=win_rates)
        chosen_idx = self._instance_to_tid[chosen_tid]
        team = self.get_team(tid=chosen_tid)
        self.payoff[home_instance, chosen_idx, PayoffEntry.MATCHES] += 1
        return chosen_idx, team, opponents[chosen_tid]

    def _win_rates(self, idx):
        games = self.payoff[idx, :, PayoffEntry.GAMES]
        no_game_mask = games == 0.0
        wins = self.payoff[idx, :, PayoffEntry.WIN]
        draws = self.payoff[idx, :, PayoffEntry.DRAW]
        win_rates = (wins + 0.5 * draws) / games
        win_rates[no_game_mask] = .5  # If no games played we divided by 0 -> NaN -> replace with .5
        return win_rates


class FSPMatchmaking(Matchmaker):
    def __init__(self, communication: Tuple[int, Tuple[Queue, Queue]], teams: List[Team], payoff: Tensor):
        super().__init__(communication, teams, payoff)
        self._sampling_strategy = FSPSampling()

    def get_match(self, home_team: Team) -> Union[None, Tuple[int, Team, OrderedDict]]:
        home_instance = self.get_instance_id(home_team)
        opponents = self.get_agents()
        chosen_tid: int = self._sampling_strategy.sample(opponents=list(opponents.keys()))
        chosen_idx = self._instance_to_tid[chosen_tid]
        team = self.get_team(tid=chosen_tid)
        self.payoff[home_instance, chosen_idx, PayoffEntry.MATCHES] += 1
        return chosen_idx, team, opponents[chosen_tid]


class BalancedMatchmaking(Matchmaker):

    def __init__(self, communication: Tuple[int, Tuple[Queue, Queue]], teams: List[Team], payoff: Tensor):
        super().__init__(communication, teams, payoff)
        super().__init__(communication, teams, payoff)

    def get_match(self, home_team: Team) -> Union[None, Tuple[int, Team, OrderedDict]]:
        idx = self.get_instance_id(home_team)
        agents = self.get_agents()
        matches = self.payoff[idx, :, PayoffEntry.MATCHES]
        # matches = matches[matches != idx] # Remove play against one self
        chosen_idx = th.argmin(matches).item()  # Get adversary we played the least
        self.payoff[idx, chosen_idx, PayoffEntry.MATCHES] += 1
        team = self.get_team(chosen_idx)
        return chosen_idx, team, agents[team]


class RandomMatchmaking(Matchmaker):
    def __init__(self, communication: Tuple[int, Tuple[Queue, Queue]], teams: List[Team], payoff: Tensor):
        super().__init__(communication, teams, payoff)
        super().__init__(communication, teams, payoff)

    def get_match(self, home_team: Team) -> Union[None, Tuple[int, Team, OrderedDict]]:
        idx = self.get_instance_id(home_team)

        agents = self.get_agents()
        ids = list(agents.keys())
        random_id = random.choice(ids)
        chosen_idx = self._instance_to_tid[random_id]

        self.payoff[idx, chosen_idx, PayoffEntry.MATCHES] += 1
        return chosen_idx, self.get_team(tid=random_id), agents[random_id]


REGISTRY = {
    "pfsp": PFSPMatchmaking,
    "uniform": BalancedMatchmaking,
    "random": RandomMatchmaking,
    "fsp": FSPMatchmaking
}
