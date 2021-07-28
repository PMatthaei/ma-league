from typing import Tuple, Dict, OrderedDict

from torch.multiprocessing import Barrier
from torch.multiprocessing.queue import Queue

from league.components import Matchmaker, PayoffEntry
from league.processes.agent_pool_instance import clone_state_dict
from league.processes.interfaces.experiment_process import ExperimentInstance
from league.utils.commands import CloseCommunicationCommand, AgentParamsUpdateCommand, AgentParamsGetCommand
from league.components.team_composer import Team


class LeagueExperimentInstance(ExperimentInstance):

    def __init__(self,
                 matchmaking: Matchmaker,
                 home_team: Team,
                 communication: Tuple[int, Tuple[Queue, Queue]],
                 sync_barrier: Barrier, **kwargs):
        """
        The process is running a single Multi-Agent training and handles communication with the central components.
        League-Play is a form of NormalPlay where the opponent can be swapped out from a pool of agents (=league).
        The opponent is fixed and is therefore not learning to prevent non-stationary environment.
        Opponents are sampled via Self-Play Sampling such as FSP, PFSP or SP.

        Opponent sampling is decided via a matchmaking component.

        :param agent_pool:
        :param matchmaking:
        :param home_team:
        :param communication:
        :param sync_barrier:
        """
        super(LeagueExperimentInstance, self).__init__(**kwargs)

        self._home_team: Team = home_team
        self._adversary_team: Team = None
        self._adversary_idx: int = None
        self._matchmaker: Matchmaker = matchmaking

        self._comm_id = communication[0]
        self._in_queue, self._out_queue = communication[1]  # In- and Outgoing Communication
        self._sync_barrier = sync_barrier  # Use to sync with other processes

        self.terminated: bool = False

        self._experiment = None

    def _run_experiment(self):
        raise NotImplementedError("Please implement a league experiment.")

    @property
    def home_agent_state(self) -> OrderedDict:
        return self._experiment.home_mac.agent.state_dict()

    def _configure_experiment(self, home: Team, ai: bool, away: Team = None):
        # In case this process needs to save models -> modify token
        self._args.env_args['match_build_plan'][0]['units'] = home.units  # mirror if no away units passed
        self._args.env_args['match_build_plan'][1]['units'] = home.units if away is None else away.units
        self._args.env_args['match_build_plan'][0]['is_scripted'] = False
        self._args.env_args['match_build_plan'][1]['is_scripted'] = ai

    def _get_agent_params(self, team: Team = None) -> Tuple[Team, OrderedDict]:
        tid = self._home_team.id_ if team is None else team.id_
        cmd = AgentParamsGetCommand(origin=self._comm_id, data=tid)
        self._in_queue.put(cmd)
        tid, agent_params = self._out_queue.get()
        return tid, agent_params

    def _share_agent_params(self, agent: OrderedDict, team: Team = None):
        """
        Share agent
        and wait until every process finished to sharing to ensure every agent is up-to-date before next match.
        This will currently require each instance to share in order to release barrier.
        :param agent:
        :return:
        """
        tid: int = self._home_team.id_ if team is None else team.id_
        agent_clone = clone_state_dict(agent)
        cmd = AgentParamsUpdateCommand(origin=self._comm_id, data=(tid, agent_clone))
        self._in_queue.put(cmd)
        del agent_clone
        self._ack()
        self._sync_barrier.wait() if self._sync_barrier is not None else None

    def _extract_result(self, env_info: dict):
        policy_team_id = self._experiment.stepper.policy_team_id
        draw = env_info["draw"]
        battle_won = env_info["battle_won"]
        if draw or all(battle_won) or not any(battle_won):
            result = PayoffEntry.DRAW  # Draw if all won or all lost
        elif battle_won[policy_team_id]:  # Policy team(= home team) won
            result = PayoffEntry.WIN
        else:
            result = PayoffEntry.LOSS  # Policy team(= home team) lost
        return result

    def _update_payoff(self, env_info: Dict):
        """
        Send the result of an episode the the central coordinator for processing.
        :param env_info:
        :return:
        """
        result = self._extract_result(env_info)
        assert self._adversary_idx is not None, "Ensure to set the opponents instance idx on matchmaking."
        self._matchmaker.payoff[self.idx, self._adversary_idx, result] += 1

    def _request_close(self):
        """
        Close the communication channel to the agent pool
        :return:
        """
        self._sync_barrier.wait() if self._sync_barrier is not None else None
        cmd = CloseCommunicationCommand(origin=self._comm_id)
        self._in_queue.put(cmd)
        self._ack()
        self._in_queue.close()
        self._out_queue.close()

    def _ack(self):
        ack = self._out_queue.get()
        if ack is not None:
            raise Exception("Illegal ACK")

    def __str__(self):
        return f"{self.__class__.__name__}: " + str(self.idx) + " with " + str(self._home_team)