from typing import Tuple, Dict

from torch.multiprocessing import Queue, Barrier

from league.components import AgentPool, Matchmaking
from league.processes.experiment_process import ExperimentProcess
from league.processes.training.utils import extract_result
from league.utils.commands import PayoffUpdateCommand, CloseLeagueProcessCommand
from league.utils.team_composer import Team
from modules.agents import AgentNetwork


class LeagueExperimentProcess(ExperimentProcess):

    def __init__(self,
                 idx,
                 params,
                 configs_dir,
                 log_dir,
                 agent_pool: AgentPool,
                 matchmaking: Matchmaking,
                 home_team: Team,
                 communication: Tuple[Queue, Queue],
                 sync_barrier: Barrier):
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
        super(LeagueExperimentProcess, self).__init__(idx, params, configs_dir, log_dir)

        self._home_team: Team = home_team
        self._away_team: Team = None
        self._agent_pool: AgentPool = agent_pool  # Process private copy of the agent pool
        self._matchmaking: Matchmaking = matchmaking

        self._in_queue, self._out_queue = communication  # In- and Outgoing Communication
        self._sync_barrier = sync_barrier  # Use to sync with other processes

        self.terminated: bool = False

        self._experiment = None

    def _run_experiment(self):
        raise NotImplementedError("Please implement a league experiment.")

    @property
    def home_agent(self) -> AgentNetwork:
        return self._experiment.home_mac.agent

    @property
    def shared_agent(self) -> Tuple[Team, AgentNetwork]:
        return self._home_team, self._agent_pool[self._home_team]

    def _get_shared_agent(self, team: Team):
        return self._agent_pool[team]

    def _share_agent(self, agent: AgentNetwork):
        self._agent_pool[self._home_team] = agent
        # Wait until every process finished to share the agent to ensure every agent is up-to-date before next match
        self._sync_barrier.wait()

    def _provide_result(self, env_info: Dict):
        """
        Send the result of an episode the the central coordinator for processing.
        :param env_info:
        :return:
        """
        result = extract_result(env_info, self._experiment.stepper.policy_team_id)
        data = ((self._home_team.id_, self._away_team.id_), result)
        cmd = PayoffUpdateCommand(origin=self._home_team.id_, data=data)
        self._in_queue.put(cmd)

    def _request_close(self):
        cmd = CloseLeagueProcessCommand(origin=self._home_team.id_)
        self._in_queue.put(cmd)
