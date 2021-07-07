from torch.multiprocessing import Process, Barrier, Queue

from types import SimpleNamespace
from typing import Dict, Tuple

from league.components.agent_pool import AgentPool
from league.components.matchmaking import Matchmaking
from league.processes.utils import extract_result
from league.utils.commands import CloseLeagueProcessCommand, PayoffUpdateCommand
from league.utils.team_composer import Team
from modules.agents import Agent
from custom_logging.logger import MainLogger
from runs.train.normal_play_run import NormalPlayRun


class EnsembleLeagueProcess(Process):
    def __init__(self,
                 agent_pool: AgentPool,
                 matchmaking: Matchmaking,
                 home_team: Team,
                 queue: Tuple[Queue, Queue],
                 args: SimpleNamespace,
                 logger: MainLogger,
                 sync_barrier: Barrier):
        """
        The process is running a single League-Play and handles communication with the central components.
        League-Play is a form of NormalPlay where the opponent can be swapped out from a pool of agents (=league).
        The opponent is fixed and is therefore not learning to prevent non-stationary environment.
        Opponents are sampled via Self-Play Sampling such as FSP, PFSP or SP.

        Opponent sampling is decided via a matchmaking component.

        :param agent_pool:
        :param matchmaking:
        :param home_team:
        :param queue:
        :param args:
        :param logger:
        :param sync_barrier:
        """
        super().__init__()
        self._args = args
        self._logger = logger

        self._home_team: Team = home_team
        self._away_team: Team = None
        self._agent_pool: AgentPool = agent_pool  # Process private copy of the agent pool
        self._matchmaking: Matchmaking = matchmaking

        self._in_queue, self._out_queue = queue  # In- and Outgoing Communication
        self._sync_barrier = sync_barrier  # Use to sync with other processes

        self.terminated: bool = False

        self._play = None

        self._ensemble = None

    def run(self) -> None:
        # Initial play to train policy of the team against AI against mirrored team -> Performed for each team
        self._configure_play(home=self._home_team, ai_opponent=True)
        self._play = NormalPlayRun(args=self._args, logger=self._logger)
        self._play.start(play_time_seconds=self._args.league_play_time_mins * 60)
        self._share_agent(agent=self._play.home_mac.agent)

        # Fetch agents from other teams trained previously and combine them into an ensemble
        self._ensemble = self._matchmaking.get_ensemble(self._home_team)
        self._play = NormalPlayRun(args=self._args, logger=self._logger)
        self._play.build_inference_mac(self._ensemble)
        self._play.evaluate_sequential(test_n_episode=200)
        self._play.save_models()
        # Share agent after training to make its current state accessible to other processes
        self._share_agent(agent=self._play.home_mac.agent)

        self._request_close()

    def _configure_play(self, home: Team, away: Team = None, ai_opponent=False):
        # In case this process needs to save models -> modify token
        self._args.unique_token += f"_team_{self._home_team.id_}"
        self._args.env_args['match_build_plan'][0]['units'] = home.units  # mirror if no away units passed
        self._args.env_args['match_build_plan'][1]['units'] = home.units if away is None else away.units
        self._args.env_args['match_build_plan'][1]['is_scripted'] = ai_opponent

    def _get_shared_agent(self, team: Team):
        return self._agent_pool[team]

    def _share_agent(self, agent: Agent):
        self._agent_pool[self._home_team] = agent
        # Wait until every process finished to share the agent to ensure every agent is up-to-date before next match
        self._sync_barrier.wait()

    def _provide_result(self, env_info: Dict):
        """
        Send the result of an episode the the central coordinator for processing.
        :param env_info:
        :return:
        """
        result = extract_result(env_info, self._play.stepper.policy_team_id)
        data = ((self._home_team.id_, self._away_team.id_), result)
        cmd = PayoffUpdateCommand(origin=self._home_team.id_, data=data)
        self._in_queue.put(cmd)

    def _request_close(self):
        cmd = CloseLeagueProcessCommand(origin=self._home_team.id_)
        self._in_queue.put(cmd)
