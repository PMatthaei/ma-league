import time
from torch.multiprocessing import Process, Barrier, Queue

from types import SimpleNamespace
from typing import Dict, Tuple

from league.components.agent_pool import AgentPool
from league.components.matchmaking import Matchmaking
from league.processes.training.utils import extract_result
from league.utils.commands import CloseLeagueProcessCommand, PayoffUpdateCommand
from league.utils.team_composer import Team
from runs.train.league_play_run import LeaguePlayRun
from custom_logging.logger import MainLogger
from runs.train.normal_play_run import NormalPlayRun


class MatchmakingLeagueProcess(Process):
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

    def run(self) -> None:
        # Initial play to train policy of the team against AI against mirrored team
        self._configure_play(home=self._home_team, ai_opponent=True)
        self._play = NormalPlayRun(args=self._args, logger=self._logger)
        self._play.start(play_time_seconds=self._args.league_play_time_mins * 60)
        self._share_agent()

        start_time = time.time()
        end_time = time.time()

        # Run real league play in self-play against pre-trained but fixed multi-agent policies
        while end_time - start_time <= self._args.league_runtime_hours * 60 * 60:
            self._away_team, away_agent = self._matchmaking.get_match(self._home_team)

            self._configure_play(home=self._home_team, away=self._away_team)

            self._play = LeaguePlayRun(args=self._args, logger=self._logger, on_episode_end=self._provide_result)
            self._play.build_ensemble_mac(agent=away_agent)
            self._play.start(play_time_seconds=self._args.league_play_time_mins * 60)

            end_time = time.time()

            # Share agent after training to make its current state accessible to other processes
            self._share_agent()

        self._request_close()

    def _configure_play(self, home: Team, away: Team = None, ai_opponent=False):
        self._args.env_args['match_build_plan'][0]['units'] = home.units # mirror if no away units passed
        self._args.env_args['match_build_plan'][1]['units'] = home.units if away is None else away.units
        self._args.env_args['match_build_plan'][1]['is_scripted'] = ai_opponent

    def _get_shared_agent(self, team: Team):
        return self._agent_pool[team]

    def _share_agent(self):
        self._agent_pool[self._home_team] = self._play.home_mac.agent
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
