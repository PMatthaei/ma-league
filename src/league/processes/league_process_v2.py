import time
from logging import warning
from torch.multiprocessing import Process, Barrier, Queue

from types import SimpleNamespace
from typing import Dict, Union, Tuple, List

from league.components.agent_pool import AgentPool
from league.components.matchmaking import Matchmaking
from league.components.payoff import PayoffEntry
from league.roles.alphastar.main_player import MainPlayer
from league.roles.players import Player
from league.utils.commands import CloseLeagueProcessCommand, PayoffUpdateCommand, CheckpointCommand
from league.utils.team_composer import Team
from runs.league_play_run import LeaguePlayRun
from custom_logging.logger import MainLogger
from runs.normal_play_run import NormalPlayRun


class LeagueProcessV2(Process):
    def __init__(self,
                 agent_pool: AgentPool,
                 matchmaking: Matchmaking,
                 home_team: Team,
                 queue: Tuple[Queue, Queue],
                 args: SimpleNamespace,
                 logger: MainLogger,
                 sync_barrier: Barrier):
        super().__init__()
        self._args = args
        self._logger = logger

        self._home_team = home_team
        self._away_team = None
        self._agent_pool: AgentPool = agent_pool  # Process private copy of the agent pool
        self._matchmaking: Matchmaking = matchmaking

        self._in_queue, self._out_queue = queue  # In- and Outgoing Communication
        self._sync_barrier = sync_barrier  # Use to sync with other processes

        self.terminated: bool = False

        self._play = None

        # Register home team
        self._args.env_args['match_build_plan'][0]['units'] = self._home_team.units
        self._args.env_args['match_build_plan'][1]['units'] = self._home_team.units

    def run(self) -> None:
        self._play = NormalPlayRun(args=self._args, logger=self._logger, on_episode_end=self._provide_result)
        self._share_agent()

        start_time = time.time()
        end_time = time.time()

        while end_time - start_time <= self._args.league_runtime_hours * 60 * 60:
            self._away_team, away_agent = self._matchmaking.get_match(self._home_team)

            # Register away team
            self._args.env_args['match_build_plan'][1]['units'] = self._away_team.units

            self._play.set_away_agent(away_agent)
            self._play.start(play_time=self._args.league_play_time_mins * 60)

            end_time = time.time()

            # Share agent after training to make its current state accessible to other processes
            self._share_agent()

        self._request_close()

    def _get_shared_agent(self, team: Team):
        return self._agent_pool[team]

    def _share_agent(self):
        self._agent_pool[self._home_team] = self._play.home_mac.agent
        self._sync_barrier.wait()  # Wait until every process finished to sync

    def _provide_result(self, env_info: Dict):
        """
        Send the result of an episode the the central coordinator for processing
        :param env_info:
        :return:
        """
        result = self._extract_result(env_info)
        data = ((self._home_team.id_, self._away_team.id_), result)
        cmd = PayoffUpdateCommand(origin=self._home_team.id_, data=data)
        self._in_queue.put(cmd)

    def _request_close(self):
        cmd = CloseLeagueProcessCommand(origin=self._home_team.id_)
        self._in_queue.put(cmd)

    def _extract_result(self, env_info: dict):
        draw = env_info["draw"]
        battle_won = env_info["battle_won"]
        if draw or all(battle_won) or not any(battle_won):
            # Draw if all won or all lost
            result = PayoffEntry.DRAW
        elif battle_won[self._play.stepper.policy_team_id]:
            result = PayoffEntry.WIN
        else:
            result = PayoffEntry.LOSS
        return result
