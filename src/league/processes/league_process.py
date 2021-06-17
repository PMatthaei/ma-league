import time
from logging import warning
from torch.multiprocessing import Process

from torch.multiprocessing import Barrier, Queue

from types import SimpleNamespace
from typing import Dict, Union, Tuple, List

from league.components.payoff import MatchResult
from league.roles.alphastar.main_player import MainPlayer
from league.roles.players import Player
from league.utils.commands import CloseLeagueProcessCommand, PayoffUpdateCommand, CheckpointCommand
from runs.league_play_run import LeaguePlayRun
from custom_logging.logger import MainLogger


class LeagueProcess(Process):
    def __init__(self,
                 players: List[Player],
                 player_id: int,
                 queue: Tuple[Queue, Queue],
                 args: SimpleNamespace,
                 logger: MainLogger,
                 barrier: Barrier):
        """
        The process is running a single League-Play and handles communication with the central component.
        League-Play is a form of NormalPlay where the opponent can be swapped out from a pool of agents (=league).
        The opponent is fixed and is therefore not learning to prevent non-stationary environment.
        Opponents are sampled via Self-Play Sampling such as FSP, PFSP or SP.
        :param players:
        :param player_id:
        :param queue:
        :param args:
        :param logger:
        :param barrier:
        """
        super().__init__()
        self._player_id = player_id
        self._shared_players = players
        self._home = self._shared_players[self._player_id]  # Process private copy of the player
        self._in_queue, self._out_queue = queue
        self._args = args
        self._logger = logger
        self._setup_barrier = barrier
        self._away: Union[Player, None] = None
        self.terminated: bool = False
        # Supply team to match plan
        self._register_team()
        self._play = LeaguePlayRun(args=self._args, logger=self._logger, episode_callback=self._provide_episode_result)

    def run(self) -> None:
        self._share_agent()

        # Wait at barrier until every league process performed the sharing step before the next step
        self._setup_barrier.wait()

        # Progress to save initial checkpoint of agents after all runs performed setup
        if isinstance(self._home, MainPlayer):  # TODO: Allow for different kinds of initial historical players
            self._request_checkpoint()  # MainPlayers are initially added as historical players

        start_time = time.time()
        end_time = time.time()

        while end_time - start_time <= self._args.league_runtime_hours * 60 * 60:
            self._away, flag = self._home.get_match()
            away_agent = self._get_shared_agent(self._away)
            if away_agent is None:
                warning("No Opponent was found.")
                continue
            self._play.set_away_agent(away_agent)

            # Start training against new opponent
            self._logger.console_logger.info(str(self))
            play_time_seconds = self._args.league_play_time_mins * 60
            # TODO: Modify environment to use away agents team -> stepper needs new env
            self._register_team(self._away)
            self._play.stepper.rebuild_env(self._args.env_args) # Rebuild env with new team
            self._play.start(play_time=play_time_seconds)
            end_time = time.time()

        self._request_close()

    def _register_team(self, player: Player = None):
        """
        Registers the team within the argument dict upon which the environment will be built
        :param player:
        :return:
        """
        match_plan = self._args.env_args['match_build_plan']
        if player is None:
            match_plan[0]['units'] = self._home.team['units']
            match_plan[1]['units'] = self._home.team['units']
        elif player == self._home:
            match_plan[0]['units'] = player.team['units']
        elif player == self._away:
            match_plan[1]['units'] = player.team['units']

    def _get_shared_agent(self, player: Player):
        return self._shared_players[player.id_].agent

    def _share_agent(self):
        # Fetch agent from training, set as players agent and insert into shared memory list
        self._home.agent = self._play.home_mac.agent
        self._shared_players[self._player_id] = self._home

    def _request_checkpoint(self):
        """
        Request to checkpoint the current version of the agent, residing in the shared memory list.
        If the current state of the agent was not propagated to the shared memory,
        the agent checkpointed is not up-to-date!
        :return:
        """
        cmd = CheckpointCommand(origin=self._player_id)
        self._in_queue.put(cmd)

    def _provide_episode_result(self, env_info: Dict):
        """
        Send the result of an episode the the central coordinator for processing
        :param env_info:
        :return:
        """
        result = self._get_result(env_info)
        data = ((self._home, self._away), result)
        cmd = PayoffUpdateCommand(origin=self._player_id, data=data)
        self._in_queue.put(cmd)

    def _request_close(self):
        cmd = CloseLeagueProcessCommand(origin=self._player_id)
        self._in_queue.put(cmd)

    def __str__(self):
        return f"LeaguePlayRun - {self._home.prettier()} playing against {self._away.prettier()}"

    @staticmethod
    def _get_result(env_info):
        draw = env_info["draw"]
        battle_won = env_info["battle_won"]
        if draw or all(battle_won) or not any(battle_won):
            # Draw if all won or all lost
            result = MatchResult.DRAW
        elif battle_won[
            0]:  # TODO BUG! the battle won bool at position 0 does not have to be the one of the home player
            result = MatchResult.WIN
        else:
            result = MatchResult.LOSS
        return result
