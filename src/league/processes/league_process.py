import time
from logging import warning
from torch.multiprocessing import Process

from torch.multiprocessing import Barrier, Queue

from types import SimpleNamespace
from typing import Dict, Union, Tuple, List

from league.components.payoff import MatchResult
from league.roles.players import Player, MainPlayer
from league.utils.commands import CloseLeagueProcessCommand, PayoffUpdateCommand, CheckpointLearnerCommand
from learners.learner import Learner
from runs.league_play_run import LeaguePlayRun
from utils.logging import LeagueLogger


def debug_learners(learners: List[Learner]):
    for learner in learners:
        print(f"{learner.name} at step {learner.trained_steps}")


class LeagueProcess(Process):
    def __init__(self,
                 players: List[Player],
                 player_id: int,
                 queue: Tuple[Queue, Queue],
                 args: SimpleNamespace,
                 logger: LeagueLogger,
                 barrier: Barrier):
        """
        LeaguePlay is a form of NormalPlay where the opponent can be swapped out from a pool of agents.
        This will cause the home player to adapt to multiple opponents but will also cause inter-non-stationarity since
        the opponent will become part of the environment.
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
        self._away_player: Union[Player, None] = None
        self.terminated: bool = False
        self._play = None

    def run(self) -> None:
        # Create play
        self._play = LeaguePlayRun(args=self._args, logger=self._logger, episode_callback=self._provide_episode_result)
        self._share_learner()

        # Wait at barrier until every league process performed the setup
        self._setup_barrier.wait()

        # Progress to form initial checkpoint agents after all runs performed setup
        if isinstance(self._home, MainPlayer):
            self._request_checkpoint()  # MainPlayers are initially added as historical players

        start_time = time.time()
        end_time = time.time()

        while end_time - start_time <= self._args.league_runtime_hours * 60 * 60:
            self._away_player, flag = self._home.get_match()
            away_learner = self._get_shared_learner(self._away_player)
            if away_learner is None:
                warning("No Opponent was found.")
                continue
            self._play.set_away_learner(away_learner)

            # Start training against new opponent
            self._logger.console_logger.info(str(self))
            play_time_seconds = self._args.league_play_time_mins * 60
            self._play.start(play_time=play_time_seconds, train_callback=debug_learners)
            end_time = time.time()

        self._request_close()

    def _get_shared_learner(self, player: Player):
        return self._shared_players[player.id_].learner

    def _share_learner(self):
        # Provide learner as part of the player and insert into shared memory list
        self._home.learner = self._play.home_learner
        self._shared_players[self._player_id] = self._home

    def _request_checkpoint(self):
        cmd = CheckpointLearnerCommand(origin=self._player_id)
        self._in_queue.put(cmd)

    def _provide_episode_result(self, env_info: Dict):
        result = self._get_result(env_info)
        data = ((self._player_id, self._away_player.id_), result)
        cmd = PayoffUpdateCommand(origin=self._player_id, data=data)
        self._in_queue.put(cmd)

    def _request_close(self):
        cmd = CloseLeagueProcessCommand(origin=self._player_id)
        self._in_queue.put(cmd)

    def __str__(self):
        return f"LeaguePlayRun - {self._home.prettier()} playing against {self._away_player.prettier()}"

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
