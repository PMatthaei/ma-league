import time
from logging import warning
from torch.multiprocessing import Process

from torch.multiprocessing import Barrier, Queue

from types import SimpleNamespace
from typing import Dict, Union, Tuple

from league.roles.players import Player, MainPlayer
from learners.learner import Learner
from runs.league_play_run import LeaguePlayRun
from utils.logging import LeagueLogger


class LeagueProcess(Process):
    def __init__(self, home: Player, queue: Tuple[Queue, Queue], args: SimpleNamespace, logger: LeagueLogger,
                 barrier: Barrier):
        """
        LeaguePlay is a form of NormalPlay where the opponent can be swapped out from a pool of agents.
        This will cause the home player to adapt to multiple opponents but will also cause inter-non-stationarity since
        the opponent will become part of the environment.
        :param home:
        :param barrier:
        :param conn:
        :param args:
        :param logger:
        """
        super().__init__()
        self._home_player = home
        self._in_queue, self._out_queue = queue
        self._args = args
        self._logger = logger
        self._setup_barrier = barrier
        self._away_player: Union[Player, None] = None
        self.terminated: bool = False

    def run(self) -> None:
        self._setup()

        start_time = time.time()
        end_time = time.time()

        while end_time - start_time <= self._args.league_runtime_hours * 60 * 60:
            # Generate new opponent to train against and load his current checkpoint
            self._away_player, flag = self._home_player.get_match()
            if self._away_player is None:
                warning("No Opponent was found.")
                continue
            away_learner = self._get_learner(self._away_player.player_id)
            self._play.away_learner = away_learner

            self._logger.console_logger.info(str(self))
            play_time_seconds = self._args.league_play_time_mins * 60
            self._play.start(play_time=play_time_seconds)
            end_time = time.time()

        self._close()

    def _setup(self):
        # Create play
        self._play = LeaguePlayRun(args=self._args, logger=self._logger, episode_callback=self._send_episode_result)
        # Provide learner to the shared home player
        self._send_learner_update()

        # Progress to form initial checkpoint agents after learners arrived
        if isinstance(self._home_player, MainPlayer):
            self._send_checkpoint_request()  # MainPlayers are initially added as historical players

    def _send_learner_update(self):
        self._in_queue.put({"learner": self._play.home_learner, "player_id": self._home_player.player_id})
        acc = self._out_queue.get()
        self._logger.console_logger.info(acc)
        self._setup_barrier.wait()

    def _get_learner(self, idx: int):
        self._in_queue.put({"away": idx, "player_id": self._home_player.player_id})
        return self._out_queue.get()

    def _send_checkpoint_request(self):
        self._in_queue.put({"checkpoint": self._home_player.player_id})

    def _send_episode_result(self, env_info: Dict):
        result = self._get_result(env_info)
        self._in_queue.put({"result": (self._home_player.player_id, self._away_player.player_id, result)})

    def _close(self):
        self._in_queue.put({"close": self._home_player.player_id})

    def __str__(self):
        return f"LeaguePlayRun - {self._home_player.prettier()} playing against {self._away_player.prettier()}"

    @staticmethod
    def _get_result(env_info):
        draw = env_info["draw"]
        battle_won = env_info["battle_won"]
        if draw or all(battle_won) or not any(battle_won):
            # Draw if all won or all lost
            result = "draw"
        elif battle_won[0]:
            result = "won"
        else:
            result = "loss"
        return result
