import time
from logging import warning
from torch.multiprocessing import Process

from torch.multiprocessing import Barrier, Queue

from types import SimpleNamespace
from typing import Dict, Union

from league.roles.players import Player, MainPlayer
from runs.league_play_run import LeaguePlayRun
from utils.logging import LeagueLogger


class LeagueProcess(Process):
    def __init__(self, home: Player, barrier: Barrier, queue: Queue, args: SimpleNamespace, logger: LeagueLogger):
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
        self._barrier = barrier
        self._queue = queue
        self._args = args
        self._logger = logger

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
            # TODO load away players learner
            #self._play.away_learner.load_models(self._away_player.latest)

            self._logger.console_logger.info(str(self))
            play_time_seconds = self._args.league_play_time_mins * 60
            self._play.start(play_time=play_time_seconds)
            end_time = time.time()

        self._close()

    def _setup(self):
        # Create play
        self._play = LeaguePlayRun(args=self._args, logger=self._logger, episode_callback=self._episode_callback)
        # Provide learner to the shared home player
        self._send_learner()
        if isinstance(self._home_player, MainPlayer):
            self._checkpoint_agent()  # MainPlayers are initially added as historical players
        self._barrier.wait()  # Synchronize - Wait until all processes performed setup

    def _send_learner(self):
        self._queue.put({"learner": self._play.home_learner, "player_id": self._home_player.player_id})

    def _checkpoint_agent(self):
        self._queue.put({"checkpoint": self._home_player.player_id})

    def _episode_callback(self, env_info: Dict):
        result = self._get_result(env_info)
        self._queue.put({"result": (self._home_player.player_id, self._away_player.player_id, result)})

    def _close(self):
        self._queue.put({"close": self._home_player.player_id})

    def __str__(self):
        return f"LeaguePlayRun - {self._home_player.prettier()} playing against opponent {self._away_player.prettier()}"

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
