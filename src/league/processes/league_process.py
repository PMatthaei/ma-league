import time
from logging import warning
from multiprocessing import Process

from multiprocessing.connection import Connection
from multiprocessing import Barrier

from types import SimpleNamespace
from typing import Dict

from league.roles.players import Player, MainPlayer
from runs.self_play_run import SelfPlayRun
from utils.logging import LeagueLogger


class LeagueRun(Process):
    def __init__(self, home: Player, barrier: Barrier, conn: Connection, args: SimpleNamespace, logger: LeagueLogger):
        super().__init__()
        self._home = home
        self._barrier = barrier
        self._conn = conn
        self._args = args
        self._logger = logger

        self._away: Player = None
        self.terminated: bool = False

    def run(self) -> None:
        self._setup()

        self._barrier.wait()  # Wait until all processes setup their checkpoints and/or learner

        start_time = time.time()
        end_time = time.time()

        while end_time - start_time <= self._args.league_runtime_hours * 60 * 60:
            # Generate new opponent to train against and load his current checkpoint
            self._away, flag = self._home.get_match()
            if self._away is None:
                warning("No Opponent was found.")
                continue

            self._logger.console_logger.info(str(self))

            self._play.away_learner = self._away.learner  # Provide away learner from the away player
            self._play.start(play_time=self._args.league_play_time_mins * 60)
            end_time = time.time()

        self._close()

    def _setup(self):
        # Create play
        self._play = SelfPlayRun(args=self._args, logger=self._logger, episode_callback=self._episode_callback)
        # Provide learner to the home player
        self._home.learner = self._play.home_learner
        if isinstance(self._home, MainPlayer):
            self.checkpoint_agent()  # MainPlayers are initially added as historical players

    def checkpoint_agent(self):
        self._conn.send({"checkpoint": self._home.player_id})

    def _episode_callback(self, env_info: Dict):
        result = self._get_result(env_info)
        self._conn.send({"result": (self._home.player_id, self._away.player_id, result)})

    def _close(self):
        self._conn.send({"close": self._home.player_id})
        self._conn.close()

    def __str__(self):
        return f"SelfPlayRun - {self._home.prettier()} playing against opponent {self._away.prettier()}"

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
