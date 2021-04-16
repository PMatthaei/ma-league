from logging import warning
from multiprocessing.dummy import Process

from multiprocessing.connection import Connection

from types import SimpleNamespace
from typing import Dict

from league.roles.players import Player
from runs.self_play_run import SelfPlayRun
from utils.logging import LeagueLogger


class LeagueRun(Process):
    def __init__(self, home: Player, conn: Connection, args: SimpleNamespace, logger: LeagueLogger):
        super().__init__()
        self.home = home
        self.conn = conn
        self.args = args
        self.logger = logger

        self.away = None
        self.terminated = False

    def run(self) -> None:
        while not self.terminated:
            # Generate new opponent to train against and load his current checkpoint
            self.away, flag = self.home.get_match()  # TODO load away and home into selfplayrun
            if self.away is None:
                warning("Opponent was none")
                continue

            self.logger.console_logger.info(self._get_match_str())

            play = SelfPlayRun(args=self.args, logger=self.logger, episode_callback=self.send_episode_result)
            play.start()

        self.send_run_finished()

    def send_episode_result(self, env_info: Dict):
        result = self._get_result(env_info)
        self.conn.send({"result": (self.home.player_id, self.away.player_id, result)})

    def send_run_finished(self):
        self.conn.send({"close": self.home.player_id})
        self.conn.close()

    def _get_match_str(self):
        player_str = f"{type(self.home).__name__} {self.home.player_id}"
        opponent_str = f"{type(self.away).__name__} {self.away.player_id} "
        return f"{player_str} playing against opponent {opponent_str} in Process {self.home.player_id}"

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
