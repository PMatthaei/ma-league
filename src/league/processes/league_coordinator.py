from multiprocessing.connection import Connection
from multiprocessing import Process
from typing import List

from league.league import League


class LeagueCoordinator(Process):
    def __init__(self, league: League, connections: List[Connection]):
        """
        Handles messages sent from learner processes to the main league process.
        :param league:
        :param connections:
        """
        self.league = league
        self.conns = connections
        super().__init__()

    def run(self) -> None:
        # Receive messages from all processes over their connections
        while any(not conn.closed for conn in self.conns):
            for conn in self.conns:
                if not conn.closed and conn.poll():
                    self._handle_message(conn)

    def _handle_message(self, conn: Connection):
        msg = conn.recv()
        if "close" in msg:
            print(f"Closing connection to process {msg['close']}")
            conn.close()
        elif "checkpoint" in msg:  # checkpoint player initiated from a connected process
            self._checkpoint(msg)
        elif "result" in msg:
            self._save_outcome(msg)
        else:
            raise Exception("Unknown message.")

    def _checkpoint(self, msg):
        """
        Save a checkpoint of the agent with the ID provided in the message.
        :param msg:
        :return:
        """
        idx = msg["checkpoint"]
        agent = self.league.get_player(idx)
        self.league.add_player(agent.checkpoint())

    def _save_outcome(self, msg):
        """
        Update the payoff matrix. After each play outcome check if the learning (=home) player should be checkpointed
        :param msg:
        :return:
        """
        home, away, outcome = msg["result"]
        home_player, _ = self.league.update(home, away, outcome)
        if home_player.ready_to_checkpoint():
            self.league.add_player(home_player.checkpoint())
