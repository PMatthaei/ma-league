from enum import Enum
from multiprocessing.connection import Connection
from multiprocessing.dummy import Process
from typing import List

from league.components.coordinator import Coordinator


class ProcessMessageTypes(Enum):
    CLOSE = 0,
    RESULT = 1


class LeagueMessageHandler(Process):
    def __init__(self, coordinator, conns):
        """
        Handles messages sent from learner processes to the main league process.
        :param coordinator:
        :param conns:
        """
        self.coordinator = coordinator
        self.conns = conns
        super().__init__()

    def run(self) -> None:
        # Receive messages from all processes
        while any(not conn.closed for conn in self.conns):
            for conn in self.conns:
                if not conn.closed and conn.poll():
                    self._handle_message(conn)

    def _handle_message(self, conn: Connection):
        msg = conn.recv()
        if "close" in msg:
            print(f"Closing connection to process {msg['close']}")
            conn.close()
        elif "result" in msg:
            home, away, result = msg["result"]
            self.coordinator.send_outcome(home, away, result)
        else:
            raise Exception("Unknown message.")
