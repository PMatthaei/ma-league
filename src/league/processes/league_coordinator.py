import logging

from torch.multiprocessing import Process, Queue
from typing import List, Tuple

from league.components.payoff import Payoff
from league.roles.players import Player


class LeagueCoordinator(Process):
    def __init__(self, logger, players: List[Player], queues: Tuple[List, List], payoff: Payoff):
        """
        Handles messages sent from league sub processes to the main league process.
        :param league:
        :param connections:
        """
        super().__init__()
        self._in_queues, self._out_queues = queues
        self._players = players
        self._payoff = payoff
        self._closed = []
        self.logger = logger.console_logger

    def run(self) -> None:
        # Receive messages from all processes over their connections
        while len(self._closed) != len(self._in_queues):
            for q in self._in_queues:
                if not q.empty():
                    self._handle_message(q)

    def _handle_message(self, queue: Queue):
        msg = queue.get_nowait()
        if "close" in msg:
            self.logger.info(f"Closing connection to process {msg['close']}")
            queue.close()
            self._closed.append(msg['close'])
        elif "checkpoint" in msg:  # checkpoint player initiated from a connected process
            self._checkpoint(msg)
        elif "result" in msg:
            self._save_outcome(msg)
        elif "learner" in msg:
            self._update_learner(msg)
        elif "away" in msg:
            self._send_away_learner(msg)
        else:
            raise Exception("Unknown message.")

    def _checkpoint(self, msg):
        """
        Save a checkpoint of the agent with the ID provided in the message.
        :param msg:
        :return:
        """
        idx = msg["checkpoint"]
        self.logger.info(f"Create historical player of player {idx}")
        self._players.append(self._players[idx].checkpoint())
        # TODO maybe this does not have to be a message

    def _save_outcome(self, msg):
        """
        Update the payoff matrix. After each play outcome check if the learning (=home) player should be checkpointed
        :param msg:
        :return:
        """
        home, away, outcome = msg["result"]
        home_player, _ = self._payoff.update(home, away, outcome)
        if home_player.ready_to_checkpoint():  # Auto-checkpoint player
            self._players.append(self._players[home_player].checkpoint())

    def _update_learner(self, msg):
        player_id = msg["player_id"]
        learner = msg["learner"]
        self.logger.info(f"Received learner {learner}")
        # --- ! Do not change this assignment
        player = self._players[player_id]
        player.learner = learner
        self._players[player_id] = player
        # --- ! Do not change this assignment
        self.logger.info(f"Updated learner of player {player_id}")
        self.logger.info(f"{self._players[player_id].learner}")
        self._out_queues[player_id].put({"updated": True, "player_id": player_id})

    def _send_away_learner(self, msg):
        player_id = msg["player_id"]
        away_id = msg["away"]
        self._out_queues[player_id].put(self._players[away_id].learner)
