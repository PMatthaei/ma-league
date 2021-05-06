from torch.multiprocessing import Process, Queue
from typing import List

from league.league import League


class LeagueCoordinator(Process):
    def __init__(self, league: League, queues: List[Queue]):
        """
        Handles messages sent from league sub processes to the main league process.
        :param league:
        :param connections:
        """
        super().__init__()
        self.league = league
        self._queues = queues
        self._closed = []

    def run(self) -> None:
        # Receive messages from all processes over their connections
        while len(self._closed) != len(self._queues):
            for q in self._queues:
                if not q.empty():
                    self._handle_message(q)

    def _handle_message(self, queue: Queue):
        msg = queue.get_nowait()
        if "close" in msg:
            print(f"Closing connection to process {msg['close']}")
            queue.close()
            self._closed.append(msg['close'])
        elif "checkpoint" in msg:  # checkpoint player initiated from a connected process
            self._checkpoint(msg)
        elif "result" in msg:
            self._save_outcome(msg)
        elif "learner" in msg:
            self._provide_learner(msg)
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
        if home_player.ready_to_checkpoint():  # Auto-checkpoint player
            self.league.add_player(home_player.checkpoint())

    def _provide_learner(self, msg):
        self.league.provide_learner(msg["player_id"], msg["learner"])
        self.league._payoff.players[msg["player_id"]].learner = msg["learner"]
        player = self.league._payoff.players[msg["player_id"]]
        print()
