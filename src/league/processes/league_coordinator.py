from torch.multiprocessing import Process, Queue
from typing import List, Tuple

from league.components.payoff import Payoff
from league.roles.players import Player
from league.utils.commands import ProvideLearnerCommand, Ack, CloseLeagueProcessCommand, PayoffUpdateCommand, \
    GetLearnerCommand, CheckpointLearnerCommand


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
        while len(self._closed) != len(set(self._in_queues)):
            for q in self._in_queues:
                if not q.empty():
                    self._handle_commands(q)
        self.logger.info("League Coordinator shut down.")

    def _handle_commands(self, queue: Queue):
        cmd = queue.get_nowait()
        if isinstance(cmd, ProvideLearnerCommand):
            self._update_learner(cmd)
        elif isinstance(cmd, CloseLeagueProcessCommand):
            self.logger.info(f"Closing connection to process {cmd.origin}")
            queue.close()
            self._closed.append(cmd.origin)
        elif isinstance(cmd, CheckpointLearnerCommand):
            self._checkpoint(cmd)
        elif isinstance(cmd, PayoffUpdateCommand):
            self._save_outcome(cmd)
        elif isinstance(cmd, GetLearnerCommand):
            self._provide_learner(cmd)
        else:
            raise Exception("Unknown message.")

    def _checkpoint(self, cmd: CheckpointLearnerCommand):
        """
        Save a checkpoint of the agent with the ID provided in the message.
        :param msg:
        :return:
        """
        self.logger.info(f"Create historical player of player {cmd.origin}")
        historical_player = self._players[cmd.origin].checkpoint()
        self._players.append(historical_player)
        # TODO maybe this does not have to be a message.
        # TODO: is the learner being checkpointed also up-to-date?

    def _save_outcome(self, cmd: PayoffUpdateCommand):
        """
        Update the payoff matrix.
        Check if the learning (=home) player should be checkpointed after each update
        :param cmd:
        :return:
        """
        (home, away), outcome = cmd.data
        home_player, _ = self._payoff.update(home, away, outcome)
        if home_player.ready_to_checkpoint():  # Auto-checkpoint player
            self._players.append(self._players[home_player].checkpoint())

    def _update_learner(self, cmd: ProvideLearnerCommand):
        """
        Receive a learner during the league sub process setup and provide it to its corresponding player.
        These learners can be requested by other sub processes.
        :param cmd:
        :return:
        """
        player_id = cmd.origin
        # --- ! Do not change this assignment
        player = self._players[player_id]
        player.learner = cmd.data
        self._players[player_id] = player
        # --- ! Do not change this assignment
        self.logger.info(f"Updated learner of player {player_id}")
        self._out_queues[player_id].put(Ack(data=cmd.id_))

    def _provide_learner(self, cmd: GetLearnerCommand):
        """
        Provide a league sub process with a requested learner, identified by its owning player.
        :param cmd:
        :return:
        """
        # TODO make sure the most recent version of the sub prcoess is sent
        self._out_queues[cmd.origin].put(self._players[cmd.data].learner)
