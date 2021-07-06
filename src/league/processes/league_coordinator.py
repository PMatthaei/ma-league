from torch.multiprocessing import Process, Queue, Barrier
from typing import List, Tuple

from league.components.payoff_role_based import RolebasedPayoff
from league.roles.players import Player
from league.utils.commands import CloseLeagueProcessCommand, PayoffUpdateCommand, CheckpointCommand


class LeagueCoordinator(Process):
    def __init__(self, logger, players: List[Player], queues: Tuple[List, List], payoff: RolebasedPayoff, sync_barrier: Barrier):
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
        self.logger = logger
        self.sync_barrier = sync_barrier

        self.updates_n = 0
        self.senders_n = len(players)
        self.last_waiting = 0

    def run(self) -> None:
        # Receive messages from all processes over their connections
        while len(self._closed) != len(set(self._in_queues)):

            self._on_sync()

            for q in self._in_queues:
                if not q.empty():
                    self._handle_commands(q)

        self.logger.info("League Coordinator shut down.")

    def _get_agents(self):
        return [(p.team, p.agent) for p in self._players]

    def _on_sync(self):
        # All processes have arrived at barrier and passed
        if self.sync_barrier.n_waiting == 0 and self.last_waiting != 0:
            self.logger.info(str(self._payoff))
        self.last_waiting = self.sync_barrier.n_waiting

    def _handle_commands(self, queue: Queue):
        cmd = queue.get_nowait()
        if isinstance(cmd, CloseLeagueProcessCommand):
            self.logger.info(f"Closing connection to process {cmd.origin}")
            queue.close()
            self._closed.append(cmd.origin)
        elif isinstance(cmd, CheckpointCommand):
            self._checkpoint(cmd)
        elif isinstance(cmd, PayoffUpdateCommand):
            self._update_payoff(cmd)
        else:
            raise Exception("Unknown message.")

    def _checkpoint(self, cmd: CheckpointCommand):
        """
        Save a checkpoint of the agent with the ID provided in the message.
        :param msg:
        :return:
        """
        self.logger.info(f"Create historical player of player {cmd.origin}")
        historical_player = self._players[cmd.origin].checkpoint()
        self._players.append(historical_player)
        # TODO: is the learner being checkpointed also up-to-date?

    def _update_payoff(self, cmd: PayoffUpdateCommand):
        """
        Update the payoff matrix.
        Check if the learning (=home) player should be checkpointed after each update
        :param cmd:
        :return:
        """
        (home, away), outcome = cmd.data
        self._payoff.update(home, away, outcome)
        self.updates_n += 1
        if home.ready_to_checkpoint():  # Auto-checkpoint player
            self._players.append(self._players[home].checkpoint())
