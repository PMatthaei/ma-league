from torch import Tensor
from torch.multiprocessing import Process, Queue, Barrier
from typing import List, Tuple, Dict
from collections import OrderedDict

from custom_logging.platforms import CustomConsoleLogger
from league.components import PayoffEntry
from league.utils.commands import CloseLeagueProcessCommand, PayoffUpdateCommand, CheckpointCommand, \
    AgentParamsUpdateCommand, \
    AgentParamsGetCommand, AgentPoolGetCommand


def clone_state_dict(dict: OrderedDict):
    clone = OrderedDict([(entry, dict[entry].clone()) for entry in dict])
    return clone

class CommandHandler(Process):
    def __init__(self, allocation: Dict[int, int], n_senders: int, communication: Tuple[List[Queue], List[Queue]],
                 payoff: Tensor,
                 sync_barrier: Barrier):
        """
        Handles messages sent from league sub processes to the main league process.
        :param league:
        :param connections:
        """
        super().__init__()
        self._agent_pool = dict()
        self._in_queues, self._out_queues = communication
        self._allocation = allocation
        self._payoff = payoff
        self._closed = []
        self._sync_barrier = sync_barrier

        self.n_updates = 0
        self.n_senders = n_senders
        self.last_waiting = 0
        self.shutdown = False
        self.running = True

        self.logger = None

    def run(self) -> None:
        self.logger = CustomConsoleLogger("league-coordinator")
        self.logger.info("League Coordinator started.")

        # Receive messages from all processes over their connections
        while self.running:
            self._on_sync()

            [self._handle_commands(in_q) for in_q in self._in_queues if not in_q.empty()]
            if self.shutdown:
                self.running = False

    def _on_sync(self):
        # All processes have arrived at barrier and passed
        if self._sync_barrier.n_waiting == 0 and self.last_waiting != 0:
            self.logger.info("LeagueCoordinator synced with training instances")
        self.last_waiting = self._sync_barrier.n_waiting

    def _handle_commands(self, in_queue: Queue):
        cmd = in_queue.get_nowait()
        if isinstance(cmd, CloseLeagueProcessCommand):
            self._close(cmd)
        elif isinstance(cmd, CheckpointCommand):
            self._checkpoint(cmd)
        elif isinstance(cmd, PayoffUpdateCommand):
            self._update_payoff(cmd)
        elif isinstance(cmd, AgentParamsUpdateCommand):
            self._update_agent_params(cmd)
        elif isinstance(cmd, AgentParamsGetCommand):
            self._get_agent(cmd)
        elif isinstance(cmd, AgentPoolGetCommand):
            self._get_agent_pool(cmd)
        else:
            raise Exception(f"Unknown command {cmd} received in LeagueCoordinator. Please implement this command.")

    def _close(self, cmd):
        self.logger.info(f"Closing connection to process {cmd.origin}")
        self._closed.append(cmd.origin)
        if len(self._closed) == len(self._in_queues):  # Shutdown
            self.shutdown = True
            self._in_queues[cmd.origin].close()
            self.logger.info("League Coordinator shut down.")
        self._out_queues[cmd.origin].put(None) # ACK
        self._out_queues[cmd.origin].close()

    def _checkpoint(self, cmd: CheckpointCommand):
        """
        Save a checkpoint of the agent with the ID provided in the message.
        :param msg:
        :return:
        """
        self.logger.info(f"Create historical player of player {cmd.origin}")
        raise NotImplementedError()

    def _update_payoff(self, cmd: PayoffUpdateCommand):
        """
        Update the payoff matrix.
        Check if the learning (=home) player should be checkpointed after each update
        :param cmd:
        :return:
        """
        (home_team, away_team), outcome = cmd.data
        home_instance, away_instance = self._allocation[home_team], self._allocation[away_team]
        self._payoff[home_instance, away_instance, PayoffEntry.GAMES] += 1
        self._payoff[home_instance, away_instance, outcome.value] += 1
        self.n_updates += 1
        self._out_queues[cmd.origin].put(None) # ACK

    def _get_agent(self, cmd: AgentParamsGetCommand):
        agent_params = clone_state_dict(self._agent_pool[cmd.data])
        self._out_queues[cmd.origin].put(agent_params)
        del agent_params

    def _get_agent_pool(self, cmd: AgentPoolGetCommand):
        pool_clone = {tid: clone_state_dict(agent_params) for tid, agent_params in self._agent_pool.items()}
        self._out_queues[cmd.origin].put(pool_clone)
        del pool_clone

    def _update_agent_params(self, cmd: AgentParamsUpdateCommand):
        tid, params = cmd.data
        self._agent_pool[tid] = params
        self.logger.info(f"Received parameter update for agent of team {tid}")
        self._out_queues[cmd.origin].put(None) # ACK

