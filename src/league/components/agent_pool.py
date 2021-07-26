from typing import Tuple
from torch.multiprocessing import Queue

from league.utils.commands import AgentPoolGetCommand, CloseCommunicationCommand


class AgentPool:

    def __init__(self, comm_id: int, communication: Tuple[Queue, Queue]):
        """
        Manages the current set of trained multi-agents which themselves are linked to their underlying team.
        Only parameters of agents are stored to prevent storing the same network architecture redundantly.
        ! WARN ! This will only work as long as all agent networks share the same architecture.
        :param shared_storage:
        """
        self._in_q, self._out_q = communication
        self._comm_id = comm_id

    @property
    def agents(self):
        cmd = AgentPoolGetCommand(origin=self._comm_id)
        self._in_q.put(cmd)
        pool = self._out_q.get()
        return pool

    def disconnect(self):
        cmd = CloseCommunicationCommand(origin=self._comm_id)
        self._in_q.put(cmd)
