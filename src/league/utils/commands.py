import uuid
from enum import Enum
from typing import  Any, Tuple, OrderedDict

from league.components import PayoffEntry


class CommandTypes(Enum):
    UPDATE = 0,
    GET = 1,
    POST = 2,
    CLOSE = 3,
    ACK = 4,


class Resources(Enum):
    AGENT = 0,
    PAYOFF = 1,
    PROCESS = 2,


class BaseCommand:

    def __init__(self, command_type: CommandTypes, origin: int, resource: Resources, data: Any):
        self.id_ = uuid.uuid4()
        self.type = command_type
        self.origin = origin
        self.resource = resource
        self.data = data


class Ack(BaseCommand):
    def __init__(self, data: uuid.UUID):
        super().__init__(CommandTypes.ACK, None, None, data)


class AgentParamsUpdateCommand(BaseCommand):

    def __init__(self, origin: int, data: Tuple[int, OrderedDict]):
        super().__init__(CommandTypes.UPDATE, origin, Resources.AGENT, data)


class AgentPoolGetCommand(BaseCommand):

    def __init__(self, origin: int):
        super().__init__(CommandTypes.GET, origin, Resources.AGENT, None)


class AgentParamsGetCommand(BaseCommand):

    def __init__(self, origin: int, data: int):
        super().__init__(CommandTypes.GET, origin, Resources.AGENT, data)


class CheckpointCommand(BaseCommand):

    def __init__(self, origin: int):
        super().__init__(CommandTypes.POST, origin, Resources.AGENT, None)


class PayoffUpdateCommand(BaseCommand):

    def __init__(self, origin: int, data: Tuple[Tuple[int, int], PayoffEntry]):
        super().__init__(CommandTypes.UPDATE, origin, Resources.PAYOFF, data)


class CloseCommunicationCommand(BaseCommand):

    def __init__(self, origin: int):
        super().__init__(CommandTypes.CLOSE, origin, Resources.PROCESS, None)
