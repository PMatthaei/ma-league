import uuid
from enum import Enum
from typing import Dict, Any, Tuple

from league.components.payoff import MatchResult
from league.roles.players import Player
from learners.learner import Learner


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


class ProvideAgentCommand(BaseCommand):

    def __init__(self, origin: int, data: Learner):
        super().__init__(CommandTypes.UPDATE, origin, Resources.AGENT, data)


class RetrieveAgentCommand(BaseCommand):

    def __init__(self, origin: int, data: int):
        super().__init__(CommandTypes.GET, origin, Resources.AGENT, data)


class CheckpointCommand(BaseCommand):

    def __init__(self, origin: int):
        super().__init__(CommandTypes.POST, origin, Resources.AGENT, None)


class PayoffUpdateCommand(BaseCommand):

    def __init__(self, origin: int, data: Tuple[Tuple[Player, Player], MatchResult]):
        super().__init__(CommandTypes.UPDATE, origin, Resources.PAYOFF, data)


class CloseLeagueProcessCommand(BaseCommand):

    def __init__(self, origin: int):
        super().__init__(CommandTypes.CLOSE, origin, Resources.PROCESS, None)
