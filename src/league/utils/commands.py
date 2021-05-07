import uuid
from enum import Enum
from typing import Dict, Any, Tuple

from learners.learner import Learner


class CommandTypes(Enum):
    UPDATE = 0,
    GET = 1,
    POST = 2,
    CLOSE = 3,
    ACK = 4,


class Resources(Enum):
    LEARNER = 0,
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


class UpdateLearnerCommand(BaseCommand):

    def __init__(self, origin: int, data: Learner):
        super().__init__(CommandTypes.UPDATE, origin, Resources.LEARNER, data)


class GetLearnerCommand(BaseCommand):

    def __init__(self, origin: int, data: int):
        super().__init__(CommandTypes.GET, origin, Resources.LEARNER, data)


class CheckpointLearnerCommand(BaseCommand):

    def __init__(self, origin: int):
        super().__init__(CommandTypes.POST, origin, Resources.LEARNER, None)


class PayoffUpdateCommand(BaseCommand):

    def __init__(self, origin: int, data: Tuple[Tuple[int, int], str]):  # TODO replace result string with enum
        super().__init__(CommandTypes.UPDATE, origin, Resources.PAYOFF, data)


class CloseLeagueProcessCommand(BaseCommand):

    def __init__(self, origin: int):
        super().__init__(CommandTypes.CLOSE, origin, Resources.PROCESS, None)
