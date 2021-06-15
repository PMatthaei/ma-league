from enum import Enum

class Originator(str, Enum):
    HOME = "home",
    AWAY = "away"

    @classmethod
    def list(cls):
        return list(map(lambda c: c.value, cls))
