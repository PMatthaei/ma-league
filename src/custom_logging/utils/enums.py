from enum import Enum


class LogDestination(Enum):
    CONSOLE = 0,
    TB = 1,
    SACRED = 2

    @property
    def log(self):
        return self.value[0]

    @classmethod
    def list(cls):
        return list(map(lambda c: c.value, cls))


class Originator(str, Enum):
    HOME = "home",
    AWAY = "away"

    @classmethod
    def list(cls):
        return list(map(lambda c: c.value, cls))
