from enum import Enum


class Originator(str, Enum):
    HOME = "home",  # The learning agent aka the home team
    AWAY = "away"  # The fixed policy / AI (part of the env) playing against the home team

    @classmethod
    def list(cls):
        return list(map(lambda c: c.value, cls))
