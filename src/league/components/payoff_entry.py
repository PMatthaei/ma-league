from enum import IntEnum


class PayoffEntry(IntEnum):
    GAMES = 0,  # How many episodes
    WIN = 1,  # How many episodes have been won
    LOSS = 2,  # How many episodes have been lost
    DRAW = 3,  # How many episodes ended in draw
    MATCHES = 4,  # How often has the corresponding instance been matched with all other instances
