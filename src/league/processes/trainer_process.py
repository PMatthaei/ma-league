import random
import time
from logging import warning
from multiprocessing.dummy import Process
from multiprocessing.connection import Connection

from league.roles.players import Player


class TrainerProcess(Process):
    def __init__(self, idx: int, player: Player, conn: Connection):
        super().__init__()
        self.idx = idx
        self.player = player
        self.conn = conn

    def run(self):
        j = 0
        while j < 5:
            # Generate new opponent to train against
            opponent, flag = self.player.get_match()

            if opponent is None:
                warning("Opponent was none")
                continue

            player_str = f"{type(self.player).__name__} {self.player.player_id}"
            opponent_str = f"{type(opponent).__name__} {opponent.player_id} "
            print(f"{player_str} playing against opponent {opponent_str} in Process {self.idx}")

            i = 0
            # Run training with current opponent 100 times
            while i < 100:
                # Fake episode play with sleep and fake result
                result = random.choice(["win", "draw", "loss"])
                self.conn.send({"result": (self.player.player_id, opponent.player_id, result)})
                i += 1

            j += 1

        self.conn.send({"close": self.idx})
        self.conn.close()
