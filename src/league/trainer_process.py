import random
import time
from logging import warning
from multiprocessing import Process
from multiprocessing.connection import Connection

from league.roles.players import Player


def run(idx: int, player: Player, conn: Connection):
    j = 0
    while j < 5:
        # Generate new opponent to train against
        opponent, flag = player.get_match()
        if opponent is None:
            warning("Opponent was none")
            continue
        player_str = f"{type(player).__name__} {player.player_id}"
        opponent_str = f"{type(opponent).__name__} {opponent.player_id} "
        print(f"{player_str} playing against opponent {opponent_str} in Process {idx}")

        i = 0
        # Run training with current opponent 100 times
        while i < 100:
            # Fake episode play with sleep and fake result
            result = random.choice(["win", "draw", "loos"])
            conn.send({"result": (player.player_id, opponent.player_id, result)})
            i += 1

        j += 1

    conn.send({"close": idx})
    conn.close()
