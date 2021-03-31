import random
import time
from multiprocessing import Process
from multiprocessing.connection import Connection

from league.roles.players import Player


def run(idx: int, player: Player, conn: Connection):
    j = 0
    while j < 5:
        # Generate new opponent to train against
        opponent, flag = player.get_match()
        print(f"Playing against opponent {opponent.player_id} in {idx}")

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


def _update_result(idx, opponent, payoff, result):
    if (idx, opponent, result) in payoff:
        payoff[idx, opponent, result] += 1
    else:
        payoff[idx, opponent, result] = 1


def _update_episodes_played(idx, opponent, payoff):
    if (idx, opponent) in payoff:
        payoff[idx, opponent] += 1
    else:
        payoff[idx, opponent] = 1
