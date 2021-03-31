import random
import time
from multiprocessing import Process
from multiprocessing.connection import Connection

from league.roles.players import Player


def run(idx, payoff: dict, conn: Connection):
    terminated = False
    while not terminated:
        # Generate new opponent to train against
        opponent = random.randint(0, 3)

        start_time = time.time()
        run_time = 0

        # Run training with current opponent
        while run_time < 2:
            run_time = time.time() - start_time

            # Fake episode play with sleep
            time.sleep(.1)

            # Random result
            result = random.choice(["win", "draw", "loose"])

            # Save result in payoff matrix
            _update_result(idx, opponent, payoff, result)
            # Save games played in payoff matrix
            _update_episodes_played(idx, opponent, payoff)

            conn.send((idx, opponent, result))
            terminated = True

    print("Finished", idx)
    conn.send("close")
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
