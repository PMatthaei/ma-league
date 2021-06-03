import numpy as np


def self_play(opponent_id):
    """
    SP: Draw same opponent for training
    :return:
    """
    return opponent_id


def fictitious_self_play():
    """
    FSP: Draw from opponent pool uniformly
    :return:
    """
    return np.random.uniform(low=0.0, high=1.0, size=None)


def prioritized_fictitious_self_play(win_rates, weighting="linear"):
    """
    PFSP: Draw from opponent pool  with priority on win rate.
    :param win_rates:
    :param weighting:
    :return:
    """
    weightings = {
        "variance": lambda x: x * (1 - x),
        "linear": lambda x: 1 - x,
        "linear_capped": lambda x: np.minimum(0.5, 1 - x),
        "squared": lambda x: (1 - x) ** 2,
    }
    fn = weightings[weighting]
    # Weight win rates
    probabilities = fn(np.asarray(win_rates))
    norm = probabilities.sum()
    if norm < 1e-10:
        return np.ones_like(win_rates) / len(win_rates)
    return probabilities / norm
