import numpy as np


def prioritized_fictitious_self_play(win_rates, weighting="linear"):
    """

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
    probabilities = fn(np.asarray(win_rates))
    norm = probabilities.sum()
    if norm < 1e-10:
        return np.ones_like(win_rates) / len(win_rates)
    return probabilities / norm
