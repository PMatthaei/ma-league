from random import shuffle
from typing import Tuple

import numpy as np
import torch as th


def avg_proportional_loss(jpc_matrix: th.tensor) -> float:
    diagonal_values = th.diagonal(jpc_matrix)
    d = th.mean(diagonal_values)
    off_mask = ~th.eye(*jpc_matrix.size(), dtype=th.bool)
    off_values = th.masked_select(jpc_matrix, off_mask)
    o = th.mean(off_values)
    if d < 0:
        raise NotImplementedError()
    if (d - o) < 0:
        print("Off-Diagonal-Mean greater than Diagonal-Mean")
    return (d - o) / d


def train_test_split(data: np.ndarray, test_size=.80) -> Tuple[np.ndarray, np.ndarray]:
    shuffle(data)
    train = data[:int((len(data) + 1) * test_size)]  # Remaining test_size% to training set
    test = data[int((len(data) + 1) * test_size):]  # Splits 1-test_size% data to test set
    return train, test
