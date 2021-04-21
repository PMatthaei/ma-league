import torch as th


def avg_proportional_loss(jpc_matrix: th.tensor) -> float:
    diagonal_values = th.diagonal(jpc_matrix)
    d = th.mean(diagonal_values)
    off_mask = ~th.eye(*jpc_matrix.size(), dtype=th.bool)
    off_values = th.masked_select(jpc_matrix, off_mask)
    o = th.mean(off_values)
    return (d - o) / d
