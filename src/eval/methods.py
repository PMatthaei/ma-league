import numpy as np


def avg_proportional_loss(jpc_matrix: np.array) -> float:
    diag_mask = np.eye(*jpc_matrix.shape, dtype=bool)
    d = np.mean(jpc_matrix[diag_mask])
    off_mask = ~diag_mask
    o = np.mean(jpc_matrix[off_mask])
    return (d - o) / d
