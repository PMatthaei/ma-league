import numpy as np
import torch as th


def percentage(x):
    return np.sum(np.array(x, dtype=int), axis=0) / len(x)


def extract_greedy_actions(episodal_actions_taken):
    if len(episodal_actions_taken) > 0:
        # Concat over episodes:
        # (episodes x t_episode_max x 2 x n_agents) -> ((episodes x t_episode_max) x 2 x n_agents)
        episodal_actions_taken = th.cat(episodal_actions_taken, dim=0)
        is_greedy = episodal_actions_taken[:, 1, :] == 1
        episodal_actions_taken = episodal_actions_taken[:, 0, :]
        n_agents = episodal_actions_taken.shape[-1:][0]  # Extract agent num (stored in last dim !)
        return [th.masked_select(episodal_actions_taken[:, i], mask=is_greedy[:, i]).tolist() for i in range(n_agents)]
    else:
        return []

