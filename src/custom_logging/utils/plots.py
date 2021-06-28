import io
import math
from collections import Sized
from typing import List

import matplotlib.pyplot as plt

from matplotlib.cm import get_cmap
from matplotlib.figure import Figure


def plot_greedy_actions(greedy_actions: dict, n_actions: int, n_agents: int, max_labels=5) -> List[Figure]:
    """
    Receive a dict of an agents greedy actions. Keys of the dict are timesteps at which the greedy actions were
    collected, values is the array of greedy taken action indices since the last plot.
    Count the occurrence of each greedy action in the time interval taken by the agent and plot as bar diagram to
    visualize action distribution under the past epsilon greedy policy.
    :param max_labels:
    :param n_agents:
    :param n_actions:
    :param greedy_actions:
    :return:
    """
    colors = get_cmap(lut=len(greedy_actions))  # Each timestep with its greedy actions receives a unique color
    xs = range(0, n_actions)
    entries_n = math.ceil(len(greedy_actions.items()) / max_labels)
    ys = []

    figures = []

    for a in range(n_agents):  # Iterate over all agent data
        fig = plt.figure(figsize=(10, 10))
        ax = fig.add_subplot(111, projection='3d')

        for i, (t, all_greedy_actions_t) in enumerate(greedy_actions.items()):  # Iterate data of all timesteps
            if not isinstance(all_greedy_actions_t, list) or len(all_greedy_actions_t) < n_agents:
                continue

            greedy_actions_a = all_greedy_actions_t[a]
            n_greedy_actions = len(greedy_actions_a)
            # Normalized counts
            counts = [greedy_actions_a.count(a) / n_greedy_actions if n_greedy_actions > 0 else 0 for a in xs]

            if i % entries_n == 0:  # Every entries_n data set, print labels on top of bars with more than 0 count
                ys.append(t)  # Show only a few y ticks

                for index, count in enumerate(counts):
                    if count > 0:  # Filter here since index would be wrong if filtered in enumerate
                        ax.text(xs[index], t, count + 0.05, s=str(round(count, 2)), fontdict=dict(fontsize=8),
                                ha='center')
                ax.bar(xs, counts, zs=t, zdir='y', color=colors(i), ec=colors(i), alpha=0.8, width=1, align='center')
            else:
                ax.bar(xs, counts, zs=t, zdir='y', color=colors(i), ec=colors(i), alpha=0.8, width=1, align='center')
            ax.set_xlim([-1, n_actions + 1])
            ax.set_zlim([0., 1.])
            ax.set_xticks(range(-1, n_actions + 1))
            ax.set_xlabel('Action')
            ax.set_ylabel('Timestep')
            ax.set_yticks(ys)
            ax.set_yticklabels(ax.get_yticks(), verticalalignment='baseline', horizontalalignment='left')
            ax.set_zlabel('Relative Pick-Rate (since last recorded timestep')
            figures.append(fig)
    return figures
