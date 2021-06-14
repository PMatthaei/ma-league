import io
import math

import matplotlib.pyplot as plt

import tensorflow as tf
from matplotlib.cm import get_cmap
from matplotlib.figure import Figure


def plot_to_image(figure: Figure):
    """
    Converts the matplotlib plot specified by 'figure' to a PNG image and
    returns it. The supplied figure is closed and inaccessible after this call.
    :return:
    """
    buf = io.BytesIO()
    plt.savefig(buf, format='png')
    buf.seek(0)
    plt.close(figure)
    image = tf.image.decode_png(buf.getvalue(), channels=4)
    image = tf.expand_dims(image, 0)
    return image


def plot_greedy_actions(greedy_actions: dict, n_actions: int, max_labels=5) -> tf.Tensor:
    """
    Receive a dict of an agents greedy actions. Keys of the dict are timesteps at which the greedy actions were
    collected, values is the array of greedy taken action indices since the last plot.
    Count the occurrence of each greedy action in the time interval taken by the agent and plot as bar diagram to
    visualize action distribution under the past epsilon greedy policy.
    :param greedy_actions:
    :return:
    """
    fig = plt.figure(figsize=(10, 10))
    colors = get_cmap(lut=len(greedy_actions))  # Each timestep with its greedy actions receives a unique color
    ax = fig.add_subplot(111, projection='3d')
    xs = range(0, n_actions)
    entries_n = math.ceil(len(greedy_actions.items()) / max_labels)
    ys = []
    for i, (t, gas) in enumerate(greedy_actions.items()):
        n_greedy_actions_taken = len(gas)
        # Normalize counts
        counts = [gas.count(a) / n_greedy_actions_taken if n_greedy_actions_taken > 0 else 0 for a in xs]
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
    fig.canvas.draw()  # Draw in blocking manner to prevent showing the figure before every bar is plotted
    return plot_to_image(fig)
