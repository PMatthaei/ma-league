import io

import matplotlib.pyplot as plt

import numpy as np
import tensorflow as tf
from matplotlib.cm import get_cmap
from matplotlib.figure import Figure


class TensorBoardPlot:
    def plot_to_image(self, figure):
        """
        Converts the matplotlib plot specified by 'figure' to a PNG image and
        returns it. The supplied figure is closed and inaccessible after this call.
        :param figure:
        :return:
        """
        buf = io.BytesIO()
        plt.savefig(buf, format='png')
        buf.seek(0)
        plt.close(figure)
        image = tf.image.decode_png(buf.getvalue(), channels=4)
        image = tf.expand_dims(image, 0)
        return image


class GreedyActionPlot(TensorBoardPlot):
    def __init__(self, agent, n_actions):
        """
        Plots an agents greedy actions as histogram.
        :param agent:
        :param n_actions:
        """
        self.agent = agent
        self.n_actions = n_actions

    def plot(self, greedy_actions: dict) -> Figure:
        """
        Receive a dict of an agents greedy actions. Keys of the dict are timesteps at which the greedy actions were
        collected, values is the array of greedy taken action indices since the last plot.
        Count the occurrence of each greedy action in the time interval taken by the agent and plot as bar diagram to
        visualize action distribution under the past epsilon greedy policy.
        :param greedy_actions:
        :return:
        """
        cmap = get_cmap(lut=len(greedy_actions)) # Each timestep with its greedy actions receives a unique color
        fig = plt.figure(figsize=(10, 10))
        ax = fig.add_subplot(111, projection='3d')

        for i, (t, gas) in enumerate(greedy_actions.items()):
            counts = [gas.count(a) for a in range(self.n_actions)]
            ax.bar(range(0,self.n_actions), counts, zs=t, zdir='y', color=cmap(i), ec=cmap(i), alpha=0.8, width=1, align='center')

        ax.set_xlim([-1, self.n_actions + 1])
        ax.set_xticks(range(-1, self.n_actions + 1))
        ax.set_xlabel('Action')
        ax.set_ylabel('Timestep')
        ax.set_zlabel('Count')

        return fig
