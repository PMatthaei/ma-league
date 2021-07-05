from typing import List

import torch as th
from torch import nn
import torch.nn.functional as F

from modules.agents import Agent


class EnsembleAgent(Agent):
    def __init__(self, input_shape, args, agents: List[Agent] = None):
        """
        Main Idea: https://discuss.pytorch.org/t/combining-trained-models-in-pytorch/28383/14
        :param input_shape:
        :param args:
        :param agents:
        """
        super(EnsembleAgent, self).__init__(input_shape, args)
        self.args = args

        self.agents = agents
        self.classifier = nn.Linear(len(self.agents) * args.n_actions, args.n_actions)

    def forward(self, inputs, hidden_state):
        qs = [agent(inputs, hidden_state) for agent in self.agents]
        qs = th.cat(qs, dim=1)
        qs = self.classifier(F.relu(qs))
        return qs,
