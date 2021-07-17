import torch.nn as nn
import torch as th
import torch.nn.functional as F
from torch.autograd import Variable

from modules.agents.agentnetwork import AgentNetwork


class DQNAgentNetwork(AgentNetwork):
    def __init__(self, input_shape, args):
        """
        Deep Q-Network
        Architecture:
        1. Fully Connected Layer (MLP): Feature Extraction of observation
        2. Fully Connected Layer (MLP): Classification into actions
        Note: Most DQNs use Conv- and BN-Layers to deal with highly dimensional input such as images.
        This is not necessary since our input features is direct game state information.
        :param input_shape: shape of the observation(=input)
        :param args:
        """
        super(DQNAgentNetwork, self).__init__(input_shape, args)
        self.args = args

        self.fc1 = nn.Linear(input_shape, args.rnn_hidden_dim)
        self.fc2 = nn.Linear(args.rnn_hidden_dim, args.n_actions)

    def init_hidden(self):
        """
        model has no hidden state, but we will pretend otherwise for consistency
        """
        vbl = Variable(th.zeros(self.args.batch_size, 1, 1))
        return vbl.cuda() if self.args.use_cuda else vbl

    def forward(self, inputs, hidden_states):
        x = F.relu(self.fc1(inputs))
        x = self.fc2(x)
        return x, hidden_states
