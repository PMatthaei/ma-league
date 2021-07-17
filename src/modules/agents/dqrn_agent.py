import torch.nn as nn
import torch.nn.functional as F

from modules.agents.agent_network import AgentNetwork


class DRQNAgentNetwork(AgentNetwork):
    def __init__(self, input_shape, args):
        """
        Deep Q-Network with Recurrent Units
        Architecture:
        1. Fully Connected Layer (MLP): Feature Extraction of observation
        2. Gated Recurrent Unit: Process and remember/forget sequence data (episode).
        3. Fully Connected Layer (MLP): Classification into actions
        :param input_shape: shape of the observation(=input)
        :param args:
        """
        super(DRQNAgentNetwork, self).__init__(input_shape, args)
        self.args = args

        self.fc1 = nn.Linear(input_shape, args.rnn_hidden_dim)  # Linear = y = x * A^T + b
        self.gru = nn.GRUCell(args.rnn_hidden_dim, args.rnn_hidden_dim)
        self.fc2 = nn.Linear(args.rnn_hidden_dim, args.n_actions)

    def init_hidden(self):
        # make hidden states on same device as model
        return self.fc1.weight.new(1, self.args.rnn_hidden_dim).zero_()

    def forward(self, inputs, hidden_state):
        x = self.fc1(inputs)  # Input with shape (batch_size, obs) in learning and (n_agents, obs) in inference
        x = F.relu(x)  # ReLu activation function
        h_in = hidden_state.reshape(-1, self.args.rnn_hidden_dim)  # (1, n_agents, hidden) -> (n_agents, hidden)
        new_hidden_state = self.gru(x, h_in)  # Feed hidden at t-1 and x at t into GRU -> receive new hidden
        q_values = self.fc2(new_hidden_state)  # (n_agents, hidden) -> (n_agents, n_actions)
        return q_values, new_hidden_state
