import torch.nn as nn


class Agent(nn.Module):
    def __init__(self, input_shape, args):
        """
        Agent Interface
        :param input_shape: shape of the observation(=input)
        :param args: additional arguments for agent building
        """
        super(Agent, self).__init__()
        self.args = args
        self.input_shape = input_shape
        self.trained_steps = 0

    def init_hidden(self):
        raise NotImplementedError()

    def forward(self, inputs, hidden_state):
        raise NotImplementedError()

    def count_parameters(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def print_parameters(self):
        for name, param in self.state_dict().items():
            print(name, param)