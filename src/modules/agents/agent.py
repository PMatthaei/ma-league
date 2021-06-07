import torch.nn as nn
import torch as th


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


class DistinctMultiAgent(Agent):
    def __init__(self, agent_type, n_agents):
        """
        Distinct-Multi-Agent Interface
        """
        super(Agent, self).__init__()
        self.agents: nn.ModuleList = nn.ModuleList([
            agent_type(self.input_shape, self.args)
            for _ in range(n_agents)
        ])

    def init_hidden(self):
        hidden_states = [agent.init_hidden() for agent in self.agents]
        return th.stack(hidden_states)

    def forward(self, inputs, hidden_states):
        outputs = [agent(inputs[:, i, :], hidden_states[i]) for i, agent in enumerate(self.agents)]
        return th.stack(outputs)
