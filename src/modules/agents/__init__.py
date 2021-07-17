from .agent_network import AgentNetwork
from .dqn_agent import DQNAgentNetwork
from .dqrn_agent import DRQNAgentNetwork

REGISTRY = {
    "rnn": DRQNAgentNetwork,
    "dqn": DQNAgentNetwork
}
