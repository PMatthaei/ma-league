from .dqn_agent import DQNAgent
from .dqrn_agent import DRQNAgent

REGISTRY = {
    "rnn": DRQNAgent,
    "dqn": DQNAgent
}
