from .q_learner import QLearner
from .sfs_learner import SFSLearner
from .coma_learner import COMALearner

REGISTRY = {
    "q": QLearner,
    "sfs": SFSLearner,
    "coma": COMALearner
}
