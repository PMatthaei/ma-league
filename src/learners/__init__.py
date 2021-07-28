from .coma_learner import COMALearner
from .q_learner import QLearner
from .sfs_learner import SFSLearner

REGISTRY = {}

REGISTRY["q"] = QLearner
REGISTRY["sfs"] = SFSLearner
REGISTRY["coma"] = COMALearner
