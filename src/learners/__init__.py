from .q_learner import QLearner
from .sfs_learner import SFSLearner
from .coma_learner import COMALearner

REGISTRY = {}

REGISTRY["q"] = QLearner
REGISTRY["sfs"] = SFSLearner
REGISTRY["coma"] = COMALearner
