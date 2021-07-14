from torch import Tensor
import torch as th


class FeatureFunction:

    def __init__(self):
        pass

    def __call__(self, state, actions, next_state) -> Tensor:
        raise NotImplementedError()


class TeamTaskSuccessorFeatures(FeatureFunction):
    """
    Extract feature to represent a team task.
    """

    def __call__(self, state, actions, next_state) -> Tensor:
        return th.rand((8,)) # TODO


REGISTRY = {
    "team_task": TeamTaskSuccessorFeatures
}
