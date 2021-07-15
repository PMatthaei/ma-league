from torch import Tensor
import torch as th


class FeatureFunction:

    def __init__(self, d):
        self.d = d

    def __call__(self, state, actions, next_state) -> Tensor:
        raise NotImplementedError()

    @property
    def n_features(self):
        return self.d


class TeamTaskSuccessorFeatures(FeatureFunction):
    """
    Extract feature to represent a team task.
    """

    def __init__(self, d):
        super().__init__(d)

    def __call__(self, state, actions, next_state) -> Tensor:
        return th.rand((8,))  # TODO


REGISTRY = {
    "team_task": TeamTaskSuccessorFeatures
}
