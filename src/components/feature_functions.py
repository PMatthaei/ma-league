from torch import Tensor
import torch as th

from components.episode_batch import EpisodeBatch


class FeatureFunction:

    def __init__(self, d):
        self.d = d

    def __call__(self, obs: Tensor, actions: Tensor, obs_next: Tensor) -> Tensor:
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

    def __call__(self, obs: Tensor, actions: Tensor, obs_next: Tensor) -> Tensor:
        return th.rand((self.d,))  # TODO


REGISTRY = {
    "team_task": TeamTaskSuccessorFeatures(d=8)
}
