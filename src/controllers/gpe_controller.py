from typing import List

from components.episode_batch import EpisodeBatch
from controllers.multi_agent_controller import MultiAgentController
from modules.networks.policy_successor_features import PolicySuccessorFeatures


class GPEController(MultiAgentController):
    def __init__(self, scheme, groups, args):
        super().__init__(scheme, groups, args)
        self.n_features = 10
        self.n_policies = 10
        self.sfs: List[PolicySuccessorFeatures] = [PolicySuccessorFeatures(in_shape=1, out_shape=self.n_policies) for _ in range(self.n_features)]

    def _build_agents(self, input_shape):
        raise NotImplementedError()

    def _get_input_shape(self, scheme):
        raise NotImplementedError()

    def select_actions(self, ep_batch: EpisodeBatch, t_ep: int, t_env: int, bs=slice(None), test_mode=False):
        raise NotImplementedError()

    def forward(self, ep_batch: EpisodeBatch, t: int, test_mode=False):
        raise NotImplementedError()

    def init_hidden(self, batch_size: int):
        raise NotImplementedError()

    def parameters(self):
        raise NotImplementedError()

    def load_state(self, other_mac: MultiAgentController):
        raise NotImplementedError()

    def cuda(self):
        raise NotImplementedError()

    def save_models(self, path: str, name: str):
        raise NotImplementedError()

    def load_models(self, path: str, name: str):
        raise NotImplementedError()

    def update_trained_steps(self, trained_steps: int):
        raise NotImplementedError()
