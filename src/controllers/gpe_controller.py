import random
from typing import List

from torch import Tensor

from components.episode_batch import EpisodeBatch
from controllers.multi_agent_controller import MultiAgentController
from modules.agents import Agent
from components.feature_functions import REGISTRY as feature_func_REGISTRY
from modules.networks.policy_successor_features import PolicySuccessorFeatures


class GPEController(MultiAgentController):
    def __init__(self, scheme, groups, args):
        super().__init__(scheme, groups, args)
        self.phi = feature_func_REGISTRY["team_task"]()
        self.policies: List[Agent] = []
        self.task_ws: List[Tensor] = []
        self.sfs: List[PolicySuccessorFeatures] = [
            PolicySuccessorFeatures(in_shape=self._get_input_shape(scheme), out_shape=len(self.policies))
            for _ in range(self.phi.n_features)
        ] # Barreto et al. propose one MLP per Feature with d=2 and two hidden layers 64 and 128

    def _build_agents(self, input_shape):
        raise NotImplementedError()

    def _get_input_shape(self, scheme):
        raise NotImplementedError()

    def select_actions(self, ep_batch: EpisodeBatch, t_ep: int, t_env: int, bs=slice(None), test_mode=False):
        avail_actions = ep_batch["avail_actions"][:, t_ep]
        agent_outs = self.forward(ep_batch, t_ep, test_mode=test_mode)
        chosen_actions, is_greedy = self.action_selector.select(agent_outs[bs], avail_actions[bs], t_env, test_mode)
        return chosen_actions, is_greedy

    def forward(self, ep_batch: EpisodeBatch, t: int, test_mode=False):
        self.agent = random.choice(self.policies)  # Pick random policy for each episode batch (!) - Change to paper
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
