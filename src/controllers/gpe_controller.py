import random
import torch as th
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
        self.agents: List[Agent] = []
        self.task_ws: List[Tensor] = []
        self.sfs: List[PolicySuccessorFeatures] = [
            PolicySuccessorFeatures(in_shape=self._get_input_shape(scheme), out_shape=len(self.agents))
            for _ in range(self.phi.n_features)
        ]  # Barreto et al. propose one MLP per Feature with d=2 and two hidden layers 64 and 128

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
        self.agent = random.choice(self.agents)  # TODO: place where it only random selects every episode
        agent_inputs = self._build_inputs(ep_batch, t)
        agent_outs = self._compute_agent_outputs(agent_inputs)
        return agent_outs.view(ep_batch.batch_size, self.n_agents, -1)

    def _compute_agent_outputs(self, agent_inputs):
        agent_outs, self.hidden_states = self.agent(agent_inputs, self.hidden_states)
        return agent_outs

    def _build_inputs(self, batch: EpisodeBatch, t: int):
        bs = batch.batch_size
        inputs = [batch["obs"][:, t]]
        if self.args.obs_last_action:
            if t == 0:
                inputs.append(th.zeros_like(batch["actions_onehot"][:, t]))
            else:
                inputs.append(batch["actions_onehot"][:, t - 1])
        if self.args.obs_agent_id:
            inputs.append(th.eye(self.n_agents, device=batch.device).unsqueeze(0).expand(bs, -1, -1))

        inputs = th.cat([x.reshape(bs * self.n_agents, -1) for x in inputs], dim=1)
        return inputs
