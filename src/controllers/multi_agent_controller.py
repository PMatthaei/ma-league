from __future__ import annotations

from components.episode_batch import EpisodeBatch
from components.action_selectors import REGISTRY as action_REGISTRY

import torch as th


class MultiAgentController:
    def __init__(self, scheme, groups, args):
        """
        The Multi-Agent Controller defines how an Multi-Agent is built from a registry of existing agents and how its
        actions are selected. It manages the hidden state and offers a saving and loading strategy of models.

        Parameters of the agents are updated via a connected Learner which is defined in the training run.
        :param scheme:
        :param groups:
        :param args:
        """
        self.n_agents = args.n_agents
        self.n_actions = args.n_actions
        self.args = args
        input_shape = self._get_input_shape(scheme)
        self.agent = self._build_agents(input_shape)
        self.agent_output_type = args.agent_output_type

        self.action_selector = action_REGISTRY[args.action_selector](args)

        self.hidden_states = None

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

    def _softmax(self, agent_outs, ep_batch, t, test_mode):
        avail_actions = ep_batch["avail_actions"][:, t]

        if getattr(self.args, "mask_before_softmax", True):
            # Make the logits for unavailable actions very negative to minimise their affect on the softmax
            reshaped_avail_actions = avail_actions.reshape(ep_batch.batch_size * self.n_agents, -1)
            agent_outs[reshaped_avail_actions == 0] = -1e10
        agent_outs = th.nn.functional.softmax(agent_outs, dim=-1)
        if not test_mode:
            # Epsilon floor
            epsilon_action_num = agent_outs.size(-1)
            if getattr(self.args, "mask_before_softmax", True):
                # With probability epsilon, we will pick an available action uniformly
                epsilon_action_num = reshaped_avail_actions.sum(dim=1, keepdim=True).float()

            agent_outs = ((1 - self.action_selector.epsilon) * agent_outs
                          + th.ones_like(agent_outs) * self.action_selector.epsilon / epsilon_action_num)

            if getattr(self.args, "mask_before_softmax", True):
                # Zero out the unavailable actions
                agent_outs[reshaped_avail_actions == 0] = 0.0
        return agent_outs
