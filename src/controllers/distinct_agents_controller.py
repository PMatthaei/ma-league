from __future__ import annotations

from typing import List

import torch as th

from components.action_selectors import REGISTRY as action_REGISTRY
from components.episode_batch import EpisodeBatch
from controllers import BasicMAC
from modules.agents import REGISTRY as agent_REGISTRY, AgentNetwork


class DistinctMAC(BasicMAC):
    def __init__(self, scheme, groups, args):
        """
        This is a multi-agent controller without shared parameters between agent networks. Each agent infers from his
        own network.
        :param scheme:
        :param groups:
        :param args:
        """
        super().__init__(scheme, groups, args)
        self.n_agents = args.n_agents
        self.args = args
        input_shape = self._get_input_shape(scheme)
        self.agents = self._build_agent(input_shape)
        self.agent_output_type = args.agent_output_type

        self.action_selector = action_REGISTRY[args.action_selector](args)

        self.hidden_states = None

    def _compute_agent_outputs(self, agent_inputs):
        agent_outs = []
        for i, agent in enumerate(self.agents):
            agent_out, self.hidden_states[i] = agent(agent_inputs[:, i, :], self.hidden_states[i])
            agent_outs.append(agent_out)
        agent_outs = th.cat(agent_outs, dim=0)
        return agent_outs

    def init_hidden(self, batch_size):
        self.hidden_states = [
            agent.init_hidden().expand(batch_size, 1, -1)  # bav
            for agent in self.agents
        ]

    @staticmethod
    def _set_trained_steps(agent, steps):
        agent.trained_steps = steps

    def update_trained_steps(self, trained_steps):
        [self._set_trained_steps(agent, trained_steps) for agent in self.agents]

    def parameters(self):
        params = []
        [params + list(agent.parameters()) for agent in self.agents]
        return params

    def load_state(self, other_mac: DistinctMAC, agents: List[AgentNetwork] = None):
        [
            agent.load_state_dict(agents[i].state_dict() if agents is not None else other_mac.agents[i].state_dict())
            for i, agent in enumerate(self.agents)
        ]

    def cuda(self):
        [agent.cuda() for agent in self.agents]

    def save_models(self, path, name):
        [
            th.save(agent.state_dict(), "{}/{}agent_{}.th".format(path, name, i))
            for i, agent in enumerate(self.agents)
        ]

    def load_models(self, path, name):
        [
            agent.load_state_dict(th.load("{}/{}agent_{}.th".format(path, name, i), lambda storage, loc: storage))
            for i, agent in enumerate(self.agents)
        ]

    def _build_agent(self, input_shape):
        return [agent_REGISTRY[self.args.agent](input_shape, self.args) for _ in range(self.n_agents)]

    def _build_inputs(self, batch: EpisodeBatch, t):
        inputs = [batch["obs"][:, t]]
        if self.args.obs_last_action:
            if t == 0:
                inputs.append(th.zeros_like(batch["actions_onehot"][:, t]))
            else:
                inputs.append(batch["actions_onehot"][:, t - 1])

        if self.args.obs_agent_id:
            raise NotImplementedError("Please deactivate agent id observation for distinct agents networks.")
            # Leave out one-hot encoded agent id

        inputs = th.cat(inputs, dim=2)
        return inputs
