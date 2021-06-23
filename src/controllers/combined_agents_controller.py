from __future__ import annotations

from typing import List

from controllers.multi_agent_controller import MultiAgentController
from exceptions.mac_exceptions import HiddenStateNotInitialized
from modules.agents import REGISTRY as agent_REGISTRY, Agent
from components.action_selectors import REGISTRY as action_REGISTRY
import torch as th


class CombinedMAC(MultiAgentController):
    def __init__(self, scheme, groups, args):
        """
        This is a multi-agent controller uses a combination of networks for inference. Each agent can choose to either
        infer with the original/native agent network or a network which was trained within a different team.
        :param scheme:
        :param groups:
        :param args:
        """
        self.n_agents = args.n_agents
        self.n_actions = args.n_actions
        self.args = args
        input_shape = self._get_input_shape(scheme)
        self.agent = self._build_agents(input_shape)  # Single sharing native network for the Multi-Agent
        self.specific_agents = dict()  # Dictionary holding the specific agent network for a given agent

        self.agent_output_type = args.agent_output_type

        self.action_selector = action_REGISTRY[args.action_selector](args)

        self.native_hidden_states = None
        self.specific_hidden_states = None

        self._all_ids = set(range(self.n_agents))

    @property
    def n_native_agents(self):
        return self.n_agents - self.n_specific_agents

    @property
    def n_specific_agents(self):
        return len(self.specific_agents)

    @property
    def native_agents_ids(self):
        return list(self._all_ids.difference(self.specific_agents_ids))

    @property
    def specific_agents_ids(self):
        return list(self.specific_agents.keys())  # Agent IDs that use specific network for inference instead of native

    def select_actions(self, ep_batch, t_ep, t_env, bs=slice(None), test_mode=False):
        avail_actions = ep_batch["avail_actions"][:, t_ep]
        agent_outs = self.forward(ep_batch, t_ep, test_mode=test_mode)
        chosen_actions, is_greedy = self.action_selector.select(agent_outs[bs], avail_actions[bs], t_env, test_mode)
        return chosen_actions, is_greedy

    def replace_agent(self, aid: int, agent: Agent):
        """
        Replaces inference for the given agent with id = aid with the inference of the provided agent by adding into the
        dict.
        :param aid:
        :param agent:
        :return:
        """
        self.specific_agents.update({aid: agent})

    def forward(self, ep_batch, t, test_mode=False):
        native_inputs, specific_inputs = self._build_inputs(ep_batch, t)
        avail_actions = ep_batch["avail_actions"][:, t]

        agent_outs = self._compute_agent_outputs(native_inputs, specific_inputs, ep_batch.batch_size)

        # Softmax the agent outputs if they're policy logits
        if self.agent_output_type == "pi_logits":

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

        return agent_outs.view(ep_batch.batch_size, self.n_agents, -1)

    def _compute_agent_outputs(self, native_inputs, specific_inputs, batch_size):
        native_agent_outs, self.native_hidden_states = self.agent(native_inputs, self.native_hidden_states)
        agent_outs = native_agent_outs.view(batch_size, self.n_agents, -1)
        for aid, specific_agent in self.specific_agents.items():
            agent_outs[:, aid, :], self.specific_hidden_states[aid] = specific_agent(specific_inputs[aid, :].view(batch_size, -1), self.specific_hidden_states[aid])
        return agent_outs.view(batch_size * self.n_agents, -1)

    def update_trained_steps(self, trained_steps):
        self.agent.trained_steps = trained_steps

    def init_hidden(self, batch_size):
        self.native_hidden_states = self.agent.init_hidden().unsqueeze(0).expand(batch_size, self.n_agents, -1)
        self.specific_hidden_states = {
            aid: agent.init_hidden().unsqueeze(0).expand(batch_size, 1, -1)  # bav
            for aid, agent in self.specific_agents.items()
        }

    def parameters(self):
        params = []
        params += self.agent.parameters()
        [params + list(agent.parameters()) for agent in self.specific_agents.values()]
        return params

    def load_state(self, other_mac: CombinedMAC):
        self.agent.load_state_dict(other_mac.agent.state_dict())
        [
            agent.load_state_dict(other_mac.specific_agents[aid].state_dict())
            for aid, agent in self.specific_agents.items()
        ]

    def cuda(self):
        self.agent.cuda()
        [agent.cuda() for agent in self.specific_agents.values()]

    def save_models(self, path, name):
        th.save(self.agent.state_dict(), "{}/{}agent_native.th".format(path, name))
        [
            th.save(agent.state_dict(), "{}/{}agent_specific_{}.th".format(path, name, i))
            for i, agent in self.specific_agents.items()
        ]

    def load_models(self, path, name):
        self.agent.load_state_dict(
            th.load("{}/{}agent_native.th".format(path, name), map_location=lambda storage, loc: storage))
        [
            agent.load_state_dict(
                th.load("{}/{}agent_specific_{}.th".format(path, name, aid), lambda storage, loc: storage))
            for aid, agent in enumerate(self.specific_agents.items())
        ]

    def _build_agents(self, input_shape):
        return agent_REGISTRY[self.args.agent](input_shape, self.args)

    def _build_inputs(self, batch, t):
        bs = batch.batch_size
        inputs = [batch["obs"][:, t]]
        if self.args.obs_last_action:
            if t == 0:
                inputs.append(th.zeros_like(batch["actions_onehot"][:, t]))
            else:
                inputs.append(batch["actions_onehot"][:, t - 1])
        if self.args.obs_agent_id:
            inputs.append(th.eye(self.n_agents, device=batch.device).unsqueeze(0).expand(bs, -1, -1))

        native_inputs = th.cat([x.reshape(bs * self.n_agents, -1) for x in inputs], dim=1)
        specific_inputs = th.cat([x.reshape(self.n_agents, -1) for x in inputs], dim=1)
        return native_inputs, specific_inputs

    def _get_input_shape(self, scheme):
        input_shape = scheme["obs"]["vshape"]
        if self.args.obs_last_action:
            input_shape += scheme["actions_onehot"]["vshape"][0]
        if self.args.obs_agent_id:
            input_shape += self.n_agents

        return input_shape
