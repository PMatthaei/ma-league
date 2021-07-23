from __future__ import annotations

import copy
from typing import Dict, OrderedDict

from controllers.multi_agent_controller import MultiAgentController
from modules.agents import REGISTRY as agent_REGISTRY, AgentNetwork
import torch as th


class EnsembleMAC(MultiAgentController):
    def __init__(self, scheme, groups, args):
        """
        This is a multi-agent controller uses a combination of networks for inference. Each agent can choose to either
        infer with the original/native agent network (self.agent) or an ensemble of networks.

        This MAC can only be used as a fixed policy opponent within league training!

        :param scheme:
        :param groups:
        :param args:
        """
        super().__init__(scheme, groups, args)
        # Dictionary holding the specific agent network for a given agent
        self.ensemble: Dict[int, AgentNetwork] = dict({0: copy.deepcopy(self.agent)})
        self.native_hidden_states = None
        self.ensemble_hidden_states = None
        self._all_ids = set(range(self.n_agents))
        if args.freeze_native:  # Freezes the native/original agent to prevent learning
            for p in self.agent.parameters():
                p.requires_grad = False

    def select_actions(self, ep_batch, t_ep, t_env, bs=slice(None), test_mode=False):
        avail_actions = ep_batch["avail_actions"][:, t_ep]
        agent_outs = self.forward(ep_batch, t_ep, test_mode=test_mode)
        chosen_actions, is_greedy = self.action_selector.select(agent_outs[bs], avail_actions[bs], t_env, test_mode)
        return chosen_actions, is_greedy

    def forward(self, ep_batch, t, test_mode=False):
        native_inputs, specific_inputs = self._build_inputs(ep_batch, t)

        agent_outs = self._compute_agent_outputs(native_inputs, specific_inputs, ep_batch.batch_size)

        # Softmax the agent outputs if they're policy logits
        if self.agent_output_type == "pi_logits":
            agent_outs = self._softmax(agent_outs, ep_batch, t, test_mode)

        return agent_outs.view(ep_batch.batch_size, self.n_agents, -1)

    def _compute_agent_outputs(self, native_inputs, specific_inputs, batch_size):
        # Infer with the original native network - Receives all obs
        native_agent_outs, self.native_hidden_states = self.agent(native_inputs, self.native_hidden_states)
        agent_outs = native_agent_outs.view(batch_size, self.n_agents, -1)

        # Replace inference for specific agents with the corresponding network from the ensemble - Receive only one obs
        for aid, ensemble_agent in self.ensemble.items():
            specific_input = specific_inputs[aid, :].view(batch_size, -1)
            hidden_state = self.ensemble_hidden_states[aid]
            agent_outs[:, aid, :], self.ensemble_hidden_states[aid] = ensemble_agent(specific_input, hidden_state)

        return agent_outs.view(batch_size * self.n_agents, -1)

    def init_hidden(self, batch_size):
        self.native_hidden_states = self.agent.init_hidden().unsqueeze(0).expand(batch_size, self.n_agents, -1)
        self.ensemble_hidden_states = {
            aid: agent.init_hidden().unsqueeze(0).expand(batch_size, 1, -1)  # bav but for a single agent
            for aid, agent in self.ensemble.items()
        }

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

    def load_state(self, other_mac: EnsembleMAC=None, agent: AgentNetwork = None, ensemble: Dict[int, AgentNetwork] = None):
        if other_mac is not None:
            self.agent.load_state_dict(other_mac.agent.state_dict())
            [agent.load_state_dict(other_mac.ensemble[idx].state_dict()) for idx, agent in self.ensemble.items()]
            # Load state from another MAC (f.e. when loading target mac)
        if agent is not None:
            self.agent.load_state_dict(agent.state_dict())
            # Load a native agent
        if ensemble is not None:
            [agent.load_state_dict(ensemble[idx].state_dict()) for idx, agent in self.ensemble.items()]
            # Load a dict of ensembles

    def load_state_dict(self, agent: OrderedDict, ensemble: Dict[int, OrderedDict] = None):
        self.agent.load_state_dict(agent) if agent is not None else None
        if ensemble is not None:
            for aid, state in ensemble.items():
                if aid in self.ensemble:
                    self.ensemble[aid].load_state_dict(state)
                else:
                    agent = copy.deepcopy(self.agent)
                    agent.load_state_dict(state)
                    self.ensemble.update({aid: agent})

    def cuda(self):
        self.agent.cuda()
        [agent.cuda() for agent in self.ensemble.values()]

    def parameters(self):
        params = []
        params += list(self.agent.parameters()) # add native agents params
        [params + list(agent.parameters()) for agent in self.ensemble.values()] # add params of each agent in the ensemble
        return params

    def update_trained_steps(self, trained_steps):
        self.agent.trained_steps = trained_steps

    def save_models(self, path, name):
        raise NotImplementedError()

    def load_models(self, path, name):
        raise NotImplementedError()

    @property
    def n_native_agents(self):
        return self.n_agents - self.n_specific_agents

    @property
    def n_specific_agents(self):
        return len(self.ensemble)

    @property
    def native_agents_ids(self):
        return list(self._all_ids.difference(self.specific_agents_ids))

    @property
    def specific_agents_ids(self):
        return list(self.ensemble.keys())  # Agent IDs that use specific network for inference instead of native