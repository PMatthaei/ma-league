from __future__ import annotations

from typing import Dict

from controllers.multi_agent_controller import MultiAgentController
from modules.agents import REGISTRY as agent_REGISTRY, Agent
import torch as th


class EnsembleInferenceMAC(MultiAgentController):
    def __init__(self, scheme, groups, args):
        """
        This is a multi-agent controller uses a combination of networks for inference. Each agent can choose to either
        infer with the original/native agent network or an ensemble of networks which were trained within different teams.

        This MAC can only be used as a fixed policy opponent within league training!

        :param scheme:
        :param groups:
        :param args:
        """
        super().__init__(scheme, groups, args)
        self.ensemble = dict()  # Dictionary holding the specific agent network for a given agent
        self.native_hidden_states = None
        self.specific_hidden_states = None

        self.update(0, self.agent)
        self._all_ids = set(range(self.n_agents))

    def update(self, aid: int, agent: Agent):
        """
        Replaces inference for the given agent with id = aid with the inference of the provided agent by adding into the
        dict.
        :param aid:
        :param agent:
        :return:
        """
        self.ensemble.update({aid: agent})

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
        native_agent_outs, self.native_hidden_states = self.agent(native_inputs, self.native_hidden_states)
        agent_outs = native_agent_outs.view(batch_size, self.n_agents, -1)
        for aid, specific_agent in self.ensemble.items():
            agent_outs[:, aid, :], self.specific_hidden_states[aid] = specific_agent(
                specific_inputs[aid, :].view(batch_size, -1), self.specific_hidden_states[aid])
        return agent_outs.view(batch_size * self.n_agents, -1)

    def update_trained_steps(self, trained_steps):
        self.agent.trained_steps = trained_steps

    def init_hidden(self, batch_size):
        self.native_hidden_states = self.agent.init_hidden().unsqueeze(0).expand(batch_size, self.n_agents, -1)
        self.specific_hidden_states = {
            aid: agent.init_hidden().unsqueeze(0).expand(batch_size, 1, -1)  # bav
            for aid, agent in self.ensemble.items()
        }

    def load_state(self, agent: Agent = None, ensemble: Dict[int, Agent] = None):
        self.agent.load_state_dict(agent.state_dict()) if agent is not None else None
        [
            agent.load_state_dict(ensemble[i].state_dict())
            for i, agent in self.ensemble.items()
        ] if ensemble is not None else None

    def cuda(self):
        self.agent.cuda()
        [agent.cuda() for agent in self.ensemble.values()]

    def parameters(self):
        raise NotImplementedError("This functionality is not available because this MAC can only perform inference.")

    def save_models(self, path, name):
        raise NotImplementedError("This functionality is not available because this MAC can only perform inference.")

    def load_models(self, path, name):
        raise NotImplementedError("This functionality is not available because this MAC can only perform inference.")

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
