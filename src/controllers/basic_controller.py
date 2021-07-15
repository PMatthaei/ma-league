from __future__ import annotations

from components.episode_batch import EpisodeBatch
from controllers.multi_agent_controller import MultiAgentController
from exceptions.mac_exceptions import HiddenStateNotInitialized
from modules.agents import REGISTRY as agent_REGISTRY
import torch as th


class BasicMAC(MultiAgentController):
    def __init__(self, scheme, groups, args):
        """
        This multi-agent controller shares parameters between agents by using a single fully connected DQRN.

        Input Building:
        Select data from the batch which should be served as a input to the agent network.
        Assumes homogeneous agents with flat observations.
        Other MACs might want to e.g. delegate building inputs to each agent
        Runs in every forward pass

        :param scheme:
        :param groups:
        :param args:
        """
        super().__init__(scheme, groups, args)

    def select_actions(self, ep_batch, t_ep, t_env, bs=slice(None), test_mode=False):
        # Only select available actions for the selected batch elements in bs
        avail_actions = ep_batch["avail_actions"][:, t_ep]
        # Run forward propagation for the batch -> Q-values
        agent_outs = self.forward(ep_batch, t_ep, test_mode=test_mode)
        # Choose action by f.e. epsilon-greedy (except test mode is False)
        chosen_actions, is_greedy = self.action_selector.select(agent_outs[bs], avail_actions[bs], t_env, test_mode)
        return chosen_actions, is_greedy

    def forward(self, ep_batch, t, test_mode=False):
        agent_inputs = self._build_inputs(ep_batch, t)
        if self.hidden_states is None:
            raise HiddenStateNotInitialized()

        agent_outs = self._compute_agent_outputs(agent_inputs)

        # Softmax the agent outputs if they're policy logits
        if self.agent_output_type == "pi_logits":

            agent_outs = self._softmax(agent_outs, ep_batch, t, test_mode)

        return agent_outs.view(ep_batch.batch_size, self.n_agents, -1)

    def _compute_agent_outputs(self, agent_inputs):
        agent_outs, self.hidden_states = self.agent(agent_inputs, self.hidden_states)
        return agent_outs

    def update_trained_steps(self, trained_steps):
        self.agent.trained_steps = trained_steps

    def init_hidden(self, batch_size):
        self.hidden_states = self.agent.init_hidden().unsqueeze(0).expand(batch_size, self.n_agents, -1)  # bav

    def parameters(self):
        return self.agent.parameters()

    def load_state(self, other_mac: BasicMAC, agent=None):
        self.agent.load_state_dict(agent.state_dict() if agent is not None else other_mac.agent.state_dict())

    def cuda(self):
        self.agent.cuda()

    def save_models(self, path, name):
        th.save(self.agent.state_dict(), "{}/{}agent.th".format(path, name))

    def load_models(self, path, name):
        self.agent.load_state_dict(
            th.load("{}/{}agent.th".format(path, name), map_location=lambda storage, loc: storage))

    def _build_agents(self, input_shape):
        return agent_REGISTRY[self.args.agent](input_shape, self.args)

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

    def _get_input_shape(self, scheme):
        input_shape = scheme["obs"]["vshape"]
        if self.args.obs_last_action:
            input_shape += scheme["actions_onehot"]["vshape"][0]
        if self.args.obs_agent_id:
            input_shape += self.n_agents

        return input_shape
