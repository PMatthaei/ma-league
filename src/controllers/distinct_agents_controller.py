from __future__ import annotations

from components.episode_buffer import EpisodeBatch
from controllers.multi_agent_controller import MultiAgentController
from exceptions.mac_exceptions import HiddenStateNotInitialized
from modules.agents import REGISTRY as agent_REGISTRY
from components.action_selectors import REGISTRY as action_REGISTRY
import torch as th


class DistinctMAC(MultiAgentController):
    def __init__(self, scheme, groups, args):
        """
        This multi-agent controller without shared parameters between agent networks.
        :param scheme:
        :param groups:
        :param args:
        """
        self.n_agents = args.n_agents
        self.args = args
        input_shape = self._get_input_shape(scheme)
        self.agents = self._build_agents(input_shape)
        self.agent_output_type = args.agent_output_type

        self.action_selector = action_REGISTRY[args.action_selector](args)

        self.hidden_states = None

    def select_actions(self, ep_batch, t_ep, t_env, bs=slice(None), test_mode=False):
        # Only select available actions for the selected batch elements in bs
        avail_actions = ep_batch["avail_actions"][:, t_ep]
        # Run forward propagation for the batch -> Q-values
        agent_outs = self.forward(ep_batch, t_ep, test_mode=test_mode)
        # Choose action by f.e. epsilon-greedy
        chosen_actions = self.action_selector.select_action(agent_outs[bs], avail_actions[bs], t_env, test_mode)
        return chosen_actions

    def forward(self, ep_batch: EpisodeBatch, t, test_mode=False):
        agent_inputs = self._build_inputs(ep_batch, t)
        avail_actions = ep_batch["avail_actions"][:, t]
        if self.hidden_states is None:
            raise HiddenStateNotInitialized()

        agent_outs = []
        for i in range(self.n_agents):
            agent_out, self.hidden_states[i] = self.agents[i](agent_inputs[:, i, :], self.hidden_states[i])
            agent_outs.append(agent_out)
        agent_outs = th.stack(agent_outs)

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

    def init_hidden(self, batch_size):
        self.hidden_states = [
            agent.init_hidden().unsqueeze(0).expand(batch_size, 1, -1)  # bav
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

    def load_state(self, other_mac: DistinctMAC):
        [agent.load_state_dict(other_mac.agents[i].state_dict()) for i, agent in enumerate(self.agents)]

    def cuda(self):
        [agent.cuda() for agent in self.agents]

    def save_models(self, path, name):
        [th.save(agent.state_dict(), "{}/{}agent.th".format(path, name)) for agent in self.agents]

    def load_models(self, path, name):
        [
            agent.load_state_dict(
                th.load("{}/{}agent.th".format(path, name), map_location=lambda storage, loc: storage))
            for agent in self.agents
        ]

    def _build_agents(self, input_shape):
        return [agent_REGISTRY[self.args.agent](input_shape, self.args) for _ in range(self.n_agents)]

    def _build_inputs(self, batch: EpisodeBatch, t):
        """
        Select data from the batch which should be served as a input to the agent network.
        Assumes homogeneous agents with flat observations.
        Other MACs might want to e.g. delegate building inputs to each agent
        Runs in every forward pass
        :param batch:
        :param t:
        :return:
        """

        bs = batch.batch_size
        inputs = [batch["obs"][:, t]]
        if self.args.obs_last_action:
            if t == 0:
                inputs.append(th.zeros_like(batch["actions_onehot"][:, t]))
            else:
                inputs.append(batch["actions_onehot"][:, t - 1])
        if self.args.obs_agent_id:
            inputs.append(th.eye(self.n_agents, device=batch.device).unsqueeze(0).expand(bs, -1, -1))

        inputs = th.cat(inputs, dim=2)
        return inputs

    def _get_input_shape(self, scheme):
        input_shape = scheme["obs"]["vshape"]
        if self.args.obs_last_action:
            input_shape += scheme["actions_onehot"]["vshape"][0]
        if self.args.obs_agent_id:
            input_shape += self.n_agents

        return input_shape
