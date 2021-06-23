from __future__ import annotations

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
        self.args = args
        input_shape = self._get_input_shape(scheme)
        self.native_agent = self._build_agents(input_shape)  # Single sharing native network for the Multi-Agent
        self.specific_agents = dict()  # Dictionary holding the specific agent network for a given agent
        self.agent_output_type = args.agent_output_type

        self.action_selector = action_REGISTRY[args.action_selector](args)

        self.hidden_states = None

    def select_actions(self, ep_batch, t_ep, t_env, bs=slice(None), test_mode=False):
        # Only select available actions for the selected batch elements in bs
        avail_actions = ep_batch["avail_actions"][:, t_ep]
        # Run forward propagation for the batch -> Q-values
        agent_outs = self.forward(ep_batch, t_ep, test_mode=test_mode)
        # Choose action by f.e. epsilon-greedy
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

    def _get_specific_agents_ids(self):
        return self.specific_agents.keys() # All agents that rely on a specific network for inference instead of native

    def forward(self, ep_batch, t, test_mode=False):
        agent_inputs = self._build_inputs(ep_batch, t)
        avail_actions = ep_batch["avail_actions"][:, t]
        if self.hidden_states is None:
            raise HiddenStateNotInitialized()

        agent_outs = self._compute_agent_outputs(agent_inputs)

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

    def _compute_agent_outputs(self, agent_inputs):

        agent_outs, self.hidden_states = self.native_agent(agent_inputs, self.hidden_states)
        specific_outs = [agent(agent_inputs[:, aid, :]) for aid, agent in self.specific_agents.items()]
        return agent_outs

    def update_trained_steps(self, trained_steps):
        self.native_agent.trained_steps = trained_steps

    def init_hidden(self, batch_size):
        native_hidden_states = self.native_agent.init_hidden().unsqueeze(0).expand(batch_size, self.n_agents, -1)  # bav
        specific_hidden_states = [
            agent.init_hidden().unsqueeze(0).expand(batch_size, 1, -1)  # bav
            for agent in self.specific_agents.values()
        ]

    def parameters(self):
        params = []
        params += self.native_agent.parameters()
        [params + list(agent.parameters()) for agent in self.specific_agents.values()]
        return params

    def load_state(self, other_mac: CombinedMAC):
        self.native_agent.load_state_dict(other_mac.native_agent.state_dict())
        [
            agent.load_state_dict(other_mac.specific_agents[aid].state_dict())
            for aid, agent in enumerate(self.specific_agents.items())
        ]

    def cuda(self):
        self.native_agent.cuda()
        [agent.cuda() for agent in self.specific_agents.values()]

    def save_models(self, path, name):
        th.save(self.native_agent.state_dict(), "{}/{}agent_native.th".format(path, name))
        [
            th.save(agent.state_dict(), "{}/{}agent_specific_{}.th".format(path, name, i))
            for i, agent in enumerate(self.specific_agents.values())
        ]

    def load_models(self, path, name):
        self.native_agent.load_state_dict(
            th.load("{}/{}agent_native.th".format(path, name), map_location=lambda storage, loc: storage))
        [
            agent.load_state_dict(th.load("{}/{}agent_specific_{}.th".format(path, name, aid), lambda storage, loc: storage))
            for aid, agent in enumerate(self.specific_agents.items())
        ]

    def _build_agents(self, input_shape):
        return agent_REGISTRY[self.args.agent](input_shape, self.args)

    def _build_inputs(self, batch, t):
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

        inputs = th.cat([x.reshape(bs * self.n_agents, -1) for x in inputs], dim=1)
        return inputs

    def _get_input_shape(self, scheme):
        input_shape = scheme["obs"]["vshape"]
        if self.args.obs_last_action:
            input_shape += scheme["actions_onehot"]["vshape"][0]
        if self.args.obs_agent_id:
            input_shape += self.n_agents

        return input_shape
