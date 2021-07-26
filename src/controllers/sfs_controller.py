from itertools import product

import torch as th
from torch import Tensor

from components.episode_batch import EpisodeBatch
from controllers.multi_agent_controller import MultiAgentController
from modules.layers.policy_successor_features import PolicySuccessorFeatures


class SFSController(MultiAgentController):
    def __init__(self, scheme, groups, args):
        self.args = args
        self.n_agents = args.n_agents
        self.n_actions = args.n_actions

        self.n_features = self._get_feature_shape(scheme)  # (=d)
        self.input_shape = self._get_input_shape(scheme)
        # Set of all task inducing weight vectors
        self.W = th.tensor([x for x in product([-1, 0, 1], repeat=self.n_features) if sum(x) >= 0], dtype=th.float)
        self.n_policies = len(self.W)  # number of policies induced via d-dim weight vectors
        self.policy_idx = None  # (=j)
        super().__init__(scheme, groups, args)

    def _build_agent(self, input_shape):
        # Barreto et al. propose one MLP per Feature with two hidden layers sized 64 and 128
        return [
            PolicySuccessorFeatures(in_shape=self.input_shape, out_shape=self.n_policies * self.n_actions)
            for _ in range(self.n_features)
        ]

    def select_actions(self, ep_batch: EpisodeBatch, t_ep: int, t_env: int, bs=slice(None), test_mode=False):
        avail_actions = ep_batch["avail_actions"][:, t_ep]
        agent_outs = self.forward(ep_batch, t=t_ep, test_mode=test_mode)
        chosen_actions, is_greedy = self.action_selector.select(agent_outs[bs], avail_actions[bs], t_env, test_mode)
        return chosen_actions, is_greedy

    def forward(self, ep_batch: EpisodeBatch, t: int, test_mode=False):
        if t == 0:  # Choose a random policy at each detected timestep reset to follow the whole episode
            self.policy_idx = th.randint(low=0, high=self.n_policies - 1, size=(1,))

        agent_inputs = self._build_inputs(ep_batch, t)
        agent_outs = self._compute_agent_outputs(agent_inputs, bs=ep_batch.batch_size)
        return agent_outs

    def _compute_agent_outputs(self, agent_inputs: Tensor, bs: int) -> Tensor:
        w_j = self.W[self.policy_idx].squeeze()
        outs = th.stack([sf(agent_inputs) for sf in self.agent])
        # Successor Features Tensor
        sfs = outs.view(bs, self.n_agents, self.n_policies, self.n_features, -1)
        sfs_T = sfs.permute(0, 1, 2, 4, 3)
        sf_j_T = sfs_T[:, :, self.policy_idx, :].squeeze(2)
        return sf_j_T @ w_j

    def _build_inputs(self, batch: EpisodeBatch, t: int) -> Tensor:
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

    def _get_feature_shape(self, scheme):
        return scheme["features"]["vshape"][0]

    def init_hidden(self, batch_size: int):
        pass

    def parameters(self):
        return []  # No parameters here, these are handled in the PSFs

    def load_state(self, other_mac: MultiAgentController):
        pass

    def cuda(self):
        pass

    def save_models(self, path: str, name: str):
        pass

    def load_models(self, path: str, name: str):
        pass

    def update_trained_steps(self, trained_steps: int):
        pass
