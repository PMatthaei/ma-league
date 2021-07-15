from components.episode_batch import EpisodeBatch
from modules.networks.mlp import MLP
import torch as th


class PolicySuccessorFeatures(MLP):
    def __init__(self, in_shape, out_shape):
        super().__init__(in_shape, out_shape, 2, [64, 128])

    def forward(self, batch: EpisodeBatch, t: int=None):
        x = self._build_input(batch, t)
        super().forward(x)

    def _build_input(self, batch: EpisodeBatch, t: int):
        # TODO Build inputs to SF
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