import torch as th

from components.episode_batch import EpisodeBatch
from controllers.sfs_controller import SFSController
from learners.learner import Learner


class SFSLearner(Learner):

    def __init__(self, mac: SFSController, scheme, logger, args, name=None):
        """
        Learns a set of successor features
        :param mac:
        :param scheme:
        :param logger:
        :param args:
        """
        super().__init__(mac, scheme, logger, args, name)
        self.mac = mac
        self.gpe_gamma = 0.9  # Discount rate
        self.gpe_lr = 0.01  # Learning rate

    def train(self, batch: EpisodeBatch, t_env: int, episode_num: int) -> None:
        # Get the relevant batch quantities
        features = batch["features"][:, :-1]
        actions = batch["actions"][:, :-1]
        terminated = batch["terminated"][:, :-1].float()
        # Filled boolean indicates if steps were filled to match max. sequence length in the batch
        mask = batch["filled"][:, :-1].float()

        mask[:, 1:] = mask[:, 1:] * (1 - terminated[:, :-1])
        avail_actions = batch["avail_actions"]

        for feature_idx, successor_feature in enumerate(self.mac.agent):  # Iterate over successor features
            mac_out = [self.mac.forward(batch, t=t) for t in range(batch.max_seq_length)]
            q_vals = th.gather(th.stack(mac_out, dim=1)[:, :-1], dim=3, index=actions).squeeze(3)  # Remove the last dim

            # We don't need the first timesteps Q-Value estimate for calculating targets
            q_next_vals = th.stack(mac_out[1:], dim=1)

            # Mask out unavailable actions by setting utility very low
            q_next_vals[avail_actions[:, 1:] == 0] = -9999999

            q_next_vals = q_next_vals.max(dim=3)[0]
            terminated_q_next_vals = (1 - terminated) * q_next_vals  # Mask out vals where env already terminated
            feature_i = th.unsqueeze(features[:, :, feature_idx], dim=2)
            td_error = feature_i + self.gpe_gamma * terminated_q_next_vals - q_vals
            # Mask out previously filled time steps if the env was already terminated in the corresponding batch entry
            mask = mask.expand_as(td_error)

            # 0-out the targets that came from padded data
            masked_td_error = td_error * mask

            # Normal L2 loss, take mean over actual data
            loss = (masked_td_error ** 2).sum() / mask.sum()

            successor_feature.optimizer.zero_grad()
            loss.backward()
            successor_feature.optimizer.step()

    def _build_sfs_inputs(self, batch: EpisodeBatch, t):
        return 1, 1, 1, 1

    def cuda(self) -> None:
        pass

    def save_models(self, path, name):
        pass

    def load_models(self, path):
        pass

    def gpi(self, observation, cumulant_weights):
        q_values = self.__call__(th.expand_dims(observation, axis=0))[0]
        q_w = th.tensordot(q_values, cumulant_weights, axes=[1, 0])  # [P,a]
        q_w_actions = th.reduce_max(q_w, axis=0)

        action = th.cast(th.argmax(q_w_actions), th.int32)

        return action

    """
    pred = model(inp)
    loss = critetion(pred, ground_truth)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    same as:
    
    pred = model(inp)
    loss = your_loss(pred)
    model.zero_grad()
    loss.backward()
    with torch.no_grad():
      for p in model.parameters():
        new_val = update_function(p, p.grad, loss, other_params)
        p.copy_(new_val)
    """
