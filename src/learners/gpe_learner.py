from components.episode_batch import EpisodeBatch
from controllers.gpe_controller import GPEController
from learners.learner import Learner
from torch.optim import RMSprop

import torch as th


class GPELearner(Learner):

    def __init__(self, mac: GPEController, scheme, logger, args):
        """
        Learns a set of
        :param mac:
        :param scheme:
        :param logger:
        :param args:
        """
        super().__init__(mac, scheme, logger, args)
        self.mac = mac
        self.gpe_gamma = 0.9  # Discount rate
        self.gpe_lr = 0.01  # Learning rate

        self.optimisers = [
            RMSprop(params=self.params, lr=args.lr, alpha=args.optim_alpha, eps=args.optim_eps)
            for _ in range(len(self.mac.sfs))
        ]

    def train(self, batch: EpisodeBatch, t_env: int, episode_num: int) -> None:
        # Get the relevant batch quantities
        features = batch["features"][:, :-1]
        actions = batch["actions"][:, :-1]
        terminated = batch["terminated"][:, :-1].float()
        # Filled boolean indicates if steps were filled to match max. sequence length in the batch
        mask = batch["filled"][:, :-1].float()

        mask[:, 1:] = mask[:, 1:] * (1 - terminated[:, :-1])
        avail_actions = batch["avail_actions"]

        # Iterate over all timesteps defined by the max. in the episode batch
        sf_tensor = []
        for t in range(batch.max_seq_length):
            feature_outs = []
            inputs = self._build_sfs_inputs(batch, t)
            for successor_feature in self.mac.sfs:  # Iterate over successor features
                outs = successor_feature(inputs)
                feature_outs.append(outs)
            feature_outs = th.stack(feature_outs, dim=1)
            sf_tensor.append(feature_outs)
        sf_tensor = th.stack(sf_tensor, dim=1)

        q = th.gather(sf_tensor[:, :-1], dim=3, index=actions).squeeze(3)  # Remove the last dim

        next_sf_tensor = []
        for t in range(batch.max_seq_length):
            feature_outs = []
            inputs = self._build_sfs_inputs(batch, t)
            for successor_feature in self.mac.sfs:
                outs = successor_feature(inputs)
                feature_outs.append(outs)
            feature_outs = th.stack(feature_outs, dim=1)
            next_sf_tensor.append(feature_outs)
        q_next = th.stack(next_sf_tensor[1:], dim=1)

        td_error = features + self.gpe_gamma * ((1 - terminated) * q_next) - q
        # Mask out previously filled time steps if the env was already terminated in the corresponding batch entry
        mask = mask.expand_as(td_error)

        # 0-out the targets that came from padded data
        masked_td_error = td_error * mask

        # Normal L2 loss, take mean over actual data
        loss = (masked_td_error ** 2).sum() / mask.sum()

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        # Update
        # for t in range(batch.max_seq_length):
        #     o, a, o_, a_ = self._build_inputs(batch, t)
        #
        #     for feature in self.mac.sfs:
        #         delta = self.mac.phi(o, a, o_) + self.gpe_gamma * feature(o_, a_, t=t) - feature(o, a, t=t)
        #         feature.zero_grad()
        #         delta.backward()
        #         with th.no_grad():
        #             for param in feature.parameters():
        #                 param.copy_(param + self.gpe_lr * delta * param.grad)

    def _build_sfs_inputs(self, batch: EpisodeBatch, t):
        return 1, 1, 1, 1

    def cuda(self) -> None:
        pass

    def save_models(self, path, name):
        pass

    def load_models(self, path):
        pass

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
