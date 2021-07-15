from components.episode_batch import EpisodeBatch
from controllers.gpe_controller import GPEController
from learners.learner import Learner

import torch as th


class GPELearner(Learner):

    def __init__(self, mac: GPEController, scheme, logger, args):
        super().__init__(mac, scheme, logger, args)
        self.mac = mac
        self.gpe_gamma = 0.9  # Discount rate
        self.gpe_lr = 0.01  # Learning rate

    def train(self, batch: EpisodeBatch, t_env: int, episode_num: int) -> None:
        # Get the relevant batch quantities
        rewards = batch["reward"][:, :-1]
        actions = batch["actions"][:, :-1]
        terminated = batch["terminated"][:, :-1].float()
        # Filled boolean indicates if steps were filled to match max. sequence length in the batch
        mask = batch["filled"][:, :-1].float()

        mask[:, 1:] = mask[:, 1:] * (1 - terminated[:, :-1])
        avail_actions = batch["avail_actions"]

        # Iterate over all timesteps defined by the max. in the batch
        for t in range(batch.max_seq_length):
            for successor in self.mac.sfs:
                o, a, o_, a_ = self._build_inputs(batch)
                delta = self.mac.phi(o, a, o_) + self.gpe_gamma * successor(o_, a_, t=t) - successor(o, a, t=t)
                successor.zero_grad()
                delta.backward()
                with th.no_grad():
                    for param in successor.parameters():
                        param.copy_(param + self.gpe_lr * delta * param.grad)

    def _build_inputs(self, batch: EpisodeBatch):
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
