from components.episode_batch import EpisodeBatch
from controllers.gpe_controller import GPEController
from learners.learner import Learner

import torch as th

from components.feature_functions import REGISTRY as FEATURE_FUNCTIONS


class GPELearner(Learner):

    def __init__(self, mac: GPEController, scheme, logger, args):
        super().__init__(mac, scheme, logger, args)
        self.mac = mac
        self.feature_func = FEATURE_FUNCTIONS["team_task"]
        self.gpe_gamma = 0.9  # Discount rate
        self.gpe_lr = 0.01  # Learning rate

    def train(self, batch: EpisodeBatch, t_env: int, episode_num: int) -> None:
        obs, a = batch["obs"], batch["actions"]
        next_obs = next_a = batch["next_obs"], batch["next_actions"]
        for successor in self.mac.sfs:
            delta = self.feature_func(obs, a, next_obs) + self.gpe_gamma * successor(next_obs, next_a) - successor(obs,a)
            successor.zero_grad()
            delta.backward()
            with th.no_grad():
                for param in successor.parameters():
                    param.copy_(param + self.gpe_lr * delta * param.grad)

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
