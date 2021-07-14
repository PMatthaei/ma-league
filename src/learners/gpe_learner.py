from components.episode_batch import EpisodeBatch
from controllers.gpe_controller import GPEController
from learners.learner import Learner
from torch.optim import RMSprop


class GPELearner(Learner):

    def __init__(self, mac: GPEController, scheme, logger, args):
        super().__init__(mac, scheme, logger, args)
        self.mac = mac
        self.optimizers = [RMSprop(params=sf.parameters(), lr=args.lr, alpha=args.optim_alpha, eps=args.optim_eps) for
                           sf in self.mac.sfs]

    def train(self, batch: EpisodeBatch, t_env: int, episode_num: int) -> None:
        for i in range(self.mac.n_features):  # For all
            self.optimizers[i].zero_grad()
            delta = feature[i] + gamma * successor_feature_i_of_policy_j(next_s,next_a) - successor_feature_i_of_policy_j(s, a)
            delta.backward()
            params_i = params_i + alpha * delta * gradient
            self.optimizers[i].step()
            pass

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
