import copy

import torch as th
from torch.optim import RMSprop

from components.episode_batch import EpisodeBatch
from learners.learner import Learner
from modules.critics.coma import COMACritic


def build_td_lambda_targets(rewards, terminated, mask, target_qs, n_agents, gamma, td_lambda):
    # Assumes  <target_qs > in B*T*A and <reward >, <terminated >, <mask > in (at least) B*T-1*1
    # Initialise  last  lambda -return  for  not  terminated  episodes
    ret = target_qs.new_zeros(*target_qs.shape)
    ret[:, -1] = target_qs[:, -1] * (1 - th.sum(terminated, dim=1))
    # Backwards  recursive  update  of the "forward  view"
    batch_size = ret.shape[1]  # Calc batch size based on incoming rewards collected
    for t in list(reversed(range(batch_size - 1))):
        ret[:, t] = td_lambda * gamma * ret[:, t + 1] + mask[:, t] \
                    * (rewards[:, t] + (1 - td_lambda) * gamma * target_qs[:, t + 1] * (1 - terminated[:, t]))
    # Returns lambda-return from t=0 to t=T-1, i.e. in B*T-1*A
    return ret[:, 0:-1]


class COMALearner(Learner):
    def __init__(self, mac, scheme, logger, args, name=None):
        super().__init__(mac, scheme, logger, args, name)
        self.args = args
        self.n_agents = args.n_agents
        self.n_actions = args.n_actions
        self.mac = mac
        self.logger = logger

        self.last_target_update_step = 0
        self.critic_training_steps = 0

        self.log_stats_t = -self.args.learner_log_interval - 1

        self.critic = COMACritic(scheme, args)
        self.target_critic = copy.deepcopy(self.critic)

        self.agent_params = list(mac.parameters())
        self.critic_params = list(self.critic.parameters())
        self.params = self.agent_params + self.critic_params

        self.agent_optimiser = RMSprop(params=self.agent_params, lr=args.lr, alpha=args.optim_alpha, eps=args.optim_eps)
        self.critic_optimiser = RMSprop(params=self.critic_params, lr=args.critic_lr, alpha=args.optim_alpha,
                                        eps=args.optim_eps)

    def train(self, batch: EpisodeBatch, t_env: int, episode_num: int):
        # Get the relevant quantities
        rs = batch["reward"][:, :-1]
        actions = batch["actions"][:, :]
        terminated = batch["terminated"][:, :-1].float()
        mask = batch["filled"][:, :-1].float()
        mask[:, 1:] = mask[:, 1:] * (1 - terminated[:, :-1])
        avail_actions = batch["avail_actions"][:, :-1]

        critic_mask = mask.clone()

        mask = mask.repeat(1, 1, self.n_agents).view(-1)

        # Q with counterfactual joint action u without action a per agent
        q_vals, critic_train_stats = self._train_critic(batch, rs, terminated, actions, avail_actions, critic_mask)

        actions = actions[:, :-1]

        mac_out = []
        self.mac.init_hidden(batch.batch_size)
        # Infer agent outputs over all timesteps but the last
        for t in range(batch.max_seq_length - 1):
            agent_outs = self.mac.forward(batch, t=t)
            mac_out.append(agent_outs)
        mac_out = th.stack(mac_out, dim=1)  # Concat over time

        mac_out[avail_actions == 0] = 0  # Mask out unavailable actions
        mac_out = mac_out / mac_out.sum(dim=-1, keepdim=True)  # Renormalize (as in action selection)
        mac_out[avail_actions == 0] = 0

        # Calculated baseline
        # !
        # Reshape assumes one agent since network is shared and agent-specific actions are deduced by agent id in obs
        # !
        q_vals = q_vals.reshape(-1, self.n_actions)  # q values for each action across all agents and timesteps
        pi = mac_out.view(-1, self.n_actions)  # pi for each action across all agents and timesteps
        baseline = (pi * q_vals).sum(dim=-1).detach()

        # Calculate policy grad with mask
        q_taken = th.gather(q_vals, dim=1, index=actions.reshape(-1, 1)).squeeze(1)  # q values of taken actions
        pi_taken = th.gather(pi, dim=1, index=actions.reshape(-1, 1)).squeeze(1)  # pi of taken actions
        pi_taken[mask == 0] = 1.0
        log_pi_taken = th.log(pi_taken)

        advantages = (q_taken - baseline).detach()

        # COMA gradient
        coma_loss = - ((log_pi_taken * advantages) * mask).sum() / mask.sum()

        # Optimise agents
        self.agent_optimiser.zero_grad()
        coma_loss.backward()
        grad_norm = th.nn.utils.clip_grad_norm_(self.agent_params, self.args.grad_norm_clip)
        self.agent_optimiser.step()

        if (self.critic_training_steps - self.last_target_update_step) / self.args.target_update_interval >= 1.0:
            self._update_targets()
            self.last_target_update_step = self.critic_training_steps

        if t_env - self.log_stats_t >= self.args.learner_log_interval:
            ts_logged = len(critic_train_stats["critic_loss"])
            for key in ["critic_loss", "critic_grad_norm", "td_error_abs", "q_taken_mean", "target_mean"]:
                self.logger.log_stat(key, sum(critic_train_stats[key]) / ts_logged, t_env)

            self.logger.log_stat(self.name + "advantage_mean", (advantages * mask).sum().item() / mask.sum().item(),
                                 t_env)
            self.logger.log_stat(self.name + "coma_loss", coma_loss.item(), t_env)
            self.logger.log_stat(self.name + "agent_grad_norm", grad_norm, t_env)
            self.logger.log_stat(self.name + "pi_max", (pi.max(dim=1)[0] * mask).sum().item() / mask.sum().item(),
                                 t_env)
            self.log_stats_t = t_env

    def _train_critic(self, batch: EpisodeBatch, rewards, terminated, actions, avail_actions, mask):
        target_q_vals = self.target_critic(batch)[:, :]  # Infer targets
        targets_taken = th.gather(target_q_vals, dim=3, index=actions).squeeze(3)  # targets of actions taken

        # Calculate td-lambda targets
        targets = build_td_lambda_targets(rewards, terminated, mask, targets_taken, self.n_agents, self.args.gamma,
                                          self.args.td_lambda)
        running_log = {
            "critic_loss": [],
            "critic_grad_norm": [],
            "td_error_abs": [],
            "target_mean": [],
            "q_taken_mean": [],
        }

        q_vals = th.zeros_like(target_q_vals)[:, :-1]  # Construct Q values tensor to fill with upcoming loop

        # Iterate over timesteps backwards but perform backward propagation of loss
        for t in reversed(range(batch.max_seq_length - 1)):
            mask_t = mask[:, t].expand(-1, self.n_agents)
            if mask_t.sum() == 0:
                continue  # Everything would be masked in this timestep> skip

            q_t = self.critic(batch, t=t)
            q_vals[:, t] = q_t.view(batch.batch_size, self.n_agents, self.n_actions)  # Copy qs at t into tensor
            q_taken = th.gather(q_t, dim=3, index=actions[:, t:t + 1]).squeeze(3).squeeze(1)  # Taken qs at timestep t
            targets_t = targets[:, t]  # Target qs at timestep

            td_error = (q_taken - targets_t.detach())

            # 0-out the targets that came from padded data -> episodes which were shorter than max episode in batch
            masked_td_error = td_error * mask_t

            # Normal L2 loss, take mean over actual data
            loss = (masked_td_error ** 2).sum() / mask_t.sum()
            self.critic_optimiser.zero_grad()  # Clear x.grad for every parameter x in the optimizer
            loss.backward()  # Compute dloss/dx for every parameter x which has requires_grad=True
            grad_norm = th.nn.utils.clip_grad_norm_(self.critic_params, self.args.grad_norm_clip)
            self.critic_optimiser.step()  # Update the value of parameters using the gradient
            self.critic_training_steps += 1

            running_log["critic_loss"].append(loss.item())
            running_log["critic_grad_norm"].append(grad_norm)
            mask_elems = mask_t.sum().item()
            running_log["td_error_abs"].append((masked_td_error.abs().sum().item() / mask_elems))
            running_log["q_taken_mean"].append((q_taken * mask_t).sum().item() / mask_elems)
            running_log["target_mean"].append((targets_t * mask_t).sum().item() / mask_elems)

        return q_vals, running_log

    def _update_targets(self):
        self.target_critic.load_state_dict(self.critic.state_dict())
        self.logger.info("Updated target network")

    def save_models(self, path, name):
        self.mac.save_models(path, name=self.name)
        th.save(self.critic.state_dict(), "{}/critic.th".format(path))
        th.save(self.agent_optimiser.state_dict(), "{}/agent_opt.th".format(path))
        th.save(self.critic_optimiser.state_dict(), "{}/critic_opt.th".format(path))

    def load_models(self, path):
        self.mac.load_models(path, name=self.name)
        self.critic.load_state_dict(th.load("{}/critic.th".format(path), map_location=lambda storage, loc: storage))
        # Not quite right but I don't want to save target networks
        self.target_critic.load_state_dict(self.critic.state_dict())
        self.agent_optimiser.load_state_dict(
            th.load("{}/agent_opt.th".format(path), map_location=lambda storage, loc: storage))
        self.critic_optimiser.load_state_dict(
            th.load("{}/critic_opt.th".format(path), map_location=lambda storage, loc: storage))
