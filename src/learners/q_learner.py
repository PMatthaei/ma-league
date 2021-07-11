import copy
import torch as th
from torch.optim import RMSprop

from controllers.multi_agent_controller import MultiAgentController
from learners.learner import Learner
from modules.mixers.qmix import QMixer
from modules.mixers.vdn import VDNMixer
from components.episode_batch import EpisodeBatch


class QLearner(Learner):
    def __init__(self, mac: MultiAgentController, scheme, logger, args, name=None):
        super().__init__(mac, scheme, logger, args, name)
        self.name += "_qlearner_"

        self.last_target_update_episode = 0

        self.mixer = None
        if args.mixer is not None:
            if args.mixer == "vdn":
                self.mixer = VDNMixer()
            elif args.mixer == "qmix":
                self.mixer = QMixer(args)
            else:
                raise ValueError("Mixer {} not recognised.".format(args.mixer))
            # Add additional mixer params for later optimization
            self.params += list(self.mixer.parameters())
            self.target_mixer = copy.deepcopy(self.mixer)

        self.optimiser = RMSprop(params=self.params, lr=args.lr, alpha=args.optim_alpha, eps=args.optim_eps)

        # TODO: a little wasteful to deepcopy (e.g. duplicates action selector), but should work for any MAC
        self.target_mac = copy.deepcopy(mac)

    def train(self, batch: EpisodeBatch, t_env: int, episode_num: int):
        # Get the relevant batch quantities
        rewards = batch["reward"][:, :-1]
        actions = batch["actions"][:, :-1]
        terminated = batch["terminated"][:, :-1].float()
        # Filled boolean indicates if steps were filled to match max. sequence length in the batch
        mask = batch["filled"][:, :-1].float()

        mask[:, 1:] = mask[:, 1:] * (1 - terminated[:, :-1])
        avail_actions = batch["avail_actions"]

        # Calculate estimated Q-Values
        mac_out = []
        self.mac.init_hidden(batch.batch_size)
        # Iterate over all timesteps defined by the max. in the batch
        for t in range(batch.max_seq_length):
            agent_outs = self.mac.forward(batch, t=t)
            mac_out.append(agent_outs)
        mac_out = th.stack(mac_out, dim=1)  # Concat over time

        # Pick the Q-Values for the actions taken by each agent
        chosen_action_qvals = th.gather(mac_out[:, :-1], dim=3, index=actions).squeeze(3)  # Remove the last dim

        # Calculate the Q-Values necessary for the target
        target_mac_out = []
        self.target_mac.init_hidden(batch.batch_size)
        for t in range(batch.max_seq_length):
            target_agent_outs = self.target_mac.forward(batch, t=t)
            target_mac_out.append(target_agent_outs)

        # We don't need the first timesteps Q-Value estimate for calculating targets
        target_mac_out = th.stack(target_mac_out[1:], dim=1)  # Concat across time

        # Mask out unavailable actions by setting utility very low
        target_mac_out[avail_actions[:, 1:] == 0] = -9999999

        if self.args.double_q:
            # Get actions that maximise live Q (for double q-learning)
            mac_out_detach = mac_out.clone().detach()
            mac_out_detach[avail_actions == 0] = -9999999
            cur_max_actions = mac_out_detach[:, 1:].max(dim=3, keepdim=True)[1]
            target_max_qvals = th.gather(target_mac_out, 3, cur_max_actions).squeeze(3)
        else:
            # Max over target Q-Values
            target_max_qvals = target_mac_out.max(dim=3)[0]

        # Mix
        if self.mixer is not None:
            chosen_action_qvals = self.mixer(chosen_action_qvals, batch["state"][:, :-1])
            target_max_qvals = self.target_mixer(target_max_qvals, batch["state"][:, 1:])

        # Calculate 1-step Q-Learning targets
        targets = rewards + self.args.gamma * (1 - terminated) * target_max_qvals

        # Td-error
        td_error = (chosen_action_qvals - targets.detach())

        # Mask out previously filled time steps if the env was already terminated in the corresponding batch entry
        mask = mask.expand_as(td_error)

        # 0-out the targets that came from padded data
        masked_td_error = td_error * mask

        # Normal L2 loss, take mean over actual data
        loss = (masked_td_error ** 2).sum() / mask.sum()

        # Optimise
        self.optimiser.zero_grad()
        # Computes dloss/dx for every parameter x which has requires_grad=True.
        loss.backward()
        grad_norm = th.nn.utils.clip_grad_norm_(self.params, self.args.grad_norm_clip)
        self.optimiser.step()

        # Update target in interval
        if (episode_num - self.last_target_update_episode) / self.args.target_update_interval >= 1.0:
            self._update_targets()
            self.last_target_update_episode = episode_num

        self.mac.update_trained_steps(t_env)

        # Log learner stats in interval
        if t_env - self.log_stats_t >= self.args.learner_log_interval:
            self.logger.log_stat(self.name + "loss", loss.item(), t_env)
            self.logger.log_stat(self.name + "grad_norm", grad_norm.cpu().numpy(), t_env)
            mask_elems = mask.sum().item()
            self.logger.log_stat(self.name + "td_error_abs", (masked_td_error.abs().sum().item() / mask_elems), t_env)
            self.logger.log_stat(self.name + "q_taken_mean",
                                 (chosen_action_qvals * mask).sum().item() / (mask_elems * self.args.n_agents), t_env)
            self.logger.log_stat(self.name + "target_mean",
                                 (targets * mask).sum().item() / (mask_elems * self.args.n_agents), t_env)
            self.log_stats_t = t_env

    def _update_targets(self):
        self.target_mac.load_state(other_mac=self.mac)
        if self.mixer is not None:
            self.target_mixer.load_state_dict(self.mixer.state_dict())
        self.logger.info("Updated {0}target network.".format(self.name))

    def cuda(self):
        self.mac.cuda()
        self.target_mac.cuda()
        if self.mixer is not None:
            self.mixer.cuda()
            self.target_mixer.cuda()

    def save_models(self, path, name):
        self.mac.save_models(path, name=self.name)
        if self.mixer is not None:
            th.save(self.mixer.state_dict(), "{}/{}mixer.th".format(path, self.name))
        th.save(self.optimiser.state_dict(), "{}/{}opt.th".format(path, self.name))

    def load_models(self, path):
        self.mac.load_models(path, self.name)
        # Not quite right but I don't want to save target networks
        self.target_mac.load_models(path, self.name)
        if self.mixer is not None:
            self.mixer.load_state_dict(
                th.load("{}/{}mixer.th".format(path, self.name), map_location=lambda storage, loc: storage))
        self.optimiser.load_state_dict(
            th.load("{}/{}opt.th".format(path, self.name), map_location=lambda storage, loc: storage))
