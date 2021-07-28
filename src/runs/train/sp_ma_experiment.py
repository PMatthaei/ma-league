from typing import OrderedDict

import torch as th

from controllers import REGISTRY as mac_REGISTRY
from runs.train.ma_experiment import MultiAgentExperiment
from steppers import SELF_REGISTRY as self_steppers_REGISTRY
from steppers.episode_stepper import EnvStepper


class SelfPlayMultiAgentExperiment(MultiAgentExperiment):

    def __init__(self, args, logger, finish_callback=None, episode_callback=None):
        """
        Self-Play replaces the opposing agent previously controlled by a static scripted AI with another static policy
        controlled agent. This agent is fixed during the training to prevent non-stationarity in the environment.
        :param args:
        :param logger:
        :param finish_callback:
        :param episode_callback:
        """
        super().__init__(args, logger)
        self.finish_callback = finish_callback
        self.episode_callback = episode_callback
        # WARN: Assuming the away agent uses the same buffer scheme!!
        self.away_mac = mac_REGISTRY[self.args.mac](self.home_buffer.scheme, self.groups, self.args)

    def load_adversary(self, agent: OrderedDict):
        self.away_mac.load_state_dict(agent=agent)
        del agent

    def _integrate_env_info(self):
        total_n_agents = self.env_info["n_agents"]
        # Since the AI is replaced with fixed policy adversary agents, we need to re-calculate
        # the amount of the learning agents for the scheme
        assert total_n_agents % 2 == 0, f"A total of {total_n_agents} agents in the env do not fit in the symmetric two-team scenario. " \
                                        f"Ensure the Self-Play scenario has two team set to is_scripted=False"
        per_team_n_agents = int(total_n_agents / 2)
        env_scheme = {
            "n_agents": per_team_n_agents,
            "n_actions": int(self.env_info["n_actions"]),
            "state_shape": int(self.env_info["state_shape"]),
            "total_n_agents": total_n_agents
        }
        self._update_args(env_scheme)
        return env_scheme

    def _build_stepper(self) -> EnvStepper:
        return self_steppers_REGISTRY[self.args.runner](args=self.args, logger=self.logger)

    def _init_stepper(self):
        # Give runner the scheme and most importantly BOTH multi-agent controllers
        self.stepper.initialize(scheme=self.scheme, groups=self.groups, preprocess=self.preprocess,
                                home_mac=self.home_mac,
                                away_mac=self.away_mac)

    def _finish(self):
        super()._finish()
        if self.finish_callback is not None:
            self.finish_callback()

    def _train_episode(self, episode_num, on_train_end=None):
        # Run for a whole episode at a time
        home_batch, _, env_info = self.stepper.run(test_mode=False)
        if self.episode_callback is not None:
            self.episode_callback(env_info)

        self.home_buffer.insert_episode_batch(home_batch)

        # Sample batch from buffer if possible
        batch_size = self.args.batch_size
        if self.home_buffer.can_sample(batch_size):
            home_sample = self.home_buffer.sample(batch_size)

            # Truncate batch to only filled timesteps
            max_ep_t_h = home_sample.max_t_filled()
            home_sample = home_sample[:, :max_ep_t_h]

            device = self.args.device
            if home_sample.device != device:
                home_sample.to(device)
            # ! WARN ! Only train the learning agent not it`s sampled self-play adversary
            self.home_learner.train(home_sample, self.stepper.t_env, episode_num)

            if on_train_end:
                on_train_end(self.learners)

    def evaluate_mean_returns(self, episode_n=1):
        self.logger.info("Evaluate for {} episodes.".format(episode_n))
        home_ep_rewards = th.zeros(episode_n)
        away_ep_rewards = home_ep_rewards.detach().clone()

        self._init_stepper()

        for i in range(episode_n):
            home_batch, away_batch, last_env_info = self.stepper.run(test_mode=True)
            home_ep_rewards[i] = th.sum(home_batch["reward"].flatten())
            away_ep_rewards[i] = th.sum(away_batch["reward"].flatten())

        self._finish()

        return th.mean(home_ep_rewards), th.mean(away_ep_rewards)
