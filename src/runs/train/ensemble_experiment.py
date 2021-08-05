from typing import OrderedDict

from marl.controllers import EnsembleMAC
from runs.train.ma_experiment import MultiAgentExperiment


class EnsembleExperiment(MultiAgentExperiment):

    def __init__(self, args, logger, on_episode_end=None, log_start_t=0):
        """
        LeaguePlay performs training of a single multi-agent and offers loading of new adversarial agents.
        :param args:
        :param logger:
        :param on_episode_end:
        """
        super().__init__(args, logger, on_episode_end, log_start_t)
        assert isinstance(self.home_mac, EnsembleMAC), 'Ensemble experiment enforces "mac"=ensemble in configuration'

    def load_ensemble(self, native: OrderedDict, foreign_agent: OrderedDict):
        """
        Build an dual ensemble where parts of the native agent infer with the foreign agent
        :param native:
        :param foreign_agent:
        :return:
        """
        self.home_mac.load_state_dict(agent=native)  # Load the native agent and freeze its weights
        self.home_mac.freeze_agent_weights()
        # ! WARN ! Currently it is enforced that all teams have the agent to swap in the first(=0) position
        self.home_mac.load_state_dict(ensemble={0: foreign_agent})  # Load foreign agent into first agent in ensemble.

        self.home_learner.update_targets() # Init target mac with ensemble
