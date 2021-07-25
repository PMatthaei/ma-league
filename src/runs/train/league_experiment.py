from typing import Dict

from controllers import EnsembleMAC
from modules.agents.agent_network import AgentNetwork

from runs.train.sp_ma_experiment import SelfPlayMultiAgentExperiment


class LeagueExperiment(SelfPlayMultiAgentExperiment):

    def __init__(self, args, logger, finish_callback=None, on_episode_end=None):
        """
        LeaguePlay performs training of a single multi-agent and offers loading of new adversarial agents.
        :param args:
        :param logger:
        :param finish_callback:
        :param on_episode_end:
        """
        super().__init__(args, logger)
        self.finish_callback = finish_callback
        self.episode_callback = on_episode_end

    def load_adversary(self, agent: AgentNetwork):
        self.away_mac.load_state(agent=agent, other_mac=None)

    def _test(self, n_test_runs):
        self.last_test_T = self.stepper.t_env
        pass  # Skip tests in league to save computing time
