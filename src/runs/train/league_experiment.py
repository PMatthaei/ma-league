from typing import OrderedDict

from runs.train.sp_ma_experiment import SelfPlayMultiAgentExperiment


class LeagueExperiment(SelfPlayMultiAgentExperiment):

    def __init__(self, args, logger, on_episode_end=None,  log_start_t=0):
        """
        LeaguePlay performs training of a single multi-agent and offers loading of new adversarial agents.
        :param args:
        :param logger:
        :param finish_callback:
        :param on_episode_end:
        """
        super().__init__(args, logger, on_episode_end=on_episode_end, log_start_t=log_start_t)

    def _test(self, n_test_runs):
        self.last_test_T = self.stepper.t_env
        pass  # Skip tests in league to save computing time

    def load_home_agent(self, agent: OrderedDict):
        self.home_mac.load_state_dict(agent=agent)
        del agent
