from modules.agents.agent import Agent

from controllers import REGISTRY as mac_REGISTRY
from runs.self_play_run import SelfPlayRun


class LeaguePlayRun(SelfPlayRun):

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
        # WARN: Assuming the away agent uses the same buffer scheme!!
        self.away_mac = mac_REGISTRY[self.args.mac](self.home_buffer.scheme, self.groups, self.args)

    def set_away_agent(self, away: Agent):
        self.away_mac.agent = away

    def _test(self, n_test_runs):
        self.last_test_T = self.stepper.t_env
        pass  # Skip tests in league
