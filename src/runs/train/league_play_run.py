from typing import Dict

from controllers import EnsembleMAC
from modules.agents.agent_network import AgentNetwork

from runs.train.self_play_run import SelfPlayRun


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

    def build_ensemble_mac(self, agent: AgentNetwork=None, ensemble: Dict[int, AgentNetwork] = None, target: str = "away"):
        """
        Create a Multi-Agent Controller only used for inference of fixed and pre-trained policies.
        The policy can either be supplied as an single agents of a ensemble of agents.
        :param agent:
        :param ensemble: 
        :return: 
        """
        if target == "away": # WARN! Assume home buffer scheme == away scheme
            self.away_mac = EnsembleMAC(self.home_buffer.scheme, self.groups, self.args)
            self.away_mac.load_state(agent=agent, ensemble=ensemble)
        elif target == "home":
            self.home_mac = EnsembleMAC(self.home_buffer.scheme, self.groups, self.args)
            self.home_mac.load_state(agent=agent, ensemble=ensemble)

    def _test(self, n_test_runs):
        self.last_test_T = self.stepper.t_env
        pass  # Skip tests in league to save computing time
