import time

from typing import Tuple, OrderedDict

from league.processes.interfaces.league_experiment_process import LeagueExperimentInstance
from league.components.team_composer import Team
from runs.train.ensemble_experiment import EnsembleExperiment
from runs.train.ma_experiment import MultiAgentExperiment


class EnsembleLeagueInstance(LeagueExperimentInstance):

    def __init__(self, **kwargs):
        """
        The process is running a single League-Play and handles communication with the central components.
        League-Play is a form of NormalPlay where the opponent can be swapped out from a pool of agents (=league).
        The opponent is fixed and is therefore not learning to prevent non-stationary environment.
        Opponents are sampled via Self-Play Sampling such as FSP, PFSP or SP.

        Opponent sampling is decided via a matchmaking component.

        :param agent_pool:
        :param matchmaking:
        :param home_team:
        :param communication:
        :param args:
        :param logger:
        :param sync_barrier:
        """
        super(EnsembleLeagueInstance, self).__init__(**kwargs)

        self._ensemble = None

    @property
    def ensemble_agent_state(self) -> OrderedDict:
        return self._experiment.home_mac.ensemble[0].state_dict()  # TODO: how get various ensemble not just first

    def _run_experiment(self) -> None:
        #
        # Initial play to train policy of the team against AI against mirrored team -> Performed for each team
        #
        self._logger.info(f"Start training in process: {self._proc_id} with {self._home_team}")
        self._configure_experiment(home=self._home_team, ai=True)
        self._experiment = MultiAgentExperiment(args=self._args, logger=self._logger)
        self._logger.info(f"Train against AI in process: {self._proc_id}")
        self._experiment.start(play_time_seconds=self._args.play_time_mins * 60)
        self._logger.info(f"Share agent from process: {self._proc_id}")
        self._share_agent_params(agent=self.home_agent_state)  # make agent accessible to other instances

        start_time = time.time()
        end_time = time.time()

        #
        # Training loop
        #
        while end_time - start_time <= self._args.league_runtime_hours * 60 * 60:
            #
            # Retrieve current version of the agent from the pool
            #
            home_team, agent_state = self._get_agent_params()

            #
            # Fetch agents from another teams training instance
            #
            adversary = (self._adversary_idx, self._adversary_team, foreign_params) = self._matchmaker.get_match(self._home_team) or (None, None, None)
            if all(adversary):
                self._logger.info(f"No match found: {adversary}. Ending {str(self)}")
                break

            self._logger.info(f"Matched foreign team {self._adversary_team.id_} in {str(self)}")

            #
            # Evaluate how the agent performs in an ensemble with the foreign agent (and its team constellation)
            #
            self._logger.info(f"Build foreign team play in {str(self)}")
            self._configure_experiment(home=self._adversary_team, ai=True)  # Set the foreign team constellation as home team
            self._logger.info(f"Build ensemble experiment in {str(self)}")
            self._experiment = EnsembleExperiment(args=self._args, logger=self._logger, on_episode_end=self._update_payoff)
            self._logger.info(f"Load ensemble agents {str(self)}")
            self._experiment.load_ensemble(native=foreign_params, foreign_agent=agent_state)
            self._logger.info(f"Evaluate ensemble in {str(self)}")
            self._experiment.evaluate_sequential(test_n_episode=self._args.n_league_evaluation_episodes)
            #
            # Train the native agent in an ensemble with the foreign agent (and its team constellation)
            #
            self._logger.info(f"Train ensemble agents {str(self)}")
            self._experiment.start(play_time_seconds=self._args.play_time_mins * 60)

            #
            # Share agent after training to make its current state accessible to other processes
            #
            self._logger.info(f"Share trained ensemble in {str(self)}")
            self._share_agent_params(agent=self.ensemble_agent_state)

            end_time = time.time()

        self._logger.info(f"Training in {str(self)} finished")

        self._request_close()
