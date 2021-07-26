import copy
import time

from typing import Tuple, OrderedDict

from league.processes.league_experiment_process import LeagueExperimentProcess
from league.utils.team_composer import Team
from runs.train.ensemble_experiment import EnsembleExperiment
from runs.train.ma_experiment import MultiAgentExperiment


class EnsembleLeagueProcess(LeagueExperimentProcess):

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
        super(EnsembleLeagueProcess, self).__init__(**kwargs)

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
        self._experiment.start(play_time_seconds=self._args.league_play_time_mins * 60)
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
            home_team, agent_state = self._get_agent_params

            #
            # Fetch agents from another teams training instance
            #
            foreign: Tuple[Team, OrderedDict] = self._matchmaking.get_match(self._home_team)
            if foreign is None:
                self._logger.info(f"No match found. Ending process: {self._proc_id} with {self._home_team}")
                break
            self._away_team, foreign_agent_state = foreign  # Note: Set away team for payoff tensor
            self._logger.info(f"Matched foreign team {self._away_team.id_} in process: {self._proc_id}")

            #
            # Evaluate how the agent performs in an ensemble with the foreign agent (and its team constellation)
            #
            self._logger.info(f"Build foreign team play in process: {self._proc_id}")
            self._configure_experiment(home=self._away_team, ai=True)  # Set the foreign team constellation as home team
            self._experiment = EnsembleExperiment(args=self._args, logger=self._logger, on_episode_end=self._send_result)
            self._experiment.load_ensemble(native=foreign_agent_state, foreign_agent=agent_state)
            self._logger.info(f"Evaluate ensemble in process: {self._proc_id}")
            self._experiment.evaluate_sequential(test_n_episode=self._args.n_league_evaluation_episodes)

            #
            # Train the native agent in an ensemble with the foreign agent (and its team constellation)
            #
            self._logger.info(f"Train ensemble in process: {self._proc_id}")
            self._configure_experiment(home=self._away_team, ai=True)  # Set the foreign team constellation as home team
            self._experiment = EnsembleExperiment(args=self._args, logger=self._logger, on_episode_end=self._send_result)
            self._experiment.load_ensemble(native=foreign_agent_state, foreign_agent=agent_state)
            self._experiment.start(play_time_seconds=self._args.league_play_time_mins * 60)

            #
            # Share agent after training to make its current state accessible to other processes
            #
            self._logger.info(f"Share trained ensemble in process: {self._proc_id}")
            self._share_agent_params(agent=self.ensemble_agent_state)

            end_time = time.time()

        self._logger.info(f"Training in process finished: {self._proc_id}")

        self._request_close()
