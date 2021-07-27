import time
from logging import warning
from torch.multiprocessing import Barrier
from torch.multiprocessing.queue import Queue

from typing import Tuple

from league.components import Matchmaker
from league.processes.interfaces.league_experiment_process import LeagueExperimentInstance
from league.roles.alphastar.main_player import MainPlayer
from league.components.team_composer import Team
from runs.train.league_experiment import LeagueExperiment


class RolebasedLeagueInstance(LeagueExperimentInstance):
    def __init__(self, matchmaking: Matchmaker, home_team: Team, communication: Tuple[int, Tuple[Queue, Queue]],
                 sync_barrier: Barrier, **kwargs):

        super().__init__(matchmaking, home_team, communication, sync_barrier, **kwargs)

    def _run_experiment(self):
        self._logger.info(f"Start pre-training with AI in process: {self._proc_id} with {self._home_team}")

        # Initial play to train policy of the team against mirrored AI
        self._configure_experiment(home=self._home_team, ai=True)
        self._experiment = LeagueExperiment(args=self._args, logger=self._logger)
        self._experiment.start(play_time_seconds=self._args.play_time_mins * 60)
        self._share_agent_params(self.home_agent_state)

        # Progress to save initial checkpoint of agents after all runs performed setup
        if isinstance(self._home, MainPlayer):  # TODO: Allow for different kinds of initial historical players
            self._request_checkpoint()  # MainPlayers are initially added as historical players

        start_time = time.time()
        end_time = time.time()

        while end_time - start_time <= self._args.league_runtime_hours * 60 * 60:

            self._away, flag = self._home.get_match()
            match = self._get_agent_params(self._away)
            if match is None:
                warning("No Opponent was found.")
                continue

            self._away_team, agent = match

            self._experiment.load_adversary(agent=match)

            self._logger.info(str(self))

            # Start training against new opponent and integrate the team of the away player
            self._register_team(self._away, rebuild=True)
            self._experiment.start(play_time_seconds=self._args.play_time_mins * 60)

            # Share agent after training
            self._share_agent_params()

            end_time = time.time()

            # Wait until every process finished to sync for printing the payoff table
            self._sync_barrier.wait()

        self._request_close()

    def _request_checkpoint(self):
        pass