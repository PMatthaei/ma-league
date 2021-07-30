import time
from torch.multiprocessing import Barrier
from torch.multiprocessing.queue import Queue

from typing import Tuple

from league.processes.interfaces import LeagueExperimentInstance
from league.components.team_composer import Team
from league.rolebased.players import Player
from runs.train.league_experiment import LeagueExperiment
from runs.train.ma_experiment import MultiAgentExperiment


class RoleBasedLeagueInstance(LeagueExperimentInstance):
    def __init__(self, matchmaker: Player, home_team: Team, communication: Tuple[int, Tuple[Queue, Queue]],
                 sync_barrier: Barrier, **kwargs):

        super().__init__(matchmaker, home_team, communication, sync_barrier, **kwargs)
        self._matchmaker = matchmaker

    def _run_experiment(self):
        self._logger.info(f"Start pre-training with AI in {str(self)} ")

        # Initial play to train policy of the team against mirrored AI
        self._configure_experiment(home=self._home_team, ai=True)
        self._experiment = MultiAgentExperiment(args=self._args, logger=self._logger)
        self._experiment.start(play_time_seconds=self._args.play_time_mins * 60)
        self._share_agent_params(self.home_agent_state)

        # Progress to save initial checkpoint of agents after all runs performed setup
        if self._matchmaker.is_main_player():  # TODO: Allow for different kinds of initial historical players
            self._request_checkpoint()  # MainPlayers are initially added as historical players

        start_time = time.time()
        end_time = time.time()

        while end_time - start_time <= self._args.league_runtime_hours * 60 * 60:

            adversary = [self._adversary_idx, self._adversary_team, adversary_params] = self._matchmaker.get_match(self._home_team) or (None, None, None)
            if adversary.count(None) > 0:  # Test if all necessary data set
                self._logger.info(f"No match found. Ending {str(self)}")
                break

            self._configure_experiment(home=self._home_team, away=self._adversary_team, ai=False)
            self._experiment = LeagueExperiment(args=self._args, logger=self._logger)
            self._experiment.load_adversary(agent=adversary_params)
            self._experiment.start(play_time_seconds=self._args.play_time_mins * 60)

            # Share agent after training
            self._share_agent_params(agent=self.home_agent_state)

            end_time = time.time()

        self._request_close()

    def _request_checkpoint(self):
        pass

    def __str__(self):
        return f"{self.__class__.__name__}: " + str(self.idx) + " with " + str(self._home_team) + f" as {self._matchmaker.__class__.__name__}"