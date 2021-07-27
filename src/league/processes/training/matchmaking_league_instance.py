import time
from torch.multiprocessing import Barrier, Queue

from typing import Tuple

from league.components.matchmaking import Matchmaking
from league.processes.league_experiment_process import LeagueExperimentInstance
from league.utils.team_composer import Team
from runs.train.league_experiment import LeagueExperiment
from runs.train.ma_experiment import MultiAgentExperiment


class MatchmakingLeagueInstance(LeagueExperimentInstance):
    def __init__(self, matchmaking: Matchmaking, home_team: Team, communication: Tuple[int, Tuple[Queue, Queue]],
                 sync_barrier: Barrier, **kwargs):

        super().__init__(matchmaking, home_team, communication, sync_barrier, **kwargs)

    def _run_experiment(self):
        self._logger.info(f"Start pre-training with AI in process: {self._proc_id} with {self._home_team}")

        # Initial play to train policy of the team against mirrored AI
        self._configure_experiment(home=self._home_team, ai=True)
        self._experiment = MultiAgentExperiment(args=self._args, logger=self._logger)
        self._experiment.start(play_time_seconds=self._args.play_time_mins * 60)
        self._share_agent_params(self.home_agent_state)

        start_time = time.time()
        end_time = time.time()

        # Run real league play in self-play against pre-trained but fixed multi-agent policies
        self._logger.info(f"Start training in process: {self._proc_id} with {self._home_team} for {self._args.league_runtime_hours} hours")
        while end_time - start_time <= self._args.league_runtime_hours * 60 * 60:
            match = self._matchmaking.get_match(self._home_team)
            if match is None:
                self._logger.info(f"No match found. Ending process: {self._proc_id} with {self._home_team}")
                break
            self._away_team, away_agent = match
            self._logger.info(f"Matched away team {self._away_team.id_} in process: {self._proc_id}  with {self._home_team}")

            self._configure_experiment(home=self._home_team, away=self._away_team, ai=False)
            self._experiment = LeagueExperiment(args=self._args, logger=self._logger, on_episode_end=self._send_result)
            self._experiment.load_adversary(agent=away_agent)
            self._experiment.start(play_time_seconds=self._args.play_time_mins * 60)

            end_time = time.time()

            # Share agent after training to make its current state accessible to other processes
            self._share_agent_params(agent=self.home_agent_state)

        self._request_close()
