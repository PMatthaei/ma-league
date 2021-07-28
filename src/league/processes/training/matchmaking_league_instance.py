import time
from torch.multiprocessing import Barrier, Queue

from typing import Tuple

from league.components.matchmaker import Matchmaker
from league.processes.interfaces.league_experiment_process import LeagueExperimentInstance
from league.components.team_composer import Team
from runs.train.league_experiment import LeagueExperiment
from runs.train.ma_experiment import MultiAgentExperiment


class MatchmakingLeagueInstance(LeagueExperimentInstance):
    def __init__(self, matchmaking: Matchmaker, home_team: Team, communication: Tuple[int, Tuple[Queue, Queue]],
                 sync_barrier: Barrier, **kwargs):

        super().__init__(matchmaking, home_team, communication, sync_barrier, **kwargs)

    def _run_experiment(self):
        self._logger.info(f"Start pre-training with AI in {str(self)}")

        # Initial play to train policy of the team against mirrored AI
        self._configure_experiment(home=self._home_team, ai=True)
        self._experiment = MultiAgentExperiment(args=self._args, logger=self._logger)
        self._experiment.start(play_time_seconds=self._args.play_time_mins * 60)
        self._share_agent_params(self.home_agent_state)

        start_time = time.time()
        end_time = time.time()

        # Run real league play in self-play against pre-trained but fixed multi-agent policies
        self._logger.info(f"Start training in {str(self)} with {self._home_team} for {self._args.league_runtime_hours} hours")
        while end_time - start_time <= self._args.league_runtime_hours * 60 * 60:
            adversary = [self._adversary_idx, self._adversary_team, adversary_params] = self._matchmaker.get_match(self._home_team) or (None, None, None)
            if adversary.count(None) > 0: # Test if all necessary data set
                self._logger.info(f"No match found. Ending process: {self._proc_id} with {self._home_team}")
                break

            self._logger.info(f"Matched away team {self._adversary_team.id_} in {str(self)}")

            self._configure_experiment(home=self._home_team, away=self._adversary_team, ai=False)
            self._logger.info(f"Prepared experiment in {str(self)}")
            self._experiment = LeagueExperiment(args=self._args, logger=self._logger, on_episode_end=self._update_payoff)
            self._logger.info(f"Loading adversary team {self._adversary_team.id_} in {str(self)}")
            self._experiment.load_adversary(agent=adversary_params)
            self._logger.info(f"Starting adversary team {self._adversary_team.id_} in {str(self)}")
            self._experiment.start(play_time_seconds=self._args.play_time_mins * 60)

            # Share agent after training to make its current state accessible to other processes
            self._share_agent_params(agent=self.home_agent_state)

            end_time = time.time()

        self._request_close()
