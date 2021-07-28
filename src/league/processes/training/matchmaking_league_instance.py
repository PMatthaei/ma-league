import time
from typing import Tuple

from torch.multiprocessing import Barrier, Queue

from league.components.matchmaker import Matchmaker
from league.components.team_composer import Team
from league.processes.interfaces.league_experiment_process import LeagueExperimentInstance
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
        hrs = self._args.league_runtime_hours
        self._logger.info(f"Start training in {str(self)} for {hrs:.2f} hours - {hrs * 60} mins")

        iters = 1
        while end_time - start_time <= hrs * 60 * 60:
            #
            # Retrieve current version of the agent from the pool
            #
            _, agent_state = self._get_agent_params()

            self._logger.info(f"Start iteration {iters} in {str(self)}")

            adversary = [self._adversary_idx, self._adversary_team, adversary_params] = self._matchmaker.get_match(self._home_team) or (None, None, None)
            if adversary.count(None) > 0:  # Test if all necessary data set
                self._logger.info(f"No match found. Ending {str(self)}")
                break

            self._logger.info(f"Matched away team {self._adversary_team.id_} in {str(self)}")

            self._configure_experiment(home=self._home_team, away=self._adversary_team, ai=False)
            self._logger.info(f"Prepared experiment in {str(self)}")
            self._experiment = LeagueExperiment(args=self._args, logger=self._logger,
                                                on_episode_end=self._update_payoff)
            self._logger.info(f"Loading adversary team {self._adversary_team.id_} in {str(self)}")
            self._experiment.load_home_agent(agent=agent_state)
            self._experiment.load_adversary(agent=adversary_params)
            self._logger.info(f"Starting adversary team {self._adversary_team.id_} in {str(self)}")
            self._experiment.start(play_time_seconds=self._args.play_time_mins * 60)

            # Share agent after training to make its current state accessible to other processes
            self._share_agent_params(agent=self.home_agent_state)
            iters += 1
            end_time = time.time()

        self._request_close()
