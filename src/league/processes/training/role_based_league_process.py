import time
from logging import warning
from torch.multiprocessing import Barrier, Queue

from typing import Union, Tuple, List

from league.components import Matchmaking
from league.processes.league_experiment_process import LeagueExperimentProcess
from league.roles.alphastar.main_player import MainPlayer
from league.roles.players import Player
from league.utils.team_composer import Team
from runs.train.league_experiment import LeagueExperiment


class RolebasedLeagueProcess(LeagueExperimentProcess):
    def __init__(self, players: List[Player], player_id: int, sync_barrier: Barrier,
                 matchmaking: Matchmaking,
                 home_team: Team, communication: Tuple[Queue, Queue], **kwargs):
        """
        The process is running a single League-Play and handles communication with the central component.
        League-Play is a form of NormalPlay where the opponent can be swapped out from a pool of agents (=league).
        The opponent is fixed and is therefore not learning to prevent non-stationary environment.
        Opponents are sampled via Self-Play Sampling such as FSP, PFSP or SP.

        Opponent sampling is decided by the current player. Each player has a different strategy for sampling/searching
        his opponent. (AlphaStar)

        :param players:
        :param player_id:
        :param queue:
        :param args:
        :param logger:
        :param sync_barrier: Barrier to synchronize all league processes
        """
        super().__init__(matchmaking, home_team, communication, sync_barrier, **kwargs)

    def _run_experiment(self):
        # Share initial agent
        self._share_agent_params()

        # Wait at barrier until every league process performed the sharing step before the next step
        self._sync_barrier.wait()

        # Progress to save initial checkpoint of agents after all runs performed setup
        if isinstance(self._home, MainPlayer):  # TODO: Allow for different kinds of initial historical players
            self._request_checkpoint()  # MainPlayers are initially added as historical players

        start_time = time.time()
        end_time = time.time()

        while end_time - start_time <= self._args.league_runtime_hours * 60 * 60:

            self._away, flag = self._home.get_match()
            away_agent = self._get_agent_params(self._away)
            if away_agent is None:
                warning("No Opponent was found.")
                continue
            self._experiment.load_adversary(agent=away_agent)

            self._logger.info(str(self))

            # Start training against new opponent and integrate the team of the away player
            self._register_team(self._away, rebuild=True)
            self._experiment.start(play_time_seconds=self._args.league_play_time_mins * 60)

            # Share agent after training
            self._share_agent_params()

            end_time = time.time()

            # Wait until every process finished to sync for printing the payoff table
            self._sync_barrier.wait()

        self._request_close()
