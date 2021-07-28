from maenv.core import RoleTypes, UnitAttackTypes
from torch.multiprocessing import Process
from argparse import Namespace
from typing import Dict

from custom_logging.platforms import CustomConsoleLogger
from league import SimpleLeague
from league.components import PayoffEntry
from torch.multiprocessing import Barrier, current_process

from league.rolebased.league import League
from league.processes.agent_pool_instance import AgentPoolInstance
from league.components.team_composer import TeamComposer
from league.processes import REGISTRY as experiment_REGISTRY
from league.components.matchmaker import REGISTRY as matchmaking_REGISTRY, Matchmaker
from utils.config_builder import ConfigBuilder


class CentralWorker(Process):
    def __init__(self, params, args: Namespace, log_dir: str, src_dir: str):
        super().__init__()
        self._params = params
        self._args = args
        self._log_dir = log_dir
        self._src_dir = src_dir  # Load league base config

        self.config_builder = ConfigBuilder(
            worker_args=self._args,
            src_dir=self._src_dir,
            log_dir=self._log_dir,
            params=self._params
        )

    def run(self) -> None:
        central_logger = CustomConsoleLogger("central-worker")
        central_worker_id = current_process()

        central_logger.info(f'Central working running in process {central_worker_id}')
        central_logger.info(f'League Parameters: {vars(self._args)}')
        central_logger.info(f'Logging league instances into directory: {self._log_dir}')

        #
        # Build league teams
        #
        composer = TeamComposer(team_size=self._args.team_size, characteristics=[RoleTypes, UnitAttackTypes])
        uid = composer.get_unique_uid(role_type=self._args.role, attack_type=self._args.attack)
        # train, test = train_test_split(np.array(composer.teams)) # TODO!
        # Sample random teams containing uid
        teams = composer.sample(k=self._args.league_size, contains=uid, unique=self._args.unique)
        # Sort ranged healer first in all teams for later consistency
        teams = composer.sort_team_units(teams, uid=uid)
        n_teams = len(teams)

        #
        # Synchronization
        #
        sync_barrier = Barrier(parties=n_teams)

        #
        # Shared objects
        #
        n_entries = len(PayoffEntry)
        from torch import zeros
        payoff = zeros((n_teams, n_teams, n_entries)).share_memory_()

        #
        # Components
        #
        agent_pool = AgentPoolInstance(
            sync_barrier=sync_barrier
        )
        matchmaker = None
        procs = []
        if self._args.experiment == "matchmaking" or self._args.experiment == "ensemble":
            matchmaker: Matchmaker = matchmaking_REGISTRY[self._args.matchmaking](
                communication=agent_pool.register(),
                payoff=payoff,
                teams=teams
            )
            #
            # Start experiment instances
            #
             # All running processes representing an agent playing in the league
            experiment = experiment_REGISTRY[self._args.experiment]
            for idx, team in enumerate(teams):
                proc = experiment(
                    idx=idx,
                    experiment_config=self.config_builder.build(idx),
                    home_team=team,  # TODO replace with teams[idx] from matchmaker
                    matchmaker=matchmaker,
                    communication=agent_pool.register(),
                    sync_barrier=sync_barrier
                )
                procs.append(proc)

            agent_pool.start()

            [p.start() for p in procs]

        elif self._args.experiment == "rolebased":
            league: SimpleLeague = SimpleLeague(
                teams=teams,
                payoff=payoff,
                agent_pool=agent_pool,
                sync=sync_barrier,
                config_builder=self.config_builder
            )
            procs = league.start()
        else:
            raise NotImplementedError("Experiment not supported.")

        #
        # Wait for experiments to finish
        #
        [p.join() for p in procs]
        matchmaker.disconnect()
        agent_pool.join()

        #
        # Print Payoff tensor
        #
        self._print_payoff(payoff, teams)

    def _print_payoff(self, payoff, teams):
        team_allocation: Dict[int, int] = dict(
            {team.id_: idx for idx, team in enumerate(teams)})  # Mapping team id -> instance id
        for team in teams:
            tid = team.id_
            index = team_allocation[tid]
            print(f'Stats for {team} from instance {index}')
            for entry in PayoffEntry:
                print(f"{entry.name.capitalize()} {payoff[index, :, entry]}")
