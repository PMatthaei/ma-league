from maenv.core import RoleTypes, UnitAttackTypes
from torch.multiprocessing import Process
from argparse import Namespace
from typing import Dict


from custom_logging.platforms import CustomConsoleLogger
from league.components import PayoffEntry
from torch.multiprocessing import Barrier, Manager, current_process

from league.processes.command_handler import CommandHandler
from league.utils.team_composer import TeamComposer
from league.processes import REGISTRY as experiment_REGISTRY
from league.components.matchmaking import REGISTRY as matchmaking_REGISTRY
from utils.config_builder import ConfigBuilder
from utils.main_utils import load_config_yaml


class CentralWorker(Process):
    def __init__(self, params, args: Namespace, log_dir: str, src_dir: str):
        super().__init__()
        self._params = params
        self._args = args
        self._log_dir = log_dir
        self._src_dir = src_dir # Load league base config

    def run(self) -> None:
        central_logger = CustomConsoleLogger("central-worker")
        central_worker_id = current_process()

        central_logger.info(f'Central working running in process {central_worker_id}')
        central_logger.info(f'League Parameters: {vars(self._args)}')
        central_logger.info(f'Logging league instances into directory: {self._log_dir}')

        config_builder = ConfigBuilder(src_dir=self._src_dir, log_dir=self._log_dir, params=self._params)

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
        manager = Manager()

        n_entries = len(PayoffEntry)
        from torch import zeros
        payoff = zeros((n_teams, n_teams, n_entries)).share_memory_()

        team_allocation: Dict[int, int] = manager.dict()  # Mapping team -> training instance it is running on

        #
        # Communication Infrastructure
        #
        # Handle command communication within the league
        handler = CommandHandler(
            allocation=team_allocation,
            payoff=payoff,
            sync_barrier=sync_barrier
        )

        #
        # Components
        #
        comm = handler.register()
        matchmaking = matchmaking_REGISTRY[self._args.matchmaking](
            communication=comm,
            payoff=payoff,
            allocation=team_allocation,
            teams=teams
        )

        #
        # Start experiment instances
        #
        procs = []  # All running processes representing an agent playing in the league
        experiment = experiment_REGISTRY[self._args.experiment]
        for idx, team in enumerate(teams):
            comm = handler.register()
            team_allocation[team.id_] = idx
            proc = experiment(
                idx=idx,
                experiment_config=config_builder.build(idx),
                #log_dir=self._log_dir,
                #src_dir=self._src_dir,
                #params=self._build_instance_params(self._params),  # Delegate params to experiment instance
                home_team=team,
                matchmaking=matchmaking,
                communication=comm,
                sync_barrier=sync_barrier
            )
            procs.append(proc)

        handler.start()

        [r.start() for r in procs]

        #
        # Wait for experiments to finish
        #
        [r.join() for r in procs]
        matchmaking.disconnect()
        handler.join()

        #
        # Print Payoff tensor
        #
        self._print_payoff(payoff, team_allocation, teams)

    def _print_payoff(self, payoff, team_allocation, teams):
        for team in teams:
            tid = team.id_
            index = team_allocation[tid]
            print(f'Stats for {team} from instance {index}')
            for entry in PayoffEntry:
                print(f"{entry.name.capitalize()} {payoff[index, :, entry]}")

    # def _build_instance_params(self, params):
    #     """
    #     Build the params for the instance with idx.
    #     :param params: Original params
    #     :return:
    #     """
    #
    #     params += [f'--{k}={v}' for k, v in self._league_config.items()]
    #     return params
