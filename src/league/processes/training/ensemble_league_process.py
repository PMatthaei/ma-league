import copy

from torch.multiprocessing import Barrier, Queue, current_process

from typing import Dict, Tuple

from league.components.agent_pool import AgentPool
from league.components.matchmaking import Matchmaking
from league.processes.experiment_process import ExperimentProcess
from league.processes.training.utils import extract_result
from league.utils.commands import CloseLeagueProcessCommand, PayoffUpdateCommand
from league.utils.team_composer import Team
from modules.agents import AgentNetwork
from runs.train.ma_experiment import MultiAgentExperiment


class EnsembleLeagueProcess(ExperimentProcess):

    def __init__(self,
                 idx,
                 params,
                 configs_dir,
                 log_dir,
                 agent_pool: AgentPool,
                 matchmaking: Matchmaking,
                 home_team: Team,
                 communication: Tuple[Queue, Queue],
                 sync_barrier: Barrier):
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
        super(EnsembleLeagueProcess, self).__init__(idx, params, configs_dir, log_dir)

        self._home_team: Team = home_team
        self._away_team: Team = None
        self._agent_pool: AgentPool = agent_pool  # Process private copy of the agent pool
        self._matchmaking: Matchmaking = matchmaking

        self._in_queue, self._out_queue = communication  # In- and Outgoing Communication
        self._sync_barrier = sync_barrier  # Use to sync with other processes

        self.terminated: bool = False

        self._experiment: MultiAgentExperiment = None

        self._ensemble = None

    @property
    def home_agent(self) -> AgentNetwork:
        return self._experiment.home_mac.agent

    @property
    def ensemble_agent(self) -> AgentNetwork:
        return self._experiment.home_mac.ensemble[0]

    @property
    def shared_agent(self) -> Tuple[Team, AgentNetwork]:
        return self._home_team, self._agent_pool[self._home_team]

    def _run_experiment(self) -> None:
        self.proc_id = current_process()

        # Initial play to train policy of the team against AI against mirrored team -> Performed for each team
        self._logger.info(f"Start training in process: {self.proc_id} with {self._home_team}")
        self._configure_play(home=self._home_team)
        self._experiment = MultiAgentExperiment(args=self._args, logger=self._logger)
        self._logger.info(f"Train against AI in process: {self.proc_id}")
        self._experiment.start(play_time_seconds=self._args.league_play_time_mins * 60)
        self._logger.info(f"Share agent from process: {self.proc_id}")
        self._share_agent(agent=self.home_agent)  # make agent accessible to other instances
        self._native_agent: AgentNetwork = copy.deepcopy(self.home_agent)  # save an instance of the original agent

        # Fetch agents from another teams training instance
        foreign: Tuple[Team, AgentNetwork] = self._matchmaking.get_match(self._home_team)
        while foreign is not None:
            foreign_team, foreign_agent = foreign
            self._logger.info(f"Matched foreign team {foreign_team.id_} in process: {self.proc_id}")

            self._logger.info(f"Build foreign team play in process: {self.proc_id}")
            self._configure_play(home=foreign_team)  # Set the foreign team constellation as home team
            self._experiment = MultiAgentExperiment(args=self._args, logger=self._logger)
            # Train the native agent in a different team setup as foreign agent
            self._experiment.build_ensemble_mac(native=foreign_agent, foreign_agent=self._native_agent)
            # Evaluate how good the mixed team performs
            self._logger.info(f"Evaluate ensemble in process: {self.proc_id}")
            self._experiment.evaluate_sequential(test_n_episode=self._args.n_league_evaluation_episodes)

            # Train only new foreign agent with the team performing as before
            self._args.freeze_native = True  # Freeze weights of native agent
            self._logger.info(f"Train ensemble in process: {self.proc_id}")
            self._configure_play(home=foreign_team)  # Set the foreign team constellation as home team
            self._experiment = MultiAgentExperiment(args=self._args, logger=self._logger)
            self._experiment.build_ensemble_mac(native=self._native_agent, foreign_agent=foreign_agent)
            self._experiment.start(play_time_seconds=self._args.league_play_time_mins * 60)

            # Share agent after training to make its current state accessible to other processes
            self._logger.info(f"Share trained ensemble in process: {self.proc_id}")
            self._share_agent(agent=self.ensemble_agent)
            # Select next agent to train
            foreign: Tuple[Team, AgentNetwork] = self._matchmaking.get_match(self._home_team)
            if foreign is not None:
                self._logger.info(f"Selected team {foreign[0].id_} for next iteration in process: {self.proc_id}")

        self._logger.info(f"Sampled all adversary teams in process: {self.proc_id}")
        self._experiment.save_models()

        self._request_close()

    def _configure_play(self, home: Team, away: Team = None, ai_opponent=True):
        # In case this process needs to save models -> modify token
        self._args.env_args['match_build_plan'][0]['units'] = home.units  # mirror if no away units passed
        self._args.env_args['match_build_plan'][1]['units'] = home.units if away is None else away.units
        self._args.env_args['match_build_plan'][0]['is_scripted'] = not ai_opponent
        self._args.env_args['match_build_plan'][1]['is_scripted'] = ai_opponent

    def _get_shared_agent(self, team: Team):
        return self._agent_pool[team]

    def _share_agent(self, agent: AgentNetwork, team: Team = None):
        self._agent_pool[self._home_team if team is None else team] = agent
        # Wait until every process finished to share the agent to ensure every agent is up-to-date before next match
        self._sync_barrier.wait()

    def _provide_result(self, env_info: Dict):
        """
        Send the result of an episode the the central coordinator for processing.
        :param env_info:
        :return:
        """
        result = extract_result(env_info, self._experiment.stepper.policy_team_id)
        data = ((self._home_team.id_, self._away_team.id_), result)
        cmd = PayoffUpdateCommand(origin=self._home_team.id_, data=data)
        self._in_queue.put(cmd)

    def _request_close(self):
        cmd = CloseLeagueProcessCommand(origin=self._home_team.id_)
        self._in_queue.put(cmd)
