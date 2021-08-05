from league.processes import MultiAgentExperimentInstance
from runs.evaluation.replay_eval_run import ReplayGenerationRun


class ReplayInstance(MultiAgentExperimentInstance):

    def __init__(self, **kwargs):
        """
        A simple Multi-Agent Experiment running as its own process for a given time.
        :param kwargs:
        """
        super(ReplayInstance, self).__init__(**kwargs)

    def _run_experiment(self) -> None:
        self._experiment = ReplayGenerationRun(args=self._args, logger=self._logger)
        self._experiment.start()
        self._logger.info(f"Training in process finished: {self._proc_id}")