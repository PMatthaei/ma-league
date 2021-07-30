from league.processes.interfaces import ExperimentInstance
from runs.train.ma_experiment import MultiAgentExperiment


class MultiAgentExperimentInstance(ExperimentInstance):

    def __init__(self, **kwargs):
        """
        A simple Multi-Agent Experiment running as its own process for a given time.
        :param kwargs:
        """
        super(MultiAgentExperimentInstance, self).__init__(**kwargs)

    def _run_experiment(self) -> None:
        self._experiment = MultiAgentExperiment(args=self._args, logger=self._logger)
        self._experiment.start(play_time_seconds=self._args.play_time_mins * 60)
        self._logger.info(f"Training in process finished: {self._proc_id}")

