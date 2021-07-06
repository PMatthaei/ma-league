from runs.experiment_run import ExperimentRun


class ReplayGenerationRun(ExperimentRun):

    def __init__(self, args, logger):
        """
        Load a saved model into the Multi-Agent Controller and perform inference for a given amount of time.
        Episodes run within this time are captured as video for later replay and visual understanding of the loaded
        model/policy.
        :param args:
        :param logger:
        """
        super().__init__(args, logger)
