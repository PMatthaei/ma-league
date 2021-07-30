from custom_logging.logger import MainLogger


class EnvStepper:
    def __init__(self, args, logger: MainLogger):
        """
        Steppers build the environment and issue steps. The rely on one or more multi-agent controller to infer actions.
        :param args: args passed from main
        :param logger: logger
        """
        self.args = args
        self.logger: MainLogger = logger
        self.batch_size = None
        self.t_env = None
        self.is_initalized = False
        self.log_start_t = None

    @property
    def log_t(self):
        return self.log_start_t + self.t_env

    def run(self, test_mode=False):
        raise NotImplementedError()

    def initialize(self, scheme, groups, preprocess, home_mac):
        raise NotImplementedError()

    def close_env(self):
        raise NotImplementedError()

    def save_replay(self):
        raise NotImplementedError()

    def get_env_info(self):
        raise NotImplementedError()
