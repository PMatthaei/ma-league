class EnvStepper:
    def __init__(self):
        self.batch_size = None
        self.t_env = None
        self.is_initalized = False

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