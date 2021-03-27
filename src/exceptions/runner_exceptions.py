class RunnerMACNotInitialized(Exception):
    def __init__(self):
        super().__init__("Multi-Agent Controller not initialized."
                         "Please run initialize() with it`s corresponding arguments to prepare the runner.")
