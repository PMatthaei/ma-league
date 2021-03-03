class HiddenStateNotInitialized(Exception):
    def __init__(self):
        super().__init__("Please run init_hidden() to initialize the hidden state before running forward pass.)")
