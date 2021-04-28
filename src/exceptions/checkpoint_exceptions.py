class NoLearnersProvided(Exception):
    def __init__(self):
        super().__init__("The provided list of learners is empty. Make sure to register learners before calling a save.")
