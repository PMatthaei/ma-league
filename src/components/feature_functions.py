class FeatureFunction:

    def __init__(self):
        pass

    def __call__(self, state, actions, next_state):
        raise NotImplementedError()


class TeamTaskSuccessorFeatures(FeatureFunction):
    """
    Extract feature to represent a team task.
    """
    def __call__(self, state, actions, next_state):
        return # TODO