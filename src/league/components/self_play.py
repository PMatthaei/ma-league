import numpy as np


class OpponentSampling:
    def __init__(self):
        # TODO: Set and update opponent pool so that it does not have to be part of the sample method arguments
        pass

    def sample(self, opponents):
        """
        Implement how opponents should be sampled. Additional data can be supplied via optional arguments.
        :param opponents:
        :return:
        """
        raise NotImplementedError()


class SPSampling(OpponentSampling):

    def __init__(self, opponent_id):
        super().__init__()
        self.opponent_id = opponent_id

    def sample(self, opponents):
        return opponents[self.opponent_id]


class FSPSampling(OpponentSampling):

    def __init__(self):
        """
        FSP: Uniform sampling from a opponent policy distribution
        """
        super().__init__()

    def sample(self, opponents):
        p = np.random.uniform(low=0.0, high=1.0, size=None)
        return np.random.choice(opponents, p=p)


class PFSPSampling(OpponentSampling):

    def __init__(self):
        """
        PFSP: Non-Uniform sampling from a opponent policy distribution via prioritization measure
        :param weighting:
        """
        super().__init__()

    def sample(self, opponents, prio_measure=None, weighting="linear"):
        if prio_measure is None:
            raise Exception("Please serve up-to-date prioritization measure.")
        weightings = {
            "variance": lambda x: x * (1 - x),
            "linear": lambda x: 1 - x,
            "linear_capped": lambda x: np.minimum(0.5, 1 - x),
            "squared": lambda x: (1 - x) ** 2,
        }
        fn = weightings[weighting]
        # Weight prioritization measure
        probabilities = fn(np.asarray(prio_measure))
        norm = probabilities.sum()
        if norm < 1e-10:
            return np.ones_like(prio_measure) / len(prio_measure)
        p = probabilities / norm
        return np.random.choice(opponents, p=p)


REGISTRY = {
    "sp": SPSampling,
    "fsp": FSPSampling,
    "pfsp": PFSPSampling
}
