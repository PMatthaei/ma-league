from modules.networks.mlp import MLP


class PolicySuccessorFeatures(MLP):

    def __init__(self, in_shape, out_shape):
        """

        :param in_shape:
        :param out_shape: Number of policies which are used for GPE
        """
        super().__init__(in_shape, out_shape, 2, [64, 128])


