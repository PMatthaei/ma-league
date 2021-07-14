from modules.networks.mlp import MLP


class PolicySuccessorFeatures(MLP):
    def __init__(self, in_shape, out_shape):
        super().__init__(in_shape, out_shape, 2, [64, 128])


