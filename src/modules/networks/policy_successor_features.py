from modules.networks.mlp import MLP


class PolicySuccessorFeatures(MLP):
    def __init__(self, input_shape, out_shape):
        super().__init__(input_shape, out_shape, 2, [64, 128])


