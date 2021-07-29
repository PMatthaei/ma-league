from torch.optim import RMSprop

from marl.modules.layers import MLP


class PolicySuccessorFeatures(MLP):
    def __init__(self, in_shape, out_shape):
        super().__init__(in_shape, out_shape, 2, [64, 128])
        self.optimizer = RMSprop(params=self.parameters(), lr=0.01, alpha=0.02, eps=0.02)
