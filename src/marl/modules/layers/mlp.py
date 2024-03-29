import torch.nn as nn

from marl.modules.layers import ReLuHiddenLayer


class MLP(nn.Module):
    def __init__(self, in_shape, out_shape, depth, hidden_shapes):
        """
        Multilayer Perceptron.

        :param in_shape:
        :param out_shape:
        :param depth: Amount of hidden layers
        :param hidden_shapes: Shapes for each hidden layer. Count of shapes must match depth
        """
        super().__init__()
        if len(hidden_shapes) == depth:
            self.hidden_shapes = hidden_shapes
        else:
            raise Exception("")

        self.hidden_shapes.insert(0, in_shape)
        self.hidden_shapes.append(out_shape)
        self.layers = nn.Sequential(
            nn.Linear(in_shape, self.hidden_shapes[1]),
            *[ReLuHiddenLayer(self.hidden_shapes[i - 1], self.hidden_shapes[i]) for i in range(len(self.hidden_shapes))
              if i - 1 > 0],
        )

    def forward(self, x):
        return self.layers(x)
