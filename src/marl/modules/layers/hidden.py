import torch.nn as nn


class ReLuHiddenLayer(nn.Module):
    def __init__(self, input_shape, output_shape):
        super().__init__()
        self.layers = nn.Sequential(
            nn.ReLU(),
            nn.Linear(input_shape, output_shape)
        )

    def forward(self, x):
        return self.layers(x)
