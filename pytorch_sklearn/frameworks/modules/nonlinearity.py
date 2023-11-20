import torch.nn as nn
from pytorch_sklearn.frameworks.modules.linear_units import get_linear_unit


class SimpleGate(nn.Module):
    def __init__(self, nonlinearity='gelu'):
        super().__init__()
        self.nonlinearity = get_linear_unit(nonlinearity)()

    def forward(self, x):
        x1, x2 = x.chunk(2, dim=1)
        return self.nonlinearity(x1) * x2
