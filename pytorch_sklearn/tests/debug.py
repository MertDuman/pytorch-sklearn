# %%
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as tvt
import torch.optim.lr_scheduler as tlrs
from torch.utils.data import DataLoader, Dataset

import matplotlib.pyplot as plt

from pytorch_sklearn.neural_network.nn_base import NeuralNetworkBase
from pytorch_sklearn.neural_network.neural_network import NeuralNetwork
from pytorch_sklearn.neural_network.generative_network import CycleGAN, R2CGAN
from pytorch_sklearn.callbacks.predefined import Verbose, History, EarlyStopping
from pytorch_sklearn.utils.progress_bar import print_progress
from pytorch_sklearn.frameworks.lr_schedulers import *

from pytorch_sklearn.neural_network.diffusion_network import DiffusionUtils

from sonn.building_blocks import Downsample2d, Upsample2d
from sonn.norm_layers import LayerNormNLP2d
from sonn.superonn_final import SuperONN2d

from PIL import Image

from collections import Iterable as CIterable
from typing import Iterable, Union, List
from pytorch_sklearn.utils.func_utils import to_device

# %%
model = nn.Sequential(nn.Conv2d(3, 3, 3, padding=1))
optim = torch.optim.Adam(model.parameters(), lr=1e-3)
crit = nn.MSELoss()

net = NeuralNetwork(model, optim, crit)

X = torch.randn(1, 3, 32, 32)
y = torch.randn(1, 3, 32, 32)

# %%
net.fit(
    train_X=X,
    train_y=y,
    max_epochs=10,
    callbacks=[Verbose()],
    use_cuda=True
)

