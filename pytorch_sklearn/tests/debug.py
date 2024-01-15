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
from pytorch_sklearn.utils.func_utils import to_safe_tensor

from pytorch_sklearn.neural_network.diffusion_network import DiffusionUtils

from sonn.building_blocks import Downsample2d, Upsample2d
from sonn.norm_layers import LayerNormNLP2d
from sonn.superonn_final import SuperONN2d

from PIL import Image

from collections import Iterable as CIterable
from typing import Iterable, Union, List
from pytorch_sklearn.utils.func_utils import to_device


### Neural Network ###
# %%
model = nn.Sequential(
    nn.Conv2d(3, 128, 3, padding=1),
    nn.ReLU(),
    nn.Conv2d(128, 128, 3, padding=1),
    nn.Conv2d(128, 128, 3, padding=1),
    nn.Conv2d(128, 128, 3, padding=1),
    nn.Conv2d(128, 128, 3, padding=1),
    nn.Conv2d(128, 128, 3, padding=1),
    nn.Conv2d(128, 128, 3, padding=1),
    nn.Conv2d(128, 128, 3, padding=1),
    nn.Conv2d(128, 128, 3, padding=1),
    nn.Conv2d(128, 128, 3, padding=1),
    nn.Conv2d(128, 128, 3, padding=1),
    nn.Conv2d(128, 128, 3, padding=1),
    nn.Conv2d(128, 128, 3, padding=1),
    nn.ReLU(),
    nn.Conv2d(128, 3, 3, padding=1),
)
optim = torch.optim.Adam(model.parameters(), lr=1e-3)
crit = nn.L1Loss()

net = NeuralNetwork(model, optim, crit)

X = torch.randn(10, 3, 320, 320)
X = (X - X.min()) / (X.max() - X.min())
y = torch.randn(10, 3, 320, 320)
y = (y - y.min()) / (y.max() - y.min())

# %%
net.fit(
    train_X=X,
    train_y=y,
    validate=True,
    val_X=X,
    val_y=y,
    max_epochs=2,
    batch_size=1,
    use_cuda=True,
    callbacks=[Verbose(verbose=3, per_batch=True)],
    metrics={'diff': lambda out, inp: (out - inp[1]).abs().mean()},
)

ypred = net.predict(X, use_cuda=True)

plt.subplot(1, 3, 1)
plt.imshow(to_safe_tensor(X[0]).permute(1, 2, 0))
plt.title('X')
plt.axis('off')
plt.subplot(1, 3, 2)
plt.imshow(to_safe_tensor(ypred[0]).permute(1, 2, 0))
plt.axis('off')
plt.title('ypred')
plt.subplot(1, 3, 3)
plt.imshow(to_safe_tensor(y[0]).permute(1, 2, 0))
plt.title('y')
plt.axis('off')
plt.show()


### CycleGAN ###

# %%
class AbsModule(nn.Module):
    def forward(self, x):
        return torch.abs(x)
    
G_A = nn.Sequential(
    nn.Conv2d(3, 32, 3, padding=1),
    nn.ReLU(),
    nn.Conv2d(32, 3, 3, padding=1),
    nn.ReLU(),
    nn.Conv2d(3, 3, 3, padding=1),
)
G_B = nn.Sequential(
    nn.Conv2d(3, 32, 3, padding=1),
    nn.ReLU(),
    nn.Conv2d(32, 3, 3, padding=1),
    nn.ReLU(),
    nn.Conv2d(3, 3, 3, padding=1),
)
D_A = nn.Sequential(
    AbsModule(),
    nn.Conv2d(3, 32, 3, padding=1),
    nn.MaxPool2d(4),
    nn.ReLU(),
    nn.Conv2d(32, 32, 3, padding=1),
    nn.MaxPool2d(4),
    nn.ReLU(),
    nn.Conv2d(32, 1, 3, padding=1),
    nn.MaxPool2d(2),
    nn.AdaptiveAvgPool2d(1),
    nn.Flatten(),
)
D_B = nn.Sequential(
    AbsModule(),
    nn.Conv2d(3, 32, 3, padding=1),
    nn.MaxPool2d(4),
    nn.ReLU(),
    nn.Conv2d(32, 32, 3, padding=1),
    nn.MaxPool2d(4),
    nn.ReLU(),
    nn.Conv2d(32, 1, 3, padding=1),
    nn.MaxPool2d(2),
    nn.AdaptiveAvgPool2d(1),
    nn.Flatten(),
)

G_optim = torch.optim.Adam(list(G_A.parameters()) + list(G_B.parameters()), lr=2e-4)
D_optim = torch.optim.Adam(list(D_A.parameters()) + list(D_B.parameters()), lr=2e-4)

cyclegan = CycleGAN(G_A, G_B, D_A, D_B, G_optim, D_optim)

class STDMetric(nn.Module):
    def __init__(self):
        super().__init__()
        self.stdA = []
        self.stdB = []
        self.stdA2B = []
        self.stdB2A = []

    def forward(self, batch_out, batch_data):
        A2B, B2A, *_ = batch_out
        A, B = batch_data
        self.stdA.append(A.std().item())
        self.stdB.append(B.std().item())
        self.stdA2B.append(A2B.std().item())
        self.stdB2A.append(B2A.std().item())
        return 0
    
class CycleGANDataset(Dataset):
    def __init__(self):
        self.A = torch.randn(10, 3, 32, 32) * .1
        self.B = torch.randn(10, 3, 32, 32) * .8

    def __len__(self):
        return 10
    
    def __getitem__(self, index):
        return self.A[index], self.B[index]
    
# %%
    
cyclegan.fit(
    train_X=CycleGANDataset(),
    max_epochs=700,
    use_cuda=True,
    callbacks=[Verbose(per_batch=False)],
    metrics={'std': STDMetric()},
)

# %%

A2B, B2A, *_ = cyclegan.predict(CycleGANDataset(), use_cuda=True)

print(A2B.std(), B2A.std())

plt.plot(cyclegan._metrics['std'].stdA, label='A')
plt.plot(cyclegan._metrics['std'].stdB, label='B')
plt.plot(cyclegan._metrics['std'].stdA2B, label='A2B')
plt.plot(cyclegan._metrics['std'].stdB2A, label='B2A')
plt.legend()
plt.show()


### GAN ###
# %%
nn.ConvTranspose2d( 1, 3 * 8, 4, 1, 0)(torch.randn(1, 1, 1, 1)).shape