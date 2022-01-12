import pickle
import copy
import warnings

import torch
from torch.nn.modules.loss import _Loss
from torch.optim.optimizer import Optimizer as _Optimizer
from torch.utils.data import DataLoader, Dataset

from pytorch_sklearn.utils import DefaultDataset
from pytorch_sklearn.callbacks import CallbackManager
from pytorch_sklearn.utils.class_utils import set_properties_hidden
from pytorch_sklearn.utils.func_utils import to_tensor, to_safe_tensor
from pytorch_sklearn.neural_network import NeuralNetwork


class GenerativeNetwork(NeuralNetwork):
    pass