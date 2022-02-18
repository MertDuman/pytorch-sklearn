import torch
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from func_utils import to_tensor


class DefaultDataset(Dataset):
    """
    A simple dataset for convenience.
    """
    def __init__(self, X, y=None):
        if not isinstance(X, torch.Tensor):
            X = torch.tensor(X)
        if y is not None and not isinstance(y, torch.Tensor):
            y = torch.tensor(y)

        self.X = X
        self.y = y
        self.n = X.shape[0]

    def __len__(self):
        return self.n

    def __getitem__(self, index):
        if self.y is None:
            return self.X[index, ...]
        else:
            return self.X[index, ...], self.y[index, ...]


class CUDADataset(Dataset):
    """
    A simple CUDA dataset, where the data will be sent to the cuda device. This is useful when the data is
    small and can be stored entirely in GPU memory, as this will speed up training.
    """
    def __init__(self, X, y=None):
        device = "cuda"
        if not isinstance(X, torch.Tensor):
            X = torch.tensor(X, device=device)
        if y is not None and not isinstance(y, torch.Tensor):
            y = torch.tensor(y, device=device)
        if X.device != device:
            X = X.to(device)
        if y is not None and y.device != device:
            y = y.to(device)

        self.X = X
        self.y = y
        self.n = X.shape[0]

    def __len__(self):
        return self.n

    def __getitem__(self, index):
        if self.y is None:
            return self.X[index, ...]
        else:
            return self.X[index, ...], self.y[index, ...]
