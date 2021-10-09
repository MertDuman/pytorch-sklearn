import torch
from torch.utils.data import Dataset

from pytorch_sklearn.utils.func_utils import to_tensor


class DefaultDataset(Dataset):
    def __init__(self, X, y, device="cpu"):
        if not isinstance(X, torch.Tensor):
            X = torch.tensor(X, device=device)
        if not isinstance(y, torch.Tensor):
            y = torch.tensor(y, device=device)
        if X.device != device:
            X = X.to(device)
        if y.device != device:
            y = y.to(device)

        self.X = X
        self.y = y
        self.n = X.shape[0]

    def __len__(self):
        return self.n

    def __getitem__(self, index):
        return self.X[index, ...], self.y[index, ...]
