import torch
from torch.utils.data import Dataset, DataLoader

from typing import TypeVar, Union, List, Tuple, Iterable

T = TypeVar('T')

MaybeList = Union[T, List[T], Tuple[T]]
MaybeIterable = Union[T, Iterable[T]]
TorchDataset = Union[torch.Tensor, Dataset, DataLoader]
