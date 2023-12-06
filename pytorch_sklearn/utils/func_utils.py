import numpy as np
import torch

from typing import Iterable, Union, List, Tuple


def create_dirs(path):
    import os
    dirs = os.path.dirname(path)
    if dirs != "" and not os.path.exists(dirs):
        os.makedirs(dirs)


def stack_if_list_of_list(arr):
    ''' 
    If arr is a list of list, combine them column-wise. Assumes the first dimension is the batch dimension for tensors.
    
    E.g. arr = [[5, torch.randn(1,2,1,1), [2, 3]],
                [6, torch.randn(1,2,1,1), [4, 5]]]
         Then this function returns 
               [torch.tensor([5, 6]), torch.cat([el[1] for el in arr]), torch.tensor([[2, 3], [4, 5]])]

    Otherwise, just returns torch.cat(arr).
    '''
    if isinstance(arr[0], (list, tuple)):
        ret = []
        for col in zip(*arr):
            if isinstance(col[0], torch.Tensor):
                ret.append(torch.cat(col, dim=0))
            else:
                ret.append(torch.tensor(np.stack(col)))
        return ret
    else:
        return torch.cat(arr)
    

def optimizer_to(optimizer, device):
    '''
    Rough workaround for PyTorch not having a .to() method for optimizers.

    Optimizers do not move their parameters when their module moves to a different device. PyTorch currently solves this by
    initializing the optimizer on the first call to ``.step()``, so it knows about the module's device. 
    However, if the module and the optimizer are loaded from a checkpoint, then there is no first call to ``.step()``.

    When loading a checkpoint in PyTorch, it doesn't matter where the checkpoint was saved. The checkpoint will be loaded
    to the device that the module is currently on. The optimizer ALSO loads to the device that the MODULE is currently on,
    i.e. when loading a checkpoint for the optimizer, PyTorch checks the device of the module and loads the optimizer to that.
    
    If you create the module, create the optimizer, load the checkpoints, and then move the module to a different device,
    the optimizer will still be on the original device. Since the optimizer will not perform the first call to ``.step()``
    (because it resumes from a checkpoint), it will not know about the module's new device.

    The correct way to go about this is to load the checkpoint for the optimizer only after the module has been moved to the
    correct device. Alternatively, you can create the optimizer after the module has been moved to the correct device.

    These options may not always be possible (or easy), so this function is a workaround. It tries to move all parameters of the optimizer
    to the given device. Note that since PyTorch does not have a ``.to()`` method for optimizers, this function is not guaranteed
    to work.
    '''
    for param in optimizer.state.values():
        for key, value in param.items():
            if isinstance(value, torch.Tensor):
                param[key] = value.to(device)


def to_numpy(X: torch.Tensor, clone=True):
    """
    Safely convert from PyTorch tensor to numpy.
    ``clone`` is set to True by default to mitigate side-effects that this function might cause.
    For instance:
        ``torch.Tensor.cpu`` will clone the object if it is in GPU, but won't if it is in CPU.
        ``clone`` allows this function to clone the input always.
    """
    if isinstance(X, np.ndarray):
        if clone:
            return X.copy()
        else:
            return X

    old_memory = get_memory_loc(X)
    if X.requires_grad:
        X = X.detach()
    if X.is_cuda:
        X = X.cpu()
    if clone and old_memory == get_memory_loc(X):
        X = X.clone()
    return X.numpy()


def to_tensor(X: Iterable, device=None, dtype=None, clone=True):
    """
    Converts the given input to ``torch.Tensor`` and optionally clones it (True by default).
    If ``clone`` is False, this function may still clone the input, read ``torch.as_tensor``.
    """
    old_memory = get_memory_loc(X)
    X = torch.as_tensor(X, device=device, dtype=dtype)
    if clone and old_memory == get_memory_loc(X):
        X = X.clone()
    return X


def to_safe_tensor(X: Union[torch.Tensor, List[torch.Tensor], Tuple[torch.Tensor]], clone=True):
    """
    Convert the given ``torch.Tensor`` or list of tensors to another one that is detached and is in cpu.
    ``clone`` is set to True by default to mitigate side-effects that this function might cause.
    For instance:
        ``torch.Tensor.cpu`` will clone the object if it is in GPU, but won't if it is in CPU.
        ``clone`` allows this function to clone the input always.
    """
    if isinstance(X, (list, tuple)):
        return list(map(lambda x: to_safe_tensor(x, clone=clone), X))

    old_memory = get_memory_loc(X)
    if X.requires_grad:
        X = X.detach()
    if X.is_cuda:
        X = X.cpu()
    if clone and old_memory == get_memory_loc(X):
        X = X.clone()
    return X


def get_memory_loc(X):
    if isinstance(X, np.ndarray):
        return X.__array_interface__['data'][0]
    if isinstance(X, torch.Tensor):
        return X.data_ptr()
    return -1
    # raise TypeError("Cannot get memory location of this data type.")
