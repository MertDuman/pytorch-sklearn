import pickle
import copy
import warnings

import numpy
import torch
import torch.nn as nn
import numpy as np
from torch.nn.modules.loss import _Loss
from torch.optim.optimizer import Optimizer as _Optimizer
from torch.utils.data import DataLoader, Dataset

from pytorch_sklearn.callbacks import Callback
from pytorch_sklearn.callbacks.predefined import History
from pytorch_sklearn.utils.datasets import DefaultDataset, CUDADataset
from pytorch_sklearn.utils.class_utils import set_properties_hidden
from pytorch_sklearn.utils.func_utils import to_tensor, to_safe_tensor, create_dirs, stack_if_list_of_list, to_device
from pytorch_sklearn.utils.custom_types import TorchDataset, MaybeIterable

from typing import Any, Callable, Iterable, Sequence, Mapping, Optional, Union


"""
TODO:
- Documentation missing.

- If train_X is passed as a dataloader, then batch_size should automatically be set to the batch_size of the dataloader.

- Currently, metrics are calculated per batch, summed up, and then divided by the number of batches. Could add an
  option to calculate metrics for all of the data instead of per batch.
  Could also add a 'reduction' parameter to the metrics, e.g. 'mean' or 'sum' or 'last'.

- If fit() is called a second time, when the model is using best weights, it will keep training. Should it?
   Maybe produce a warning which asks if we should continue training with these new weights.

- If net.predict() is called within a callback, it will change the pass_type to 'test', even though we are still in 'train'. It should set pass_type to
  'train' again, once it is finished.
"""


class NeuralNetwork:
    def __init__(self, module: nn.Module, optimizer: _Optimizer, criterion: _Loss):
        # Base parameters
        self.module = module        # SAVED
        self.optimizer = optimizer  # SAVED
        self.criterion = criterion  # SAVED

        # Maintenance parameters
        self._callbacks: Sequence[Callback] = [History()]               # SAVED
        self._using_original = True                                     # SAVED
        self._original_state_dict: Optional[Mapping[str, Any]] = None   # SAVED
        self.keep_training = True
        self.custom_backward = False    # If true, then the user must call backward themselves.
        self.loss_names = ["loss"]      # If the loss is a tuple, then this should be a list of the names of the losses.

        # Fit function parameters
        self._train_X: TorchDataset
        self._train_y: Optional[torch.Tensor]
        self._validate: bool
        self._val_X: TorchDataset
        self._val_y: Optional[torch.Tensor]
        self._max_epochs: int
        self._batch_size: int
        self._use_cuda: bool
        self._fits_gpu: bool
        self._metrics: Mapping[str, Callable]

        # Fit runtime parameters
        self._epoch: int        # SAVED
        self._batch: int        # SAVED
        self._batch_data: Any
        self._batch_out: MaybeIterable[torch.Tensor]
        self._batch_loss: MaybeIterable[torch.Tensor]
        self._pass_type: str
        self._num_batches: int
        self._train_loader: DataLoader
        self._val_loader: DataLoader
        self._device: str

        # Predict function parameters
        self._test_X: TorchDataset
        self._decision_func: Optional[Callable]
        self._decision_func_kw: Mapping[str, Any]

        # Predict runtime parameters
        self._predict_loader: DataLoader
        self._pred_y: Iterable
        self._batch: int
        self._batch_data: Any

        # Score function parameters
        self._test_X: TorchDataset
        self._test_y: Optional[torch.Tensor]
        self._score_func: Optional[Callable]
        self._score_func_kw: Mapping[str, Any]

        # Score runtime parameters
        self._score_loader: DataLoader
        self._out: torch.Tensor
        self._score: Iterable
        self._batch: int
        self._batch_data: Any

    @property
    def history(self) -> History:
        assert isinstance(self.callbacks[0], History)
        return self.callbacks[0]
    
    @property
    def callbacks(self):
        return self._callbacks
    
    @callbacks.setter
    def callbacks(self, callbacks: Sequence[Callback]):
        if len(callbacks) == 0:
            self._callbacks = [self._callbacks[0]]  # Keep history.
        elif not isinstance(callbacks[0], History):
            self._callbacks = [self._callbacks[0]]  # Keep history.
            self._callbacks.extend(callbacks)
        else:
            self._callbacks = callbacks
        
    def zero_grad(self, optimizer: _Optimizer):
        optimizer.zero_grad(set_to_none=True)

    def compute_grad(self, loss: torch.Tensor):
        loss.backward()

    def step_grad(self, optimizer: _Optimizer):
        optimizer.step()

    def backward(self, loss: torch.Tensor, optimizer: _Optimizer):
        self.zero_grad(optimizer)
        self._notify(f"on_grad_compute_begin")
        self.compute_grad(loss)
        self._notify(f"on_grad_compute_end")
        self.step_grad(optimizer)

    def forward(self, X):
        ''' Forward pass of the model. This method should be overridden by subclasses. 

        The default implementation simply calls the module.

        Parameters
        ----------
        X:
            All the data necessary for the model forward pass. By default, this is the first element returned from ``unpack_fit_batch``.
        '''
        return self.module(X)
    
    def compute_loss(self, out, y):
        ''' Computes the loss of the model. This method should be overridden by subclasses.

        The default implementation simply calls the criterion.
        
        Parameters
        ----------
        out:
            The output of the model. By default, this is the output of ``forward``.
            
        y:
            The target values. By default, this is the second element returned from ``unpack_fit_batch``.
        '''
        return self.criterion(out, y)

    # Model Modes
    def train(self):
        self.module.train()
        self._pass_type = "train"

    def val(self):
        self.module.eval()
        self._pass_type = "val"

    def test(self):
        self.module.eval()
        self._pass_type = "test"

    # Model Training Main Functions
    def fit(
        self,
        train_X: TorchDataset,
        train_y: Optional[torch.Tensor] = None,
        validate: bool = False,
        val_X: Optional[TorchDataset] = None,
        val_y: Optional[torch.Tensor] = None,
        max_epochs: int = 10,
        batch_size: int = 32,
        use_cuda: bool = True,
        fits_gpu: bool = False,
        callbacks: Optional[Sequence[Callback]] = None,
        metrics: Optional[Mapping[str, Callable]] = None,
    ):
        # Handle None inputs.
        # Assume we have callbacks. If not, then set as empty array.
        callbacks = self.callbacks if callbacks is None else callbacks
        callbacks = [] if callbacks is None else callbacks
        metrics = {} if metrics is None else metrics
        device = "cuda" if use_cuda else "cpu"
        if not use_cuda and fits_gpu:
            fits_gpu = False
            warnings.warn("Fits gpu is true, but not using CUDA.")

        if max_epochs == -1:
            max_epochs = float("inf") # type: ignore
            warnings.warn("max_epochs is set to -1. Make sure to pass an early stopping method.")

        if isinstance(train_X, DataLoader):
            batch_size = train_X.batch_size

        #  Set fit class parameters
        fit_params = locals().copy()
        fit_params.pop("callbacks")
        set_properties_hidden(**fit_params)

        # Handle Callbacks
        self.callbacks = callbacks

        # Define DataLoaders
        self._train_X = self._to_tensor(self._train_X)
        self._train_y = self._to_tensor(self._train_y) # type: ignore
        self._train_loader = self.get_dataloader(self._train_X, self._train_y, shuffle=True) # type: ignore
        if self._validate:
            self._val_X = self._to_tensor(self._val_X)
            self._val_y = self._to_tensor(self._val_y) # type: ignore
            self._val_loader = self.get_dataloader(self._val_X, self._val_y, shuffle=False) # type: ignore

        self.to_device(self._device)
        self._notify("on_fit_begin")
        self._epoch = 1
        while self._epoch < self._max_epochs + 1:
            if not self.keep_training:
                self._notify("on_fit_interrupted")
                break
            self.train()
            self.fit_epoch(self._train_loader)
            if self._validate:
                with torch.no_grad():
                    self.val()
                    self.fit_epoch(self._val_loader)

            if self._epoch == self._max_epochs:
                break  # so that self._epoch == self._max_epochs when loop exits.
            self._epoch += 1
        self._notify("on_fit_end")

    def fit_epoch(self, data_loader):
        self._num_batches = len(data_loader)
        self._notify(f"on_{self._pass_type}_epoch_begin")
        for self._batch, self._batch_data in enumerate(data_loader, start=1):
            self._batch_data = to_device(self._batch_data, self._device)
            self._notify(f"on_{self._pass_type}_batch_begin")

            self._batch_out, self._batch_loss = self.fit_batch(self._batch_data)
            if self._pass_type == "train" and not self.custom_backward:
                self.backward(self._batch_loss, self.optimizer)

            self._notify(f"on_{self._pass_type}_batch_end")
        self._notify(f"on_{self._pass_type}_epoch_end")
    
    def unpack_fit_batch(self, batch_data):
        ''' Unpacks batch data into X and y. This method should be overridden by subclasses. 
        
        The default implementation assumes that ``batch_data`` is a tuple of ``(X, y)`` and returns ``X`` and ``y``.
        This is a convenience method for subclasses that don't want to override ``fit_batch`` but return many values in their dataloader.
        '''
        return batch_data
                       
    def fit_batch(self, batch_data):
        ''' Compute and return the output and loss for a batch. This method should be overridden by subclasses.
            
        The default implementation assumes that ``batch_data`` is a tuple of ``(X, y)`` and that the model
        outputs a single tensor. The loss is computed using the criterion, model output, and target ``y``.

        Parameters
        ----------
        batch_data:
            Batch data as returned by the dataloader provided to ``fit``.
        '''
        # batch_data = to_device(batch_data, self._device)
        X, y = self.unpack_fit_batch(batch_data)
        out = self.forward(X)
        loss = self.compute_loss(out, y)
        return out, loss

    def predict(
        self,
        test_X: TorchDataset,
        batch_size: Optional[int] = None,
        use_cuda: Optional[bool] = None,
        fits_gpu: Optional[bool] = None,
        decision_func: Optional[Callable] = None,
        **decision_func_kw
    ):
        # Handle None inputs.
        batch_size = batch_size if batch_size is not None else (getattr(self, "_batch_size", None) if getattr(self, "_batch_size", None) is not None else 32)
        use_cuda = use_cuda if use_cuda is not None else (getattr(self, "_use_cuda", None) if getattr(self, "_use_cuda", None) is not None else torch.cuda.is_available())
        fits_gpu = fits_gpu if fits_gpu is not None else (getattr(self, "_fits_gpu", None) if getattr(self, "_fits_gpu", None) is not None else False)

        device = "cuda" if use_cuda else "cpu"
        if not use_cuda and fits_gpu:
            fits_gpu = False
            warnings.warn("Fits gpu is true, but not using CUDA.")

        #  Set predict class parameters
        predict_params = locals().copy()
        set_properties_hidden(**predict_params)

        self._test_X = self._to_tensor(self._test_X)
        self._predict_loader = self.get_dataloader(self._test_X, None, shuffle=False)

        with torch.no_grad():
            self.to_device(self._device)
            self.test()
            self._notify("on_predict_begin")
            self._pred_y = []
            for self._batch, self._batch_data in enumerate(self._predict_loader, start=1):
                self._batch_data = to_device(self._batch_data, self._device)
                pred_y = self.predict_batch(self._batch_data, self._decision_func, **self._decision_func_kw)
                self._pred_y.append(pred_y)
            self._pred_y = stack_if_list_of_list(self._pred_y)
            self._notify("on_predict_end")
        return self._pred_y

    def predict_generator(
        self,
        test_X: TorchDataset,
        batch_size: Optional[int] = None,
        use_cuda: Optional[bool] = None,
        fits_gpu: Optional[bool] = None,
        decision_func: Optional[Callable] = None,
        **decision_func_kw
    ):
        # Handle None inputs.
        batch_size = batch_size if batch_size is not None else (getattr(self, "_batch_size", None) if getattr(self, "_batch_size", None) is not None else 32)
        use_cuda = use_cuda if use_cuda is not None else (getattr(self, "_use_cuda", None) if getattr(self, "_use_cuda", None) is not None else torch.cuda.is_available())
        fits_gpu = fits_gpu if fits_gpu is not None else (getattr(self, "_fits_gpu", None) if getattr(self, "_fits_gpu", None) is not None else False)

        device = "cuda" if use_cuda else "cpu"
        if not use_cuda and fits_gpu:
            fits_gpu = False
            warnings.warn("Fits gpu is true, but not using CUDA.")

        #  Set predict class parameters
        predict_params = locals().copy()
        set_properties_hidden(**predict_params)

        self._test_X = self._to_tensor(self._test_X)
        self._predict_loader = self.get_dataloader(self._test_X, None, shuffle=False)

        with torch.no_grad():
            self.to_device(self._device)
            self.test()
            self._notify("on_predict_begin")
            for self._batch, self._batch_data in enumerate(self._predict_loader, start=1):
                self._batch_data = to_device(self._batch_data, self._device)
                self._pred_y = self.predict_batch(self._batch_data, self._decision_func, **self._decision_func_kw)
                yield self._pred_y
            self._notify("on_predict_end")
    
    def unpack_predict_batch(self, batch_data):
        ''' Unpacks batch data into X. This method should be overridden by subclasses. 
        
        The default implementation assumes that ``batch_data`` is a single tensor.
        This is a convenience method for subclasses that don't want to override ``predict_batch`` but return many values in their dataloader.
        '''
        return batch_data

    def predict_batch(self, batch_data, decision_func: Optional[Callable] = None, **decision_func_kw):
        ''' Compute and return the output for a batch. This method should be overridden by subclasses.
        
        The default implementation assumes that ``batch_data`` is a single tensor.
                
        Parameters
        ----------
        batch_data:
            Batch data as returned by the dataloader provided to ``predict`` or ``predict_generator``.
        decision_func:
            Decision function passed to ``predict`` or ``predict_generator``.. If None, the output of the model is returned.
            Takes model output as input and returns the desired output.
        **decision_func_kw:
            Keyword arguments passed to ``decision_func``, provided to ``predict`` or ``predict_generator``.
        '''
        # batch_data = to_device(batch_data, self._device)
        X = self.unpack_predict_batch(batch_data)
        out = self.forward(X)
        if decision_func is not None:
            out = decision_func(out, **decision_func_kw)
        return out

    def score(
        self,
        test_X: TorchDataset,
        test_y: Optional[torch.Tensor] = None,
        batch_size: Optional[int] = None,
        use_cuda: Optional[bool] = None,
        fits_gpu: Optional[bool] = None,
        score_func: Optional[Callable] = None,
        **score_func_kw
    ):
        # Handle None inputs.
        batch_size = batch_size if batch_size is not None else (getattr(self, "_batch_size", None) if getattr(self, "_batch_size", None) is not None else 32)
        use_cuda = use_cuda if use_cuda is not None else (getattr(self, "_use_cuda", None) if getattr(self, "_use_cuda", None) is not None else torch.cuda.is_available())
        fits_gpu = fits_gpu if fits_gpu is not None else (getattr(self, "_fits_gpu", None) if getattr(self, "_fits_gpu", None) is not None else False)

        device = "cuda" if use_cuda else "cpu"
        if not use_cuda and fits_gpu:
            fits_gpu = False
            warnings.warn("Fits gpu is true, but not using CUDA.")

        #  Set score class parameters
        score_params = locals().copy()
        set_properties_hidden(**score_params)

        self._test_X = self._to_tensor(self._test_X)
        self._test_y = self._to_tensor(self._test_y) # type: ignore
        self._score_loader = self.get_dataloader(self._test_X, self._test_y, shuffle=False) # type: ignore

        with torch.no_grad():
            self.to_device(self._device)
            self.test()
            self._score = []
            for self._batch, self._batch_data in enumerate(self._score_loader, start=1):
                self._batch_data = to_device(self._batch_data, self._device)
                batch_score = self.score_batch(self._batch_data, self._score_func, **self._score_func_kw)
                self._score.append(batch_score)
                
            self._score = torch.tensor(np.stack(self._score)).float().mean(dim=0)
        return self._score
    
    def unpack_score_batch(self, batch_data):
        ''' Unpacks batch data into X and y. This method should be overridden by subclasses. 
        
        The default implementation assumes that ``batch_data`` is a tuple of ``(X, y)`` and returns ``X`` and ``y``.
        This is a convenience method for subclasses that don't want to override ``score_batch`` but return many values in their dataloader.
        '''
        return batch_data
    
    def score_batch(self, batch_data, score_func: Optional[Callable[[Any, Any], Any]] = None, **score_func_kw):
        ''' Compute and return the score for a batch. This method should be overridden by subclasses.
        
        The default implementation assumes that ``batch_data`` is a tuple of tensors ``(X, y)``.
        If ``score_func`` is None, score is the model loss.

        Parameters
        ----------
        batch_data:
            Batch data as returned by the dataloader provided to ``score``.
        score_func:
            Score function passed to ``score``. If None, the criterion is used by default.
            Takes a tuple of tensors ``(y_pred, y_true)`` as input and returns a scalar, tuple of scalars, tensor, or tuple of tensors.
        **score_func_kw:
            Keyword arguments passed to ``score_func``, provided to ``score``.
        '''
        # batch_data = to_device(batch_data, self._device)
        X, y = self.unpack_score_batch(batch_data)
        out = self.module(X)  # TODO: Why did I not use self.forward here?
        
        if score_func is None:
            score = self.compute_loss(out, y).item()
        else:
            score = score_func(self._to_safe_tensor(out), self._to_safe_tensor(y), **score_func_kw)
        return score

    def get_dataloader(self, X: TorchDataset, y: Optional[torch.Tensor], shuffle):
        ''' Return a dataloader for the given X and y. Handles the cases where X is a DataLoader, Dataset, or Tensor. '''
        if isinstance(X, DataLoader):
            return X
        if isinstance(X, Dataset):
            return DataLoader(X, batch_size=self._batch_size, shuffle=shuffle, num_workers=0, pin_memory=True)
        dataset = DefaultDataset(X, y)
        return DataLoader(dataset, batch_size=self._batch_size, shuffle=shuffle, num_workers=0, pin_memory=not X.is_cuda)

    def _notify(self, method_name, **cb_kwargs):
        for callback in self.callbacks:
            # This check is removed because it was stopping methods from subclasses of valid callbacks from being called.
            # if method_name in callback.__class__.__dict__:  # check if method is overridden
            getattr(callback, method_name)(self, **cb_kwargs)

    def _to_tensor(self, X: Any):
        if X is None or isinstance(X, (DataLoader, Dataset)):
            return X
        if numpy.ndim(X) == 0:  # Some type without a dimension.
            return X
        return to_tensor(X, clone=False)

    def _to_safe_tensor(self, X):
        return to_safe_tensor(X, clone=False)
    
    def to_device(self, device):
        self.module = self.module.to(device)
        if isinstance(self.criterion, torch.nn.Module):
            self.criterion = self.criterion.to(device)

    def load_weights(self, weight_checkpoint, strict=True):
        if self._using_original:
            self._original_state_dict = copy.deepcopy(self.get_module_weights())
        self.load_module_weights(weight_checkpoint.best_weights, strict=strict)
        self._using_original = False

    def load_weights_from_path(self, weight_path, strict=True):
        if self._using_original:
            self._original_state_dict = copy.deepcopy(self.get_module_weights())
        if torch.cuda.is_available():
            self.load_module_weights(torch.load(weight_path), strict=strict)
        else:
            self.load_module_weights(torch.load(weight_path, map_location=torch.device("cpu")), strict=strict)
        self._using_original = False

    def load_original_weights(self, strict=True):
        if not self._using_original and self._original_state_dict is not None:  # second condition is always true, but there for type checking.
            self.load_module_weights(self._original_state_dict, strict=strict)
            self._using_original = True
            self._original_state_dict = None

    def set_current_as_original_weights(self):
        self._using_original = True
        self._original_state_dict = None

    def get_module_weights(self):
        ''' Returns the state dict of all the modules that are part of this network. '''
        return self.module.state_dict()
    
    def load_module_weights(self, state_dict, strict=True, map_param_names: Mapping[str, str] = None):
        ''' Loads the given state dict into the modules that are part of this network. '''
        # Workaround for optimizer being on the wrong device. Check ``func_utils.optimizer_to`` for more info.
        checkpoint_device = state_dict[next(iter(state_dict))].device
        self.to_device(checkpoint_device)

        if map_param_names is not None:
            state_dict = {map_param_names.get(k, k): v for k, v in state_dict.items()}
        self.module.load_state_dict(state_dict, strict=strict)

    def get_module_parameters(self):
        ''' Returns all the parameters of the modules that are part of this network. '''
        yield from self.module.parameters()

    def state_dict(self):
        return {
            "module_state": self.get_module_weights(),  # Can modify get_module_weights to return a dict {"module_state": state_dict}
            "original_module_state": self._original_state_dict,
            "using_original": self._using_original,
            "optimizer_state": self.optimizer.state_dict(),
            "criterion_state": self.criterion.state_dict(),
            "epoch": self._epoch,
            "batch": self._batch
        }

    def load_state_dict(self, state_dict, strict=True, map_param_names: Mapping[str, str] = None):
        self.load_module_weights(state_dict["module_state"], strict=strict, map_param_names=map_param_names)
        self._original_state_dict = state_dict["original_module_state"]
        self._using_original = state_dict["using_original"]
        self.optimizer.load_state_dict(state_dict["optimizer_state"])
        self.criterion.load_state_dict(state_dict["criterion_state"])
        self._epoch = state_dict["epoch"]
        self._batch = state_dict["batch"]

    @classmethod
    def save_class(cls, net: "NeuralNetwork", savepath: str):
        d = {
            "net_state": net.state_dict(),
            "callbacks": []
        }

        for i, callback in enumerate(net.callbacks):
            d["callbacks"].append(callback.state_dict())

        create_dirs(savepath)
        with open(savepath, "wb") as f:
            torch.save(d, f)

    @classmethod
    def load_class(cls, net: "NeuralNetwork", callbacks: Sequence[Callback], loadpath: str, strict=True, map_param_names: Mapping[str, str] = None):
        ''' Loads the state of the network from the given checkpoint path.

        Parameters
        ----------
        net:
            The network to load the state into.
        callbacks:
            If provided, the callbacks to load the state into. These should be the same callbacks, in the same order, as they were when the state was saved.
            This could be an empty list, in which case the callbacks are not loaded. This is useful if only the network state is needed, e.g. for inference.
        loadpath:
            The path to the checkpoint file.
        strict:
            Same as in ``nn.Module.load_state_dict``, the keys in the checkpoint state dict must match the keys returned by this network's module's ``state_dict``.
        map_param_names:
            A mapping from the keys in the checkpoint state dict to the keys in this network's module's ``state_dict``. This is a convenience feature for when
            the module's parameter names have changed since the checkpoint was saved. One can load the checkpoint by providing a mapping from the old names to the new names,
            and save the checkpoint with the new names.
        '''
        with open(loadpath, "rb") as f:
            map_location = None if torch.cuda.is_available() else torch.device("cpu")  # force CPU if no CUDA
            try:
                d = torch.load(f, map_location=map_location)
            except RuntimeError:
                print('Failed to load with torch. Trying with pickle.')
                f.seek(0)
                d = pickle.load(f)  # backwards compat last resort, likely fails if the weights are on GPU and there is no CUDA
                
        net.load_state_dict(d["net_state"], strict=strict, map_param_names=map_param_names)
        net.callbacks = callbacks
        for i, callback in enumerate(net.callbacks):
            try:
                callback.load_state_dict(d["callbacks"][i])
            except:
                print(f"Couldn't load state for callback {i}: {type(callback).__name__}")
