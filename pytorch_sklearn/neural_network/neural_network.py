import pickle
import copy
import warnings

import numpy
import torch
import numpy as np
from torch.nn.modules.loss import _Loss
from torch.optim.optimizer import Optimizer as _Optimizer
from torch.utils.data import DataLoader, Dataset

from pytorch_sklearn.callbacks import CallbackManager, Callback
from pytorch_sklearn.utils.datasets import DefaultDataset, CUDADataset
from pytorch_sklearn.utils.class_utils import set_properties_hidden
from pytorch_sklearn.utils.func_utils import to_tensor, to_safe_tensor, create_dirs

from typing import Iterable, Optional, Union


"""
TODO:
- Documentation missing.
  
- Adding metrics at a second or third fit call results in an error, because we only initialize metrics to the
  history track on the first fit call.

- Currently, metrics are calculated per batch, summed up, and then divided by the number of batches. Could add an
  option to calculate metrics for all of the data instead of per batch.

- If fit() is called a second time, when the model is using best weights, it will keep training. Should it?
   Maybe produce a warning which asks if we should continue training with these new weights.
"""


class NeuralNetwork:
    def __init__(self, module: torch.nn.Module, optimizer: _Optimizer, criterion: _Loss):
        # Base parameters
        self.module = module  # SAVED
        self.optimizer = optimizer  # SAVED
        self.criterion = criterion  # SAVED
        self.cbmanager = CallbackManager()  # SAVED
        self.keep_training = True

        # Maintenance parameters
        self._using_original = True  # SAVED
        self._original_state_dict = None  # SAVED

        # Fit function parameters
        self._train_X = None
        self._train_y = None
        self._validate = None
        self._val_X = None
        self._val_y = None
        self._max_epochs = None
        self._batch_size = None
        self._use_cuda = None
        self._fits_gpu = None
        self._callbacks = None
        self._metrics = None

        # Fit runtime parameters
        self._epoch = None  # SAVED
        self._batch = None  # SAVED
        self._batch_X = None
        self._batch_y = None
        self._batch_args = None
        self._batch_out = None
        self._batch_loss = None
        self._pass_type = None
        self._num_batches = None
        self._train_loader = None
        self._val_loader = None
        self._device = None

        # Predict function parameters
        self._test_X = None
        self._decision_func = None
        self._decision_func_kw = None

        # Predict runtime parameters
        self._predict_loader = None
        self._pred_y = None
        self._batch = None
        self._batch_X = None
        self._batch_args = None

        # Score function parameters
        self._test_X = None
        self._test_y = None
        self._score_func = None
        self._score_func_kw = None

        # Score runtime parameters
        self._score_loader = None
        self._out = None
        self._score = None
        self._batch = None
        self._batch_X = None
        self._batch_y = None
        self._batch_args = None

    @property
    def callbacks(self):
        return self.cbmanager.callbacks

    @property
    def history(self):
        return self.cbmanager.history

    # Model Training Core Functions
    def forward(self, X: torch.Tensor, *args: Iterable):
        ''' Simply perform forward pass through the model and return the batch output. '''
        return self.module(X)

    def get_loss(self, y_pred: torch.Tensor, y_true: torch.Tensor, *args: Iterable):
        ''' Calculate and return a single loss for the given batch. '''
        return self.criterion(y_pred, y_true)
        
    def zero_grad(self):
        self.optimizer.zero_grad(set_to_none=True)

    def compute_grad(self, loss: torch.Tensor):
        loss.backward()

    def step_grad(self):
        self.optimizer.step()

    def backward(self, loss: torch.Tensor):
        self.zero_grad()
        self._notify(f"on_grad_compute_begin")
        self.compute_grad(loss)
        self._notify(f"on_grad_compute_end")
        self.step_grad()

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
        train_X,
        train_y=None,
        validate=False,
        val_X=None,
        val_y=None,
        max_epochs=10,
        batch_size=32,
        use_cuda=True,
        fits_gpu=False,
        callbacks=None,
        metrics=None
    ):
        # Handle None inputs.
        # Assume cbmanager has callbacks. If not, then set as empty array.
        callbacks = self.cbmanager._callbacks if callbacks is None else callbacks
        callbacks = [] if callbacks is None else callbacks
        metrics = {} if metrics is None else metrics
        device = "cuda" if use_cuda else "cpu"
        if not use_cuda and fits_gpu:
            fits_gpu = False
            warnings.warn("Fits gpu is true, but not using CUDA.")

        if max_epochs == -1:
            max_epochs = float("inf")
            warnings.warn("max_epochs is set to -1. Make sure to pass an early stopping method.")

        #  Set fit class parameters
        fit_params = locals().copy()
        set_properties_hidden(**fit_params)

        # Handle CallbackManager
        self.cbmanager.callbacks = callbacks

        # Define DataLoaders
        self._train_X = self._to_tensor(self._train_X)
        self._train_y = self._to_tensor(self._train_y)
        self._train_loader = self.get_dataloader(self._train_X, self._train_y, shuffle=True)
        if self._validate:
            self._val_X = self._to_tensor(self._val_X)
            self._val_y = self._to_tensor(self._val_y)
            self._val_loader = self.get_dataloader(self._val_X, self._val_y, shuffle=False)

        # Begin Fit
        self.module = self.module.to(self._device)
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
        for self._batch, (self._batch_X, self._batch_y, *self._batch_args) in enumerate(self.unpack_train_loader(data_loader), start=1):
            self._batch_X = self._batch_X.to(self._device, non_blocking=True)
            self._batch_y = self._batch_y.to(self._device, non_blocking=True)
            self._fit_batch_wrapper(self._batch_X, self._batch_y, *self._batch_args)
        self._notify(f"on_{self._pass_type}_epoch_end")

    def _fit_batch_wrapper(self, X, y, *args):
        ''' Internal wrapper for fit_batch that notifies the callbacks and handles the backward pass. '''
        self._notify(f"on_{self._pass_type}_batch_begin")
        self._batch_out, self._batch_loss = self.fit_batch(X, y, *args)
        if self._pass_type == "train":
            self.backward(self._batch_loss)
        self._notify(f"on_{self._pass_type}_batch_end")
                       
    def fit_batch(self, X, y, *args):
        ''' Compute and return the output and loss for a batch. '''
        out = self.forward(X, *args)
        loss = self.get_loss(out, y, *args)
        return out, loss

    def predict(
        self,
        test_X,
        batch_size=None,
        use_cuda=None,
        fits_gpu=None,
        decision_func=None,
        **decision_func_kw
    ):
        # Handle None inputs.
        batch_size = batch_size if batch_size is not None else self._batch_size
        use_cuda = use_cuda if use_cuda is not None else self._use_cuda
        fits_gpu = fits_gpu if fits_gpu is not None else self._fits_gpu

        # These asserts will trigger if predict is called before calling fit and not passing these parameters.
        assert batch_size is not None, "Batch size is not set."
        assert use_cuda is not None, "Device is not set."
        assert fits_gpu is not None, "fits_gpu is not set."

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
            self.module = self.module.to(self._device)
            self.test()
            self._notify("on_predict_begin")
            self._pred_y = []
            for self._batch, (self._batch_X, *self._batch_args) in enumerate(self.unpack_predict_loader(self._predict_loader), start=1):
                self._batch_X = self._batch_X.to(self._device, non_blocking=True)
                pred_y = self.forward(self._batch_X, *self._batch_args)
                if self._decision_func is not None:
                    pred_y = self._decision_func(pred_y, *self._batch_args, **self._decision_func_kw)
                self._pred_y.append(pred_y)
            self._pred_y = torch.cat(self._pred_y)
            self._notify("on_predict_end")
        return self._pred_y

    def predict_generator(
        self,
        test_X,
        batch_size=None,
        use_cuda=None,
        fits_gpu=None,
        decision_func=None,
        **decision_func_kw
    ):
        # Handle None inputs.
        batch_size = batch_size if batch_size is not None else self._batch_size
        use_cuda = use_cuda if use_cuda is not None else self._use_cuda
        fits_gpu = fits_gpu if fits_gpu is not None else self._fits_gpu

        # These asserts will trigger if predict is called before calling fit and not passing these parameters.
        assert batch_size is not None, "Batch size is not set."
        assert use_cuda is not None, "Device is not set."
        assert fits_gpu is not None, "fits_gpu is not set."

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
            self.module = self.module.to(self._device)
            self.test()
            self._notify("on_predict_begin")
            for self._batch, (self._batch_X, *self._batch_args) in enumerate(self.unpack_predict_loader(self._predict_loader), start=1):
                self._batch_X = self._batch_X.to(self._device, non_blocking=True)
                self._pred_y = self.forward(self._batch_X, *self._batch_args)
                if self._decision_func is not None:
                    self._pred_y = self._decision_func(self._pred_y, *self._batch_args, **self._decision_func_kw)
                yield self._pred_y
            self._notify("on_predict_end")

    def score(
        self,
        test_X,
        test_y=None,
        batch_size=None,
        use_cuda=None,
        fits_gpu=None,
        score_func=None,
        **score_func_kw
    ):
        # Handle None inputs.
        batch_size = batch_size if batch_size is not None else self._batch_size
        use_cuda = use_cuda if use_cuda is not None else self._use_cuda
        fits_gpu = fits_gpu if fits_gpu is not None else self._fits_gpu

        # These asserts will trigger if score is called before calling fit and not passing these parameters.
        assert batch_size is not None, "Batch size is not set."
        assert use_cuda is not None, "Device is not set."
        assert fits_gpu is not None, "fits_gpu is not set."

        device = "cuda" if use_cuda else "cpu"
        if not use_cuda and fits_gpu:
            fits_gpu = False
            warnings.warn("Fits gpu is true, but not using CUDA.")

        #  Set score class parameters
        score_params = locals().copy()
        set_properties_hidden(**score_params)

        self._test_X = self._to_tensor(self._test_X)
        self._test_y = self._to_tensor(self._test_y)
        self._score_loader = self.get_dataloader(self._test_X, self._test_y, shuffle=False)

        with torch.no_grad():
            self.module = self.module.to(self._device)
            self.test()
            self._score = []
            for self._batch, (self._batch_X, self._batch_y, *self._batch_args) in enumerate(self.unpack_score_loader(self._score_loader), start=1):
                self._batch_X = self._batch_X.to(self._device, non_blocking=True)
                self._batch_y = self._batch_y.to(self._device, non_blocking=True)
                batch_out = self.forward(self._batch_X, *self._batch_args)
                if self._score_func is None:
                    batch_loss = self.get_loss(batch_out, self._batch_y, *self._batch_args).item()
                else:
                    batch_loss = self._score_func(self._to_safe_tensor(batch_out), self._to_safe_tensor(self._batch_y), *self._batch_args, **self._score_func_kw)
                self._score.append(batch_loss)
                
            self._score = torch.tensor(np.stack(self._score)).float().mean(dim=0)
        return self._score

    def get_dataloader(self, X: Union[torch.Tensor, Dataset, DataLoader], y: Optional[torch.Tensor], shuffle):
        ''' Return a dataloader for the given X and y. Handles the cases where X is a DataLoader, Dataset, or Tensor. '''
        if isinstance(X, DataLoader):
            return X
        if isinstance(X, Dataset):
            return DataLoader(X, batch_size=self._batch_size, shuffle=shuffle, num_workers=0, pin_memory=True)
        dataset = DefaultDataset(X, y)
        return DataLoader(dataset, batch_size=self._batch_size, shuffle=shuffle, num_workers=0, pin_memory=not X.is_cuda)

    def unpack_train_loader(self, train_loader):
        ''' Override this to unpack the dataloader into X, y, *args. By default, it is assumed that the dataloader returns X, y. '''
        for X, y in train_loader:
            yield X, y,

    def unpack_predict_loader(self, predict_loader):
        ''' Override this to unpack the dataloader into X, *args. By default, it is assumed that the dataloader returns X. '''
        for X in predict_loader:
            yield X,

    def unpack_score_loader(self, score_loader):
        ''' Override this to unpack the dataloader into X, y, *args. By default, it is assumed that the dataloader returns X, y. '''
        for X, y in score_loader:
            yield X, y,

    def _notify(self, method_name, **cb_kwargs):
        for callback in self.cbmanager.callbacks:
            if method_name in callback.__class__.__dict__:  # check if method is overridden
                getattr(callback, method_name)(self, **cb_kwargs)

    def _to_tensor(self, X):
        if X is None or isinstance(X, (DataLoader, Dataset)):
            return X
        if numpy.ndim(X) == 0:  # Some type without a dimension.
            return X
        return to_tensor(X, clone=False)

    def _to_safe_tensor(self, X):
        return to_safe_tensor(X, clone=False)

    def load_weights(self, weight_checkpoint):
        if self._using_original:
            self._original_state_dict = copy.deepcopy(self.module.state_dict())
        self.module.load_state_dict(weight_checkpoint.best_weights)
        self._using_original = False

    def load_weights_from_path(self, weight_path):
        if self._using_original:
            self._original_state_dict = copy.deepcopy(self.module.state_dict())
        if torch.cuda.is_available():
            self.module.load_state_dict(torch.load(weight_path))
        else:
            self.module.load_state_dict(torch.load(weight_path, map_location=torch.device("cpu")))
        self._using_original = False

    def load_original_weights(self):
        if not self._using_original:
            self.module.load_state_dict(self._original_state_dict)
            self._using_original = True
            self._original_state_dict = None

    def set_current_as_original_weights(self):
        self._using_original = True
        self._original_state_dict = None

    def state_dict(self):
        return {
            "module_state": self.module.state_dict(),
            "original_module_state": self._original_state_dict,
            "using_original": self._using_original,
            "optimizer_state": self.optimizer.state_dict(),
            "criterion_state": self.criterion.state_dict(),
            "epoch": self._epoch,
            "batch": self._batch
        }

    def load_state_dict(self, state_dict):
        self.module.load_state_dict(state_dict["module_state"])
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
    def load_class(cls, net: "NeuralNetwork", callbacks: Iterable[Callback], loadpath: str):
        with open(loadpath, "rb") as f:
            if torch.cuda.is_available():
                d = torch.load(f)
            else:
                d = torch.load(f, map_location=torch.device("cpu"))
        net.load_state_dict(d["net_state"])
        net.cbmanager.callbacks = callbacks
        for i, callback in enumerate(net.callbacks):
            try:
                callback.load_state_dict(d["callbacks"][i])
            except:
                print(f"Couldn't load state for callback {i}: {type(callback).__name__}")
