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
from pytorch_sklearn.utils.func_utils import to_tensor


"""
TODO:
1) If fit() is called a second time, when the model is using best weights, it will keep training. Should it?
   Maybe produce a warning which asks if we should continue training with these new weights.
   
2) Currently, neither of predict(), predict_proba(), and score() can evaluate the performance of the model based on some
   metric other than self.criterion. For instance, we can't get the model accuracy in a simple way.
   
3) Documentation missing.

4) predict_proba() is a misleading name, as the unmodified network output does not need to be probabilities.

5) Allow direct read access from NeuralNetwork to History.

6) Validation needs batch size as well, because the data might not fit into memory. [DONE]
"""


class NeuralNetwork:
    @property
    def callbacks(self):
        return self.cbmanager.callbacks

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
        self._device = None
        self._callbacks = None
        self._metrics = None

        # Fit runtime parameters
        self._epoch = None
        self._batch = None
        self._batch_X = None
        self._batch_y = None
        self._batch_out = None
        self._batch_loss = None
        self._pass_type = None
        self._num_batches = None
        self._train_loader = None
        self._val_loader = None

        # Predict runtime parameters
        self._test_X = None
        self._decision_func = None
        self._decision_func_kw = None
        self._y_pred = None

        # Predict proba runtime parameters
        self._test_X = None
        self._proba = None

        # Score runtime parameters
        self._test_X = None
        self._test_y = None
        self._out = None
        self._score = None

    # Model Training Core Functions
    def forward(self, X: torch.Tensor):
        return self.module(X)

    def get_loss(self, y_pred: torch.Tensor, y_true: torch.Tensor):
        return self.criterion(y_pred, y_true)

    def zero_grad(self):
        self.optimizer.zero_grad()

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
        train_X=None,
        train_y=None,
        validate=False,
        val_X=None,
        val_y=None,
        max_epochs=10,
        batch_size=32,
        use_cuda=True,
        callbacks=None,
        metrics=None
    ):
        # Handle None inputs.
        callbacks = [] if callbacks is None else callbacks
        metrics = {} if metrics is None else metrics
        device = "cuda" if use_cuda else "cpu"
        if max_epochs == -1:
            max_epochs = float("inf")
            warnings.warn("max_epochs is set to -1. Make sure to pass an early stopping method.")

        #  Set fit class parameters
        fit_params = locals().copy()
        set_properties_hidden(**fit_params)

        # Handle CallbackManager
        self.cbmanager.callbacks = callbacks

        # Define DataLoaders
        self._train_loader = self.get_dataloader(self._train_X, self._train_y, self._batch_size, shuffle=True)
        if self._validate:
            self._val_loader = self.get_dataloader(self._val_X, self._val_y, self._batch_size, shuffle=True)

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
        for self._batch, (self._batch_X, self._batch_y) in enumerate(data_loader, start=1):
            self.fit_batch(self._batch_X, self._batch_y)
        self._notify(f"on_{self._pass_type}_epoch_end")

    def fit_batch(self, X, y):
        self._notify(f"on_{self._pass_type}_batch_begin")
        self._batch_out = self.forward(X)
        self._batch_loss = self.get_loss(self._batch_out, y)
        if self._pass_type == "train":
            self.backward(self._batch_loss)
        self._notify(f"on_{self._pass_type}_batch_end")

    def get_dataloader(self, X, y, batch_size, shuffle):
        dataset = self.get_dataset(X, y)
        return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)

    def get_dataset(self, X, y):
        if isinstance(X, Dataset):
            return X
        return DefaultDataset(X, y, device=self._device)

    def predict(self, test_X, decision_func=None, **decision_func_kw):
        #  Set predict class parameters
        predict_params = locals().copy()
        set_properties_hidden(**predict_params)

        self._test_X = self._to_tensor(self._test_X)
        with torch.no_grad():
            self.test()
            self._notify("on_predict_begin")
            self._y_pred = self.forward(self._test_X)
            if self._decision_func is not None:
                self._y_pred = self._decision_func(self._y_pred, **self._decision_func_kw)
            self._notify("on_predict_end")
        return self._y_pred

    def predict_proba(self, test_X):
        #  Set predict_proba class parameters
        proba_params = locals().copy()
        set_properties_hidden(**proba_params)

        self._test_X = self._to_tensor(self._test_X)
        with torch.no_grad():
            self.test()
            self._notify("on_predict_proba_begin")
            self._proba = self.forward(self._test_X)
            self._notify("on_predict_proba_end")
        return self._proba

    def score(self, test_X, test_y):
        #  Set score class parameters
        score_params = locals().copy()
        set_properties_hidden(**score_params)

        self._test_X = self._to_tensor(self._test_X)
        self._test_y = self._to_tensor(self._test_y)
        with torch.no_grad():
            self.test()
            self._out = self.forward(self._test_X)
            self._score = self.get_loss(self._out, self._test_y)
        return self._score

    def _notify(self, method_name, **cb_kwargs):
        for callback in self.cbmanager.callbacks:
            if method_name in callback.__class__.__dict__:  # check if method is overridden
                getattr(callback, method_name)(self, **cb_kwargs)

    def _to_tensor(self, X):
        return to_tensor(X, device=self._device, clone=False)

    def load_weights(self, weight_checkpoint):
        if self._using_original:
            self._original_state_dict = copy.deepcopy(self.module.state_dict())
        self.module.load_state_dict(weight_checkpoint.best_weights)
        self._using_original = False

    def load_weights_from_path(self, weight_path):
        if self._using_original:
            self._original_state_dict = copy.deepcopy(self.module.state_dict())
        self.module.load_state_dict(torch.load(weight_path))
        self._using_original = False

    def load_original_weights(self):
        if not self._using_original:
            self.module.load_state_dict(self._original_state_dict)
            self._using_original = True
            self._original_state_dict = None

    @classmethod
    def save_class(cls, net, savepath):
        d = {
            "module_state": net.module.state_dict(),
            "original_module_state": net._original_state_dict,
            "using_original": net._using_original,
            "optimizer_state": net.optimizer.state_dict(),
            "criterion_state": net.criterion.state_dict(),
            "cbmanager": net.cbmanager
        }
        with open(savepath, "wb") as f:
            pickle.dump(d, f)

    @classmethod
    def load_class(cls, loadpath, module=None, optimizer=None, criterion=None):
        with open(loadpath, "rb") as f:
            d = pickle.load(f)

        if module is not None:
            module.load_state_dict(d["module_state"])
        if optimizer is not None:
            optimizer.load_state_dict(d["optimizer_state"])
        if criterion is not None:
            criterion.load_state_dict(d["criterion_state"])
        net = NeuralNetwork(module, optimizer, criterion)
        net.cbmanager = d["cbmanager"]
        net._using_original = d["using_original"]
        net._original_state_dict = d["original_module_state"]
        return net

