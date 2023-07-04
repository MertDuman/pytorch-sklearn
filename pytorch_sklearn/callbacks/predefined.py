# __all__ = ["History", "Verbose", "EarlyStopping", "LossPlot", "WeightCheckpoint", "LRScheduler", "Tracker", "CallbackInfo"]

from pytorch_sklearn.callbacks import Callback
from pytorch_sklearn.utils.func_utils import to_safe_tensor
from pytorch_sklearn.callbacks.utils import Tally
import pytorch_sklearn as psk

import torch

import matplotlib.pyplot as plt
import numpy as np
import copy

# WeightCheckpoint
import os

# ImageOutputWriter
from os.path import join as osj

# Verbose
import time

# LossPlot
# IPython.get_ipython
import matplotlib as mpl
import sys
import importlib

# Custom
from pytorch_sklearn.utils.progress_bar import print_progress


class History(Callback):
    def __init__(self):
        super().__init__()
        self.name = "History"
        self.track = {}
        self.sessions = []
        self.epoch_metrics: np.ndarray
        self.key_index = {}  # given key in track, return index in epoch_metrics.
        self.num_metrics = -1
        self.session = -1

    def init_track(self, net, pass_type):
        if f"{pass_type}_loss" not in self.track:
            self.track[f"{pass_type}_loss"] = []
            self.key_index[f"{pass_type}_loss"] = 0
        for i, name in enumerate(net._metrics.keys(), start=1):
            if f"{pass_type}_{name}" not in self.track:
                self.track[f"{pass_type}_{name}"] = []
                self.key_index[f"{pass_type}_{name}"] = i

    def on_fit_begin(self, net):
        self.session += 1
        self.num_metrics = len(net._metrics) + 1
        self.init_track(net, "train")
        if net._validate:
            self.init_track(net, "val")
        session_start = len(self.track[next(iter(self.track))]) + 1  # new session starts at epoch = len(first_key) + 1
        self.sessions.append(session_start)  

    def on_train_epoch_begin(self, net):
        self.epoch_metrics = np.zeros(self.num_metrics)  # reset epoch metrics, this is possible because dict keys are ordered in Python 3.7

    def on_val_epoch_begin(self, net):
        self.epoch_metrics = np.zeros(self.num_metrics)  # reset epoch metrics

    def on_train_epoch_end(self, net):
        self._save_metrics(net)

    def on_val_epoch_end(self, net):
        self._save_metrics(net)

    def _save_metrics(self, net):
        self.epoch_metrics = self.epoch_metrics / net._num_batches
        self.track[f"{net._pass_type}_loss"].append(self.epoch_metrics[0])
        for i, name in enumerate(net._metrics.keys(), start=1):
            self.track[f"{net._pass_type}_{name}"].append(self.epoch_metrics[i])

    def on_train_batch_end(self, net):
        self._calculate_metrics(net)

    def on_val_batch_end(self, net):
        self._calculate_metrics(net)

    def _calculate_metrics(self, net: "psk.NeuralNetwork"):
        batch_out = to_safe_tensor(net._batch_out)
        _, batch_y = net.unpack_fit_batch(net._batch_data)
        self.epoch_metrics[0] += net._batch_loss.item()
        for i, metric in enumerate(net._metrics.values(), start=1):
            self.epoch_metrics[i] += metric(batch_out, batch_y)


class CycleGANHistory(History):
    def __init__(self):
        super().__init__()
        self.name = "History"
        self.track = {}
        self.sessions = []
        self.epoch_metrics: np.ndarray
        self.key_index = {}  # given key in track, return index in epoch_metrics.
        self.num_metrics = -1
        self.session = -1

    def init_track(self, net, pass_type):
        if f"{pass_type}_G_A_loss" not in self.track:
            self.track[f"{pass_type}_G_A_loss"] = []
            self.key_index[f"{pass_type}_G_A_loss"] = 0
        if f"{pass_type}_G_B_loss" not in self.track:
            self.track[f"{pass_type}_G_B_loss"] = []
            self.key_index[f"{pass_type}_G_B_loss"] = 1
        if f"{pass_type}_D_A_loss" not in self.track:
            self.track[f"{pass_type}_D_A_loss"] = []
            self.key_index[f"{pass_type}_D_A_loss"] = 2
        if f"{pass_type}_D_B_loss" not in self.track:
            self.track[f"{pass_type}_D_B_loss"] = []
            self.key_index[f"{pass_type}_D_B_loss"] = 3
        for i, name in enumerate(net._metrics.keys(), start=4):
            if f"{pass_type}_{name}" not in self.track:
                self.track[f"{pass_type}_{name}"] = []
                self.key_index[f"{pass_type}_{name}"] = i

    def on_fit_begin(self, net):
        self.session += 1
        self.num_metrics = len(net._metrics) + 4  # 2 generators, 2 discriminators
        self.init_track(net, "train")
        if net._validate:
            self.init_track(net, "val")
        session_start = len(self.track[next(iter(self.track))]) + 1  # new session starts at epoch = len(first_key) + 1
        self.sessions.append(session_start)  

    def on_train_epoch_begin(self, net):
        self.epoch_metrics = np.zeros(self.num_metrics)  # reset epoch metrics, this is possible because dict keys are ordered in Python 3.7

    def on_val_epoch_begin(self, net):
        self.epoch_metrics = np.zeros(self.num_metrics)  # reset epoch metrics

    def on_train_epoch_end(self, net):
        self._save_metrics(net)

    def on_val_epoch_end(self, net):
        self._save_metrics(net)

    def _save_metrics(self, net):
        self.epoch_metrics = self.epoch_metrics / net._num_batches
        self.track[f"{net._pass_type}_G_A_loss"].append(self.epoch_metrics[0])
        self.track[f"{net._pass_type}_G_B_loss"].append(self.epoch_metrics[1])
        self.track[f"{net._pass_type}_D_A_loss"].append(self.epoch_metrics[2])
        self.track[f"{net._pass_type}_D_B_loss"].append(self.epoch_metrics[3])
        for i, name in enumerate(net._metrics.keys(), start=4):
            self.track[f"{net._pass_type}_{name}"].append(self.epoch_metrics[i])

    def on_train_batch_end(self, net):
        self._calculate_metrics(net)

    def on_val_batch_end(self, net):
        self._calculate_metrics(net)

    def _calculate_metrics(self, net: "psk.NeuralNetwork"):
        batch_out = to_safe_tensor(net._batch_out)
        _, batch_y = net.unpack_fit_batch(net._batch_data)
        self.epoch_metrics[0:4] += [loss.item() for loss in net._batch_loss]
        for i, metric in enumerate(net._metrics.values(), start=4):
            self.epoch_metrics[i] += metric(batch_out, batch_y)


class Verbose(Callback):
    def __init__(self, verbose=4, notebook=False):
        """
        Prints the following training information:
            - Current Epoch / Total Epochs
            - Current Batch / Total Batches
            - Loss
            - Metrics
            - Total Time + ETA

        Parameters
        ----------
        verbose : int
            Controls how much is printed. Higher levels include the info from lower levels.
            The following levels are valid:
                0: Current Epoch / Total Epochs
                1: Current Batch / Total Batches
                2: Loss
                3: Metrics
                4: Total Time + ETA
        """
        super().__init__()
        self.name = "Verbose"
        self.verbose = verbose
        self.notebook = notebook

        # Time info
        self.total_time = 0
        self.rem_time = 0
        self.start_time = 0
        self.end_time = 0

        # Previous epochs if this model has stopped and resumed training
        self.prev_epochs = 0

    def state_dict(self):
        return {}

    def on_fit_begin(self, net):
        try: self.prev_epochs = len(net.history.track[next(iter(net.history.track))])
        except: pass

    def on_train_epoch_begin(self, net):
        if self.verbose == 0:
            print(f"Epoch {self.prev_epochs + net._epoch}/{self.prev_epochs + net._max_epochs}", end='\r', flush=True)  # TODO: Bug, no newline sometimes
        else:
            print(f"Epoch {self.prev_epochs + net._epoch}/{self.prev_epochs + net._max_epochs}")

        if self.verbose >= 4:
            self.start_time = time.perf_counter()

    def on_val_epoch_begin(self, net):
        if self.verbose >= 4:
            self.start_time = time.perf_counter()

    def on_train_batch_end(self, net):
        if self.verbose >= 1:
            self._print(net)

    def on_val_batch_end(self, net):
        if self.verbose >= 1:
            self._print(net)

    def _print(self, net):
        # Calculate data
        epoch_metrics = net.history.epoch_metrics / net._batch  # mean batch loss
        if self.verbose >= 4:
            self.end_time = time.perf_counter()
            self.total_time = self.end_time - self.start_time
            self.rem_time = ((net._num_batches - net._batch) * self.total_time) / net._batch

        # Fill print data
        opt = []
        if self.verbose >= 2:
            opt = [f"{net._pass_type}_loss: {epoch_metrics[0]:.3f}"]
        if self.verbose >= 3:
            opt.extend([f"{net._pass_type}_{name}: {epoch_metrics[i]:.3f}" for i, name in enumerate(net._metrics.keys(), start=1)])
        if self.verbose >= 4:
            opt.extend([f"Time: {self.total_time:.2f}", f"ETA: {self.rem_time:.2f}"])
        print_progress(net._batch, net._num_batches, opt=opt, notebook=self.notebook)


class LossPlot(Callback):
    def __init__(self,
                 new_backend="Qt5Agg",
                 pyplot_name="matplotlib.pyplot",
                 max_col=1,
                 block_on_finish=False,
                 savefig=False,
                 savename=None,
                 figure_kw=None):
        super(LossPlot, self).__init__()
        self.get_ipython = __import__('IPython', globals(), locals(), ['get_ipython'], 0).get_ipython

        self.new_backend = new_backend
        self.old_backend = mpl.get_backend()
        self.pyplot_name = pyplot_name
        self.max_col = max_col
        self.is_ipython = self.get_ipython() is not None
        self.block_on_finish = block_on_finish
        self.savefig = savefig
        self.savename = savename
        if self.savefig:
            assert self.savename is not None, "You must provide a savename."
        self.figure_kw = {} if figure_kw is None else figure_kw

        # on_fit_begin
        self.fig = None
        self.axes = None

    def state_dict(self):
        return {"fig": self.fig, "axes": self.axes}

    def on_fit_begin(self, net):
        self.switch_qt5()
        plt.ion()  # turn on interactive mode

        num_metrics = net.history.num_metrics
        nrows = int(np.ceil(num_metrics / self.max_col))

        self.fig, self.axes = plt.subplots(nrows, self.max_col, sharex="all", squeeze=False, **self.figure_kw)

        # Delete unused subplots
        for i in range(num_metrics, nrows * self.max_col):
            self.fig.delaxes(self.axes[i // self.max_col, i % self.max_col])
            # self.axes[i // self.max_col, i % self.max_col].set_visible(False)

        # Define empty lines for loss line
        self.axes[0, 0].set_title(f"{net.criterion.__class__.__name__}")
        self.axes[0, 0].plot([], [], "-o", label="train loss")
        if net._validate:
            self.axes[0, 0].plot([], [], "-o", label="val loss")
        self.axes[0, 0].legend()

        # Define empty lines for other metric lines
        for i, name in enumerate(net._metrics.keys(), start=1):
            ax = self.axes[i // self.max_col, i % self.max_col]
            ax.set_title(f"{name.capitalize()}")
            ax.plot([], [], "-o", label=f"train {name}")
            if net._validate:
                ax.plot([], [], "-o", label=f"val {name}")
            ax.legend()

        # h, l = self.axes[0, 0].get_legend_handles_labels()
        # self.fig.legend(l)

    def on_train_epoch_end(self, net):
        self.plot_metrics(net)

    def on_val_epoch_end(self, net):
        self.plot_metrics(net)

    def plot_metrics(self, net):
        track = net.history.track
        line_idx = 0 if net._pass_type == "train" else 1

        # Change plot for loss line
        data = track[f"{net._pass_type}_loss"]
        self.axes[0, 0].lines[line_idx].set_data(np.arange(len(data)), data)
        self.axes[0, 0].relim()
        self.axes[0, 0].autoscale_view()

        # Change plot for other metric lines
        for i, name in enumerate(net._metrics.keys(), start=1):
            data = track[f"{net._pass_type}_{name}"]
            ax = self.axes[i // self.max_col, i % self.max_col]
            ax.lines[line_idx].set_data(np.arange(len(data)), data)
            ax.relim()
            ax.autoscale_view()

        self.force_draw()

    def on_fit_end(self, net):
        if self.savefig:
            self.fig.savefig(self.savename, bbox_inches="tight")
        if self.block_on_finish:
            plt.show(block=True)
        self.switch_normal(self.old_backend)

    def force_draw(self):
        self.fig.canvas.draw()
        self.fig.canvas.flush_events()

    def switch_qt5(self):
        if self.is_ipython:
            self.switch_magic("qt")
        else:
            self.switch_normal("Qt5Agg")

    def switch_normal(self, backend):
        mpl.use(backend, force=True)
        importlib.reload(sys.modules[self.pyplot_name])

    def switch_magic(self, backend):
        self.get_ipython().run_line_magic("matplotlib", backend)


class NetCheckpoint(Callback):
    def __init__(self, savepath, per_epoch=1):
        super().__init__()
        self.savepath = savepath
        self.per_epoch = per_epoch

    def state_dict(self):
        return {}

    def load_state_dict(self, state_dict: dict):
        pass

    def on_train_epoch_end(self, net):
        if not net._validate and (net._epoch - 1) % self.per_epoch == 0:
            psk.NeuralNetwork.save_class(net, self.savepath)

    def on_val_epoch_end(self, net):
        if net._validate and (net._epoch - 1) % self.per_epoch == 0:
            psk.NeuralNetwork.save_class(net, self.savepath)

    def on_fit_end(self, net: "psk.NeuralNetwork"):
        psk.NeuralNetwork.save_class(net, self.savepath)


class WeightCheckpoint(Callback):
    def __init__(self, tracked, mode, savepath=None, save_per_epoch=False):
        super().__init__()
        self._tally = Tally(recorded=tracked, mode=mode, best_epoch=-1, best_weights=None)
        self.savepath = savepath
        self.save_per_epoch = save_per_epoch

    def state_dict(self):
        return {"tally_state": self._tally.state_dict()}

    def load_state_dict(self, state_dict: dict):
        self._tally.load_state_dict(state_dict["tally_state"])

    @property
    def tracked(self):
        return self._tally.recorded

    @property
    def mode(self):
        return self._tally.mode

    @property
    def best_epoch(self):
        return self._tally.best_epoch

    @property
    def savefile(self):
        return os.path.basename(self.savepath)

    @property
    def best_weights(self):
        if self._tally.best_weights is not None:
            return self._tally.best_weights
        elif self.savepath is not None:
            return torch.load(self.savepath)
        else:
            raise RuntimeError("There are no best_weights loaded in RAM or saved to disk.")  # TODO: Return None instead?

    def on_train_epoch_end(self, net):
        if not net._validate:
            self._track(net)

    def on_val_epoch_end(self, net):
        if net._validate:
            self._track(net)

    def _track(self, net: "psk.NeuralNetwork"):
        track = net.history.track
        new_record = track[self._tally.recorded][-1]
        self._tally.evaluate_record(new_record=new_record,
                                    best_epoch=net._epoch,
                                    best_weights=copy.deepcopy(net.get_module_weights()))
        if self.save_per_epoch:
            self._save_weights(False)

    def on_fit_end(self, net):
        self._save_weights(True)

    def _save_weights(self, fit_end):
        if self.savepath is not None:
            if self._tally.best_weights is None:
                self._tally.best_weights = self.best_weights  # This can happen if we train the net a second time.
            torch.save(self._tally.best_weights, self.savepath)
            if fit_end:
                self._tally.best_weights = None  # No need to keep it in memory after saving.


class EarlyStopping(Callback):
    """
    Implements early stopping functionality to the added NeuralNetwork.
    It will monitor the given metric as `monitor` and if that metric does not improve
    `patience` times in a row, training will be stopped early.
    """
    def __init__(self, tracked: str, mode: str, patience: int = 20):
        super(EarlyStopping, self).__init__()
        self._tally = Tally(recorded=tracked, mode=mode, best_epoch=-1, best_weights=None)
        self.patience = patience
        self.current_patience = 0

    def state_dict(self):
        return {"tally_state": self._tally.state_dict(), "current_patience": self.current_patience}

    def load_state_dict(self, state_dict: dict):
        tally_state = state_dict.pop("tally_state")
        self.__dict__.update(state_dict)
        self._tally.load_state_dict(tally_state)

    @property
    def tracked(self):
        return self._tally.recorded

    @property
    def mode(self):
        return self._tally.mode

    @property
    def best_epoch(self):
        return self._tally.best_epoch

    @property
    def best_weights(self):
        return self._tally.best_weights

    def on_train_epoch_end(self, net):
        if not net._validate:
            self._track(net)

    def on_val_epoch_end(self, net):
        if net._validate:
            self._track(net)

    def _track(self, net: "psk.NeuralNetwork"):
        track = net.history.track
        new_record = track[self._tally.recorded][-1]
        is_better_record = self._tally.is_better_record(new_record)

        if is_better_record:
            self._tally.evaluate_record(new_record=new_record,
                                        best_epoch=net._epoch,
                                        best_weights=copy.deepcopy(net.get_module_weights()))

            self.current_patience = 0
        else:
            self.current_patience += 1
            if self.current_patience >= self.patience:
                net.keep_training = False


class LRScheduler(Callback):
    """
    Applies the given learning rate scheduler at the end of each epoch or batch.

    Parameters
    ----------
    lr_scheduler
        Scheduler to step.
    per_epoch : bool
        Whether we should step the scheduler every epoch or every batch.
    per_step : int
        Call lr_scheduler.step after this many steps, where a step is an epoch if per_epoch=True, or a batch if per_epoch=False.
    store_lrs : bool
        Keep track of all the learning rates.
    reset_on_fit_end : bool
        Does scheduler go to its initial state when fit ends?
    interval : int or 2-element array_like
        Steps when this scheduler is active, left-side inclusive, right-side exclusive, like [2, 5).
        This first step is step 0. By default, the interval is [0, -1), meaning it is always active.

        E.g. if interval is [2, 5) on a LinearLR scheduler going from 1 to 0 over 3 steps:
            step 0: no update,  lr = 1
            step 1: no update,  lr = 1
            step 2: update,     lr = 0.666
            step 3: update,     lr = 0.333
            step 4: update,     lr = 0.000
            step 5: no update,  lr = 0

        If int, it is the same as passing [0, interval).
    pass_metric : str, default: None
        Name of the metric to pass down to the scheduler on step, e.g. train_loss.
    """
    def __init__(self, lr_scheduler, per_epoch=True, per_step=None, store_lrs=False, reset_on_fit_end=True, reset_on_epoch_end=False, interval=None, pass_metric=None):
        super().__init__()
        self.lr_scheduler = lr_scheduler
        self.init_state_dict = lr_scheduler.state_dict()
        self.per_epoch = per_epoch
        self.per_step = per_step
        self.store_lrs = store_lrs
        self.reset_on_fit_end = reset_on_fit_end
        self.reset_on_epoch_end = reset_on_epoch_end
        self.interval = self._correct_interval(interval)
        self.pass_metric = pass_metric

        self.step_count = 0
        self.lrs = []
        if store_lrs:
            try:
                self.lrs.append(lr_scheduler.get_last_lr())
            except AttributeError:
                try:
                    self.lrs.append(self.lr_scheduler._last_lr)
                except AttributeError:
                    pass
                # Some bug with SequentialLR not having get_last_lr available before calling step.
                # Should be fixed in torch 1.12.1
                # Another bug with ReduceLROnPlateau not being a _LRScheduler, so it doesn't have get_last_lr()

    def state_dict(self):
        return {
            "step_count": self.step_count,
            "lrs": self.lrs,
            "lr_scheduler": self.lr_scheduler.state_dict()
        }

    def load_state_dict(self, state_dict: dict):
        lr_scheduler_state = state_dict.pop("lr_scheduler")
        self.__dict__.update(state_dict)
        self.lr_scheduler.load_state_dict(lr_scheduler_state)

    def on_train_batch_end(self, net):
        if not self.per_epoch:
            metric_index = net.history.key_index.get(self.pass_metric, None)
            metric = None
            if metric_index is not None:
                metric = net.history.epoch_metrics[metric_index] / net._batch  # batch mean if we update per batch!
            self.step_and_store_lr(metric)

    def on_train_epoch_end(self, net):
        if self.per_epoch:
            metric = net.history.track.get(self.pass_metric, [None])[-1]
            self.step_and_store_lr(metric)

        if self.reset_on_epoch_end:
            self.lr_scheduler.load_state_dict(self.init_state_dict)
            self.step_count = 0

    def on_fit_end(self, net):
        if self.reset_on_fit_end:
            self.lr_scheduler.load_state_dict(self.init_state_dict)
            self.step_count = 0

    def step_and_store_lr(self, metric=None):
        if self.should_step():
            self.lr_scheduler.step(metric)
            if self.store_lrs:
                # For some reason ReduceLROnPlateau is not a _LRScheduler, so it doesn't have get_last_lr()
                try:
                    self.lrs.append(self.lr_scheduler.get_last_lr())
                except AttributeError:
                    self.lrs.append(self.lr_scheduler._last_lr)

        self.step_count += 1

    def should_step(self):
        if self.interval[0] <= self.step_count < self.interval[1] and \
            (self.per_step is None or self.step_count % self.per_step == 0):
            return True
        return False

    def _correct_interval(self, interval):
        if interval is None:
            interval = [0, float("inf")]
        elif isinstance(interval, int):
            interval = [0, interval]
        elif interval[1] == -1:
            interval[1] = float("inf")
        return interval


class ImageOutputWriter(Callback):
    """
    Saves the batch input, output, and the ground truth as images every 'freq' batches.

    Parameters
    ----------
    save_path
        The folder to save in.
    freq : int
        Saves images every 'freq' batches.
    num_saved : int
        If there are too many images per batch, you can choose how many to save. None by default, which means save all the images.
    start : int
        The saved images are named as: "epoch_{poch}_batch_{batch}_{current:03d}.png". This variable determines the starting point of 'current'.
    clamp01 : bool
        Whether the images are clamp between 0 and 1 (applied before 'preprocess').
    create_path : bool
        Whether the given 'save_path' should be created by this class or not.
    preprocess : callable(img) -> img
        A callable that expects a PyTorch tensor of shape (C x H x W), detached and on CPU, and outputs the processed tensor of the same shape.
    """
    def __init__(self, save_path, freq, num_saved=None, start=0, clamp01=True, create_path=False, preprocess=None):
        super().__init__()
        self.save_path = save_path
        self.freq = freq
        self.num_saved = num_saved  # None means all the images in the batch
        self.clamp01 = clamp01
        self.current = start
        self.create_path = create_path
        self.preprocess = preprocess
        if create_path:
            if not os.path.exists(save_path):
                os.makedirs(save_path)

    def state_dict(self):
        return {}

    def load_state_dict(self, state_dict: dict):
        pass

    def identity(self, x):
        return x

    def on_train_batch_end(self, net: "psk.NeuralNetwork"):
        if net._batch % self.freq == 0:
            batch_out = to_safe_tensor(net._batch_out)
            batch_X, batch_y = net.unpack_fit_batch(net._batch_data)
            batch_X = to_safe_tensor(batch_X)
            batch_y = to_safe_tensor(batch_y)

            if self.clamp01:
                batch_out = torch.clamp(batch_out, 0, 1)

            for i, (img, X, y) in enumerate(zip(batch_out, batch_X, batch_y)):
                if self.num_saved is not None and i >= self.num_saved:
                    break
                
                if self.preprocess is not None:
                    img = self.preprocess(img)
                    X = self.preprocess(X)
                    y = self.preprocess(y)

                img = img.permute(1,2,0)
                X = X.permute(1,2,0)
                y = y.permute(1,2,0)

                plt.figure(figsize=(20,5))
                plt.subplot(1,3,1)
                plt.imshow(X)
                plt.xlabel(f"Input")
                plt.xticks([]); plt.yticks([])
                plt.subplot(1,3,2)
                plt.imshow(img)
                plt.xlabel(f"Output")
                plt.xticks([]); plt.yticks([])
                plt.subplot(1,3,3)
                plt.imshow(y)
                plt.xlabel(f"GT")
                plt.xticks([]); plt.yticks([])
                plt.savefig(osj(self.save_path, f"{self.current:04d}_epoch_{net._epoch}_batch_{net._batch}.png"), bbox_inches="tight")
                plt.close()
            
                self.current += 1


class Tracker(Callback):
    """
    Tracks the given metric as `tracked`.
    """
    def __init__(self, tracked: str, mode: str):
        super(Tracker, self).__init__()
        self._tally = Tally(recorded=tracked, mode=mode, best_epoch=-1)

    def state_dict(self):
        return {"tally_state": self._tally.state_dict()}

    def load_state_dict(self, state_dict):
        self._tally.load_state_dict(state_dict["tally_state"])

    @property
    def tracked(self):
        return self._tally.recorded

    @property
    def mode(self):
        return self._tally.mode

    @property
    def best_epoch(self):
        return self._tally.best_epoch

    def on_train_epoch_end(self, net):
        if not net._validate:
            self._track(net)

    def on_val_epoch_end(self, net):
        if net._validate:
            self._track(net)

    def _track(self, net):
        track = net.history.track
        self._tally.evaluate_record(track[self._tally.recorded][-1], best_epoch=net._epoch)


class CallbackInfo(Callback):
    """
    Collects and prints the ``neural_network`` parameters at the first time the callback function is called.
    Use this to get an intuition on which parameters will be available on each callback function.
    """
    def __init__(self):
        self.name = "CallbackInfo"
        self.called = np.zeros(15, dtype=bool)
        self.parameters = {}

    def state_dict(self):
        return {}

    def on_fit_begin(self, net):
        if not self.called[0]:
            self.parameters["on_fit_begin"] = [k for k, v in net.__dict__.items() if v is not None]
            print("on_fit_begin: [")
            for k, v in net.__dict__.items():
                if v is not None:
                    print("\t", k)
            print("]")
            self.called[0] = True

    def on_fit_end(self, net):
        if not self.called[1]:
            self.parameters["on_fit_end"] = [k for k, v in net.__dict__.items() if v is not None]
            print("on_fit_end: [")
            for k, v in net.__dict__.items():
                if v is not None:
                    print("\t", k)
            print("]")
            self.called[1] = True

    def on_fit_interrupted(self, net):
        if not self.called[2]:
            self.parameters["on_fit_interrupted"] = [k for k, v in net.__dict__.items() if v is not None]
            print("on_fit_interrupted: [")
            for k, v in net.__dict__.items():
                if v is not None:
                    print("\t", k)
            print("]")
            self.called[2] = True

    def on_train_epoch_begin(self, net):
        if not self.called[3]:
            self.parameters["on_train_epoch_begin"] = [k for k, v in net.__dict__.items() if v is not None]
            print("on_train_epoch_begin: [")
            for k, v in net.__dict__.items():
                if v is not None:
                    print("\t", k)
            print("]")
            self.called[3] = True

    def on_train_epoch_end(self, net):
        if not self.called[4]:
            self.parameters["on_train_epoch_end"] = [k for k, v in net.__dict__.items() if v is not None]
            print("on_train_epoch_end: [")
            for k, v in net.__dict__.items():
                if v is not None:
                    print("\t", k)
            print("]")
            self.called[4] = True

    def on_train_batch_begin(self, net):
        if not self.called[5]:
            self.parameters["on_train_batch_begin"] = [k for k, v in net.__dict__.items() if v is not None]
            print("on_train_batch_begin: [")
            for k, v in net.__dict__.items():
                if v is not None:
                    print("\t", k)
            print("]")
            self.called[5] = True

    def on_train_batch_end(self, net):
        if not self.called[6]:
            self.parameters["on_train_batch_end"] = [k for k, v in net.__dict__.items() if v is not None]
            print("on_train_batch_end: [")
            for k, v in net.__dict__.items():
                if v is not None:
                    print("\t", k)
            print("]")
            self.called[6] = True

    def on_val_epoch_begin(self, net):
        if not self.called[7]:
            self.parameters["on_val_epoch_begin"] = [k for k, v in net.__dict__.items() if v is not None]
            print("on_val_epoch_begin: [")
            for k, v in net.__dict__.items():
                if v is not None:
                    print("\t", k)
            print("]")
            self.called[7] = True

    def on_val_epoch_end(self, net):
        if not self.called[8]:
            self.parameters["on_val_epoch_end"] = [k for k, v in net.__dict__.items() if v is not None]
            print("on_val_epoch_end: [")
            for k, v in net.__dict__.items():
                if v is not None:
                    print("\t", k)
            print("]")
            self.called[8] = True

    def on_val_batch_begin(self, net):
        if not self.called[9]:
            self.parameters["on_val_batch_begin"] = [k for k, v in net.__dict__.items() if v is not None]
            print("on_val_batch_begin: [")
            for k, v in net.__dict__.items():
                if v is not None:
                    print("\t", k)
            print("]")
            self.called[9] = True

    def on_val_batch_end(self, net):
        if not self.called[10]:
            self.parameters["on_val_batch_end"] = [k for k, v in net.__dict__.items() if v is not None]
            print("on_val_batch_end: [")
            for k, v in net.__dict__.items():
                if v is not None:
                    print("\t", k)
            print("]")
            self.called[10] = True

    def on_predict_begin(self, net):
        if not self.called[11]:
            self.parameters["on_predict_begin"] = [k for k, v in net.__dict__.items() if v is not None]
            print("on_predict_begin: [")
            for k, v in net.__dict__.items():
                if v is not None:
                    print("\t", k)
            print("]")
            self.called[11] = True

    def on_predict_end(self, net):
        if not self.called[12]:
            self.parameters["on_predict_end"] = [k for k, v in net.__dict__.items() if v is not None]
            print("on_predict_end: [")
            for k, v in net.__dict__.items():
                if v is not None:
                    print("\t", k)
            print("]")
            self.called[12] = True

    def on_predict_proba_begin(self, net):
        if not self.called[13]:
            self.parameters["on_predict_proba_begin"] = [k for k, v in net.__dict__.items() if v is not None]
            print("on_predict_proba_begin: [")
            for k, v in net.__dict__.items():
                if v is not None:
                    print("\t", k)
            print("]")
            self.called[13] = True

    def on_predict_proba_end(self, net):
        if not self.called[14]:
            self.parameters["on_predict_proba_end"] = [k for k, v in net.__dict__.items() if v is not None]
            print("on_predict_proba_end: [")
            for k, v in net.__dict__.items():
                if v is not None:
                    print("\t", k)
            print("]")
            self.called[14] = True