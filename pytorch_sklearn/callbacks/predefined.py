# __all__ = ["History", "Verbose", "EarlyStopping", "LossPlotter", "WeightCheckpoint", "LRScheduler", "Tracker", "CallbackInfo"]

from pytorch_sklearn.callbacks import Callback
from pytorch_sklearn.utils.func_utils import to_safe_tensor
from pytorch_sklearn.callbacks.utils import Tally
from pytorch_sklearn.utils.model_utils import get_receptive_field
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

# LossPlotter
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
        # Register losses returned from fit_batch
        for i, name in enumerate(net.loss_names):
            if f"{pass_type}_{name}" not in self.track:
                self.track[f"{pass_type}_{name}"] = []
                self.key_index[f"{pass_type}_{name}"] = i

        # Register metrics
        for i, name in enumerate(net._metrics.keys(), start=len(net.loss_names)):
            if f"{pass_type}_{name}" not in self.track:
                self.track[f"{pass_type}_{name}"] = []
                self.key_index[f"{pass_type}_{name}"] = i

    def on_fit_begin(self, net):
        self.session += 1
        self.num_metrics = len(net._metrics) + len(net.loss_names)
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

        # Save losses returned from fit_batch
        for i, name in enumerate(net.loss_names):
            self.track[f"{net._pass_type}_{name}"].append(self.epoch_metrics[i])
            
        # Save metrics
        for i, name in enumerate(net._metrics.keys(), start=len(net.loss_names)):
            self.track[f"{net._pass_type}_{name}"].append(self.epoch_metrics[i])

    def on_train_batch_end(self, net):
        self._calculate_metrics(net)

    def on_val_batch_end(self, net):
        self._calculate_metrics(net)

    def _calculate_metrics(self, net: "psk.NeuralNetwork"):
        with torch.no_grad():
            # batch_out = to_safe_tensor(net._batch_out)
            # TODO: A better fix for this - fails when user returns a tuple even if there is a single loss.
            batch_loss = net._batch_loss if len(net.loss_names) > 1 else (net._batch_loss, )

            # Calculate losses returned from fit_batch
            for i, name in enumerate(net.loss_names):
                self.epoch_metrics[i] += batch_loss[i].item()
                
            # Calculate metrics
            for i, metric in enumerate(net._metrics.values(), start=len(net.loss_names)):
                self.epoch_metrics[i] += metric(net._batch_out, net._batch_data)  # metric must return a scalar or a tensor with a single element


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

    def _save_metrics(self, net):
        self.epoch_metrics = self.epoch_metrics / net._num_batches
        self.track[f"{net._pass_type}_G_A_loss"].append(self.epoch_metrics[0])
        self.track[f"{net._pass_type}_G_B_loss"].append(self.epoch_metrics[1])
        self.track[f"{net._pass_type}_D_A_loss"].append(self.epoch_metrics[2])
        self.track[f"{net._pass_type}_D_B_loss"].append(self.epoch_metrics[3])
        for i, name in enumerate(net._metrics.keys(), start=4):
            self.track[f"{net._pass_type}_{name}"].append(self.epoch_metrics[i])

    def _calculate_metrics(self, net: "psk.NeuralNetwork"):
        with torch.no_grad():
            # batch_out = to_safe_tensor(net._batch_out)
            self.epoch_metrics[0:4] += [loss.item() for loss in net._batch_loss]  # 0:4 diff from History
            for i, metric in enumerate(net._metrics.values(), start=4):  # 1 -> 4 diff from History
                self.epoch_metrics[i] += metric(net._batch_out, net._batch_data)


class Verbose(Callback):
    def __init__(self, verbose=3, per_batch=True, notebook=False, old_version=False):
        """
        Prints the following training information:
            - Current Epoch / Total Epochs
            - Current Batch / Total Batches
            - Loss
            - Metrics
            - Total Time + ETA

        Parameters
        ----------
        verbose:
            Controls how much is printed. Higher levels include the info from lower levels.
            The following levels are valid:
                0: Only Batch or Epoch
                1: Loss
                2: Metrics
                3: Total Time + ETA
        per_batch:
            Whether to print the info per batch or per epoch.
        notebook:
            Whether the training is done in a Jupyter Notebook or not.
        old_version:
            For development only, ignore.
        """
        super().__init__()
        self.name = "Verbose"
        self.verbose = verbose
        self.per_batch = per_batch
        self.per_epoch = not per_batch
        self.notebook = notebook
        self.old_version = old_version

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

        if self.verbose >= 3 and self.per_epoch:
            self.start_time = time.perf_counter()

    def on_train_epoch_begin(self, net):
        if self.per_batch:
            print(f"Epoch {self.prev_epochs + net._epoch}/{self.prev_epochs + net._max_epochs}")

            if self.verbose >= 3:
                self.start_time = time.perf_counter()

    def on_train_epoch_end(self, net):
        if self.per_epoch:
            # Fill print data
            opt = None
            if self.verbose >= 1:
                # Print losses returned from fit_batch
                opt = []
                for i, name in enumerate(net.loss_names):
                    lossval = net.history.track[f"{net._pass_type}_{name}"][-1]
                    opt.append(f"{net._pass_type}_{name}: {lossval:.3f}")
            if self.verbose >= 2:
                # Print metrics
                opt.extend([f"{net._pass_type}_{name}: {net.history.track[f'{net._pass_type}_{name}'][-1]:.3f}" for name in net._metrics.keys()])
            if self.verbose >= 3:
                # Print time info
                self.end_time = time.perf_counter()
                self.total_time = self.end_time - self.start_time
                self.rem_time = ((net._max_epochs - net._epoch) * self.total_time) / net._max_epochs
                opt.extend([f"Time: {self.total_time:.2f}", f"ETA: {self.rem_time:.2f}"])
            print_progress(self.prev_epochs + net._epoch, self.prev_epochs + net._max_epochs, pre='Epoch ', opt=opt, notebook=self.notebook, old_version=self.old_version)

    def on_val_epoch_begin(self, net):
        if self.verbose >= 3 and self.per_batch:
            self.start_time = time.perf_counter()

    def on_train_batch_end(self, net):
        if self.per_batch:
            self._print(net)

    def on_val_batch_end(self, net):
        if self.per_batch:
            self._print(net)

    def _print(self, net):
        # Calculate data
        epoch_metrics = net.history.epoch_metrics / net._batch  # mean batch loss

        # Fill print data
        opt = None
        if self.verbose >= 1:
            # Print losses returned from fit_batch
            opt = [f"{net._pass_type}_{name}: {epoch_metrics[i]:.3f}" for i, name in enumerate(net.loss_names)]
        if self.verbose >= 2:
            # Print metrics
            opt.extend([f"{net._pass_type}_{name}: {epoch_metrics[net.history.key_index[f'{net._pass_type}_{name}']]:.3f}" for name in net._metrics.keys()])
        if self.verbose >= 3:
            # Print time info
            self.end_time = time.perf_counter()
            self.total_time = self.end_time - self.start_time
            self.rem_time = ((net._num_batches - net._batch) * self.total_time) / net._batch
            opt.extend([f"Time: {self.total_time:.2f}", f"ETA: {self.rem_time:.2f}"])
        print_progress(net._batch, net._num_batches, opt=opt, notebook=self.notebook, old_version=self.old_version)



class BatchHistory(Callback):
    def __init__(self):
        super().__init__()
        self.name = "BatchHistory"
        self.track = {}
        self.sessions = []
        self.key_index = {}  # given key in track, return index in epoch_metrics.
        self.num_metrics = -1
        self.session = -1
        self.last_epoch_metrics: np.ndarray

    def init_track(self, net, pass_type):
        # Register losses returned from fit_batch
        for i, name in enumerate(net.loss_names):
            if f"{pass_type}_{name}" not in self.track:
                self.track[f"{pass_type}_{name}"] = []
                self.key_index[f"{pass_type}_{name}"] = i

        # Register metrics
        for i, name in enumerate(net._metrics.keys(), start=len(net.loss_names)):
            if f"{pass_type}_{name}" not in self.track:
                self.track[f"{pass_type}_{name}"] = []
                self.key_index[f"{pass_type}_{name}"] = i

    def on_fit_begin(self, net):
        self.session += 1
        self.num_metrics = len(net._metrics) + len(net.loss_names)
        self.init_track(net, "train")
        if net._validate:
            self.init_track(net, "val")
        session_start = len(self.track[next(iter(self.track))]) + 1  # new session starts at epoch = len(first_key) + 1
        self.sessions.append(session_start)

    def on_train_epoch_begin(self, net):
        self.last_epoch_metrics = np.zeros(self.num_metrics)  # reset last epoch metrics

    def on_val_epoch_begin(self, net):
        self.last_epoch_metrics = np.zeros(self.num_metrics)  # reset last epoch metrics

    def on_train_batch_end(self, net):
        self._save_batch_data(net)

    def on_val_batch_end(self, net):
        self._save_batch_data(net)

    def _save_batch_data(self, net):
        # Calculate data
        epoch_metrics = net.history.epoch_metrics - self.last_epoch_metrics  # current batch loss
        self.last_epoch_metrics = net.history.epoch_metrics.copy()  # save for next batch

        # Save losses returned from fit_batch
        for i, name in enumerate(net.loss_names):
            self.track[f"{net._pass_type}_{name}"].append(epoch_metrics[i])
            
        # Save metrics
        for i, name in enumerate(net._metrics.keys(), start=len(net.loss_names)):
            self.track[f"{net._pass_type}_{name}"].append(epoch_metrics[i])





class LossPlotter(Callback):
    def __init__(self,
                 per_step=1,
                 max_col=1,
                 block_on_finish=False,
                 savefig=False,
                 savename=None,
                 plot_kw=None,
                 figure_kw=None,
                 interactive=False,
                 new_backend="Qt5Agg",
                 pyplot_name="matplotlib.pyplot",):
        super().__init__()
        self.get_ipython = __import__('IPython', globals(), locals(), ['get_ipython'], 0).get_ipython

        self.per_step = per_step
        self.max_col = max_col
        self.block_on_finish = block_on_finish
        self.savefig = savefig
        self.savename = savename
        if self.savefig:
            assert self.savename is not None, "You must provide a savename."
        self.plot_kw = {} if plot_kw is None else plot_kw
        self.figure_kw = {} if figure_kw is None else figure_kw
        self.interactive = interactive
        self.new_backend = new_backend
        self.old_backend = mpl.get_backend()
        self.pyplot_name = pyplot_name
        self.is_ipython = self.get_ipython() is not None

        # on_fit_begin
        self.fig = None
        self.axes = None

    def state_dict(self):
        return {"fig": self.fig, "axes": self.axes}

    def on_fit_begin(self, net):
        # self.switch_qt5()
        if self.interactive:
            plt.ion()  # turn on interactive mode

        num_metrics = net.history.num_metrics
        nrows = int(np.ceil(num_metrics / self.max_col))

        self.fig, self.axes = plt.subplots(nrows, self.max_col, sharex="all", squeeze=False, **self.figure_kw)

        # Delete unused subplots
        for i in range(num_metrics, nrows * self.max_col):
            self.fig.delaxes(self.axes[i // self.max_col, i % self.max_col])
            # self.axes[i // self.max_col, i % self.max_col].set_visible(False)

        # Define empty lines for loss line
        for i, name in enumerate(net.loss_names):
            ax = self.axes[i // self.max_col, i % self.max_col]
            ax.set_title(f"{name}")
            ax.plot([], [], "-o", label=f"train {name}", **self.plot_kw)
            if net._validate:
                ax.plot([], [], "-o", label=f"val {name}", **self.plot_kw)
            ax.legend()

        # Define empty lines for other metric lines
        for i, name in enumerate(net._metrics.keys(), start=len(net.loss_names)):
            ax = self.axes[i // self.max_col, i % self.max_col]
            ax.set_title(f"{name.capitalize()}")
            ax.plot([], [], "-o", label=f"train {name}", **self.plot_kw)
            if net._validate:
                ax.plot([], [], "-o", label=f"val {name}", **self.plot_kw)
            ax.legend()

        # h, l = self.axes[0, 0].get_legend_handles_labels()
        # self.fig.legend(l)

    def on_train_epoch_end(self, net):
        if net._epoch % self.per_step == 0:
            self.plot_metrics(net)
            if self.savefig:
                self.fig.savefig(self.savename, bbox_inches="tight")

    def on_val_epoch_end(self, net):
        if net._epoch % self.per_step == 0:
            self.plot_metrics(net)
            if self.savefig:
                self.fig.savefig(self.savename, bbox_inches="tight")

    def plot_metrics(self, net):
        track = net.history.track
        line_idx = 0 if net._pass_type == "train" else 1

        # Change plot for loss line
        for i, name in enumerate(net.loss_names):
            data = track[f"{net._pass_type}_{name}"]
            ax = self.axes[i // self.max_col, i % self.max_col]
            ax.lines[line_idx].set_data(np.arange(len(data)), data)
            ax.relim()
            ax.autoscale_view()

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
        # self.switch_normal(self.old_backend)

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
        if not net._validate and (net._epoch % self.per_epoch == 0 or net._epoch == 1):
            psk.NeuralNetwork.save_class(net, self.savepath)

    def on_val_epoch_end(self, net):
        if net._validate and (net._epoch % self.per_epoch == 0 or net._epoch == 1):
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
    If threshold is passed, then the training is stopped when the tracked metric gets better than the threshold.
    In this case, patience is ignored.
    """
    def __init__(self, tracked: str, mode: str, patience: int = 20, threshold: float = None):
        super(EarlyStopping, self).__init__()
        self._tally = Tally(recorded=tracked, mode=mode, best_epoch=-1, best_weights=None)
        self.patience = patience
        self.threshold = threshold
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
            if self.threshold is None:
                self.current_patience += 1
                if self.current_patience >= self.patience:
                    net.keep_training = False

        if self.threshold is not None:
            if self._tally.mode == 'max':
                if new_record > self.threshold:
                    net.keep_training = False
            elif self._tally.mode == 'min':
                if new_record < self.threshold:
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




class LRSchedulingEarlyStopper(Callback):
    """
    Monitors the given metric and applies the learning rate scheduler if the metric doesn't improve for `patience` times in a row.
    When the scheduler is stepped 'max_steps' times and after that the metric doesn't improve for 'patience' times, training is stopped.

    Parameters
    ----------
    lr_scheduler
        Scheduler to step.
    max_steps : int
        The lr will be stepped at most this many times, after which training will be stopped.
    tracked : str
        The metric to monitor.
    mode : str
        Which way the metric should improve, 'min' or 'max'.
    patience : int
        The number of times the metric can not improve before lr scheduler is stepped or training is stopped.
    patience_multiplier : int
        When the lr scheduler is stepped, the patience is multiplied by this number. This way patience decreases as the metric fails to improve.
        When the metric improves, the patience is reset to the original value.
    min_patience : int
        The minimum patience value. Maximum is always the original value.
    relevance_threshold : float
        For a new metric to be considered better, it must be better than the previous metric by this amount.
        E.g. mode=max, relevance_threshold=0.1, relevance_threshold_mode=abs then new_metric must be at least 0.1 larger than the previous metric to be considered better.
    relevance_threshold_mode : str
        'abs' or 'rel'. If 'abs', the relevance_threshold is an absolute value. If 'rel', the relevance_threshold is relative to current best.
        E.g. mode=max, and this is 'rel', then new_metric > best_metric * (1 + relevance_threshold) is considered better.
        E.g. mode=max, and this is 'abs', then new_metric > best_metric + relevance_threshold is considered better.
    """
    def __init__(self, lr_scheduler, max_steps: int, tracked: str, mode: str, patience: int = 20, patience_multiplier=1, min_patience=0, relevance_threshold=0, relevance_threshold_mode='rel'):
        super().__init__()
        self.lr_scheduler = lr_scheduler
        self.max_steps = max_steps
        self._tally = Tally(recorded=tracked, mode=mode, threshold=relevance_threshold, threshold_mode=relevance_threshold_mode, best_epoch=-1, best_weights=None)
        self.patience = patience
        self.patience_multiplier = patience_multiplier
        self.max_patience = patience
        self.min_patience = min_patience

        self.current_step = 0
        self.current_patience = 0

    def state_dict(self):
        return {
            "current_step": self.current_step,
            "current_patience": self.current_patience,
            "tally_state": self._tally.state_dict(),
            "lr_scheduler": self.lr_scheduler.state_dict()
        }

    def load_state_dict(self, state_dict: dict):
        lr_scheduler_state = state_dict.pop("lr_scheduler")
        tally_state = state_dict.pop("tally_state")
        self.__dict__.update(state_dict)
        self.lr_scheduler.load_state_dict(lr_scheduler_state)
        self._tally.load_state_dict(tally_state)

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
            self.patience = self.max_patience
        else:
            # If the record doesn't improve, update patience
            self.current_patience += 1

            # If we have reached the patience limit
            if self.current_patience >= self.patience:
                # If we have reached the max steps, stop training
                if self.current_step >= self.max_steps:
                    net.keep_training = False
                # Otherwise, step the lr scheduler, update step and reset patience
                else:
                    self.lr_scheduler.step()
                    self.current_step += 1
                    self.current_patience = 0
                    self.patience = max(self.min_patience, self.patience * self.patience_multiplier)



class ReceptiveFieldVisualizer(Callback):
    def __init__(self, save_path: str, dummy_input: torch.Tensor, target_output=None, per_epoch=True, create_path=False):
        """
        Visualizes the receptive field of the model by calculating the gradient of the center pixel with respect to the input.
        Assumes the model takes an image as input. The receptive field is absoluted and normalized.

        Parameters
        ----------
        dummy_input
            The input to pass through the model to calculate the receptive field. If unsure, use torch.ones(1, C, H, W) * 0.01 where C,H,W are suitable for the model.
        target_output
            If the model has multiple outputs, this specifies which output to calculate the receptive field for.
        per_epoch
            If True, saves the receptive field at the end of every epoch. If False, saves the receptive field at the end of every batch.
        """
        super().__init__()
        assert len(dummy_input.shape) == 4, "dummy_input must be of shape (N, C, H, W)"
        assert dummy_input.shape[0] == 1, "dummy_input must have a batch size of 1"
        assert dummy_input.shape[1] == 1 or dummy_input.shape[1] == 3, "dummy_input must be Grayscale or RGB"

        self.save_path = save_path
        self.dummy_input = dummy_input
        self.cmap = "gray" if dummy_input.shape[1] == 1 else "viridis"
        self.per_epoch = per_epoch
        self.create_path = create_path
        self.target_output = target_output
        if create_path:
            if not os.path.exists(save_path):
                os.makedirs(save_path)

        # Previous epochs if this model has stopped and resumed training
        self.prev_epochs = 0

    # Saving/Loading
    def state_dict(self):
        return {}

    def load_state_dict(self, state_dict: dict):
        pass

    def on_fit_begin(self, net):
        try: self.prev_epochs = len(net.history.track[next(iter(net.history.track))])
        except: pass

    def on_train_epoch_end(self, net: "psk.NeuralNetwork"):
        self.dummy_input = self.dummy_input.to(net._device)
        rf = get_receptive_field(self.dummy_input, net.module, absnorm=True, target_output=self.target_output)
        rf = rf.detach().cpu()
        plt.figure(figsize=(10,10))
        plt.imshow(rf[0].permute(1, 2, 0), cmap="gray")
        plt.savefig(osj(self.save_path, f"rf_ep{net._epoch + self.prev_epochs:04d}.png"), bbox_inches="tight")
        plt.close()


class ImageOutputWriter(Callback):
    def __init__(self, save_path, freq, num_saved=None, start=0, clamp01=True, create_path=False, preprocess=None):
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