import math

import numpy as np
import torch
from torch.optim import Optimizer
import warnings

from pytorch_sklearn.utils.class_utils import set_properties


class OptimizerGroupFilter(Optimizer):
    """
    Workaround to be able to specify which groups a learning rate scheduler should update. This will filter and send down
    the parameters of the given optimizer to learning rate schedulers. This class is experimental.

    As of writing this, learning rate schedulers seem to only query these variables:
        Optimizer.param_groups
        Optimizer.defaults
        Optimizer.step

    Parameters
    ----------
    optimizer : Optimizer
        The actual optimizer that should be updated.
    groups : array_like of int
        Which groups are returned when self.param_groups is called. If None, all of them are returned.

    Currently, PyTorch has no way of specifying which parameter groups a learning rate schedulers should update.
    While some learning rate schedulers allow multiple parameter groups to be updated with different scalars, not all of them do,
    and even then this is not the same thing is as NOT updating those groups.
    Another solution to this would be to use different optimizers instead of different parameter groups, but this is not easily scalable.
    """
    def __init__(self, optimizer, groups=None):
        # super().__init__([torch.zeros(1)], {})
        self._optimizer = optimizer
        self._groups = list(range(len(self._optimizer.param_groups))) if groups is None else groups

        self._param_groups = []
        self._defaults = {}

    @property
    def param_groups(self):
        return [group for i, group in enumerate(self._optimizer.param_groups) if i in self._groups]

    @param_groups.setter
    def param_groups(self, v):
        self._param_groups = v

    @property
    def defaults(self):
        return self._optimizer.defaults

    @defaults.setter
    def defaults(self, v):
        self._defaults = v

    def step(self, *args, **kwargs):
        return self._optimizer.step(*args, **kwargs)


class Tally:
    """
    Tally a given record's best state.
    """
    def __init__(self, recorded: str, mode: str, **kwargs):
        self.__dict__.update(kwargs)
        self.recorded = recorded
        self.mode = mode
        self.best_record = -np.Inf if mode == "max" else np.Inf

    def is_better_record(self, new_record):
        if self.mode == "max":
            return new_record > self.best_record
        return new_record < self.best_record

    def evaluate_record(self, new_record, **kwargs):
        """
        Check if ``new_metric`` is better than ``self.best_metric``.
        If it is, update this class's properties with ``**kwargs``.
        """
        if self.is_better_record(new_record):
            self.best_record = new_record
            self.__dict__.update(**kwargs)


class DynamicLRScheduler(object):
    from enum import Enum

    class Trend(Enum):
        NONE = 0,
        OSCILLATING = 1,
        PLATEAU = 2,
        LINEAR = 3

    def __init__(self,
                 optimizer,
                 mode='min',
                 down_factor=1,
                 up_factor=1,
                 target=None,
                 target_close_thresh=0,
                 target_distance="linear",
                 plateau_after=10,
                 threshold=1e-4,
                 threshold_mode='rel',
                 cooldown=0,
                 displacement_samples=10,
                 deciding_samples=3,
                 oscillation_thresh=0.008,
                 avg_step_thresh=0.01,
                 cum_disp_thresh=0.015,
                 slope_thresh=0.0001,
                 min_lr=0,
                 max_lr=None,
                 eps=1e-8,
                 verbose=False
                 ):
        from collections import defaultdict

        if not isinstance(optimizer, Optimizer):
            raise TypeError(f"{type(optimizer).__name__} is not an Optimizer")

        if up_factor > 1 and max_lr is None:
            raise ValueError("When up factor > 1, max_lr must be passed.")

        if mode not in {'min', 'max'}:
            raise ValueError('mode ' + mode + ' is unknown!')

        if threshold_mode not in {'rel', 'abs'}:
            raise ValueError('threshold mode ' + threshold_mode + ' is unknown!')

        if not np.isscalar(min_lr) and hasattr(min_lr, "__len__"):
            if len(min_lr) != len(optimizer.param_groups):
                raise ValueError(f"Expected {len(optimizer.param_groups)} min_lrs, got {len(min_lr)}.")
            self.min_lrs = list(min_lr)
        else:
            self.min_lrs = [min_lr] * len(optimizer.param_groups)

        if not np.isscalar(max_lr) and hasattr(max_lr, "__len__"):
            if len(max_lr) != len(optimizer.param_groups):
                raise ValueError(f"Expected {len(optimizer.param_groups)} min_lrs, got {len(max_lr)}.")
            self.max_lrs = list(max_lr)
        else:
            self.max_lrs = [max_lr] * len(optimizer.param_groups)

        self.optimizer = optimizer
        self.mode = mode
        self.down_factor = down_factor
        self.up_factor = up_factor
        self.target = target
        self.target_close_thresh = target_close_thresh
        self.target_distance = target_distance
        self.plateau_after = plateau_after
        self.threshold = threshold
        self.threshold_mode = threshold_mode
        self.cooldown = cooldown
        self.displacement_samples = displacement_samples
        self.deciding_samples = deciding_samples
        self.oscillation_thresh = oscillation_thresh
        self.avg_step_thresh = avg_step_thresh
        self.cum_disp_thresh = cum_disp_thresh
        self.slope_thresh = slope_thresh
        self.eps = eps
        self.verbose = verbose

        if self.mode == "min":
            self.best = np.inf
        elif self.mode == "max":
            self.best = -np.inf

        self.metrics = []
        self.analysis = defaultdict(list)
        self.decisions = []
        self.cooldown_counter = 0
        self.num_bad_steps = 0
        self.last_step = 0
        self._last_lr = [group['lr'] for group in self.optimizer.param_groups]

    def step(self, metric):
        # convert `metrics` to float, in case it's a zero-dim Tensor
        current = float(metric)
        self.metrics.append(current)
        self.last_step += 1

        if len(self.metrics) >= self.displacement_samples:
            self._analyze_metrics()

            if self.cooldown_counter > 0:
                self.cooldown_counter -= 1
            else:
                if self._try_update_lr(current):
                    self.cooldown_counter = self.cooldown
        else:
            self.decisions.append(self.Trend.NONE)

        self._last_lr = [group['lr'] for group in self.optimizer.param_groups]

    def _try_update_lr(self, current):
        # avg = np.mean(self.analysis["avg"][-self.deciding_samples:])
        avg_abs = np.mean(self.analysis["avg_abs"][-self.deciding_samples:])
        # cum = np.mean(self.analysis["cum"][-self.deciding_samples:])
        avg_cum = np.mean(self.analysis["avg_cum"][-self.deciding_samples:])
        std = np.mean(self.analysis["std"][-self.deciding_samples:])
        slope = np.mean(self.analysis["slope"][-self.deciding_samples:])

        did_update = False

        if std > self.oscillation_thresh and avg_abs > self.avg_step_thresh:
            self.decisions.append(self.Trend.OSCILLATING)
            self._update_lr(self.last_step, self.down_factor)
            did_update = True
        elif avg_abs < self.avg_step_thresh and np.abs(avg_cum) < self.cum_disp_thresh and np.abs(slope) < self.slope_thresh:
            self.decisions.append(self.Trend.PLATEAU)
            self._update_lr(self.last_step, self.down_factor)
            did_update = True
        elif avg_abs < self.avg_step_thresh and np.abs(avg_cum) > self.cum_disp_thresh and np.abs(slope) > self.slope_thresh:
            self.decisions.append(self.Trend.LINEAR)
            if self._is_target_close(current):
                self._update_lr(self.last_step, self.down_factor)
            else:
                self._update_lr(self.last_step, self.up_factor)
            did_update = True
        else:
            self.decisions.append(self.Trend.NONE)

        return did_update

    def _update_lr(self, step, factor):
        for i, param_group in enumerate(self.optimizer.param_groups):
            old_lr = float(param_group['lr'])
            new_lr = min(max(old_lr * factor, self.min_lrs[i]), self.max_lrs[i])
            if abs(old_lr - new_lr) > self.eps:
                param_group['lr'] = new_lr
                if self.verbose:
                    print(f"Step {step}: updating learning rate of group {i} from {old_lr:.4e} to {new_lr:.4e}.")

    def _analyze_metrics(self):
        metrics = np.array(self.metrics[-self.displacement_samples:])
        disp = np.diff(metrics)

        avg_disp = disp.mean()
        avg_abs_disp = np.abs(disp).mean()
        cum_disp = (metrics - metrics[-1]).sum()
        avg_cum_disp = (metrics - metrics[-1]).mean()
        std_disp = disp.std()
        slope = np.polyfit(np.arange(len(metrics)), metrics, 1)[0]

        self.analysis["avg"].append(avg_disp)
        self.analysis["avg_abs"].append(avg_abs_disp)
        self.analysis["cum"].append(cum_disp)
        self.analysis["avg_cum"].append(avg_cum_disp)
        self.analysis["std"].append(std_disp)
        self.analysis["slope"].append(slope)

    def _is_target_close(self, current):
        distance = self._distance_to_target(current)
        if distance <= self.target_close_thresh:
            return True
        return False

    def _distance_to_target(self, current):
        if self.target is None:
            return 0

        if isinstance(self.target_distance, type(lambda x: x)):  # is function
            return self.target_distance(current, self.target)

        if self.target_distance == "linear":
            return np.abs(self.target - current)

    def _is_better(self, current, best):
        if self.mode == 'min' and self.threshold_mode == 'rel':
            rel_epsilon = 1. - self.threshold
            return current < best * rel_epsilon

        elif self.mode == 'max' and self.threshold_mode == 'rel':
            rel_epsilon = 1. + self.threshold
            return current > best * rel_epsilon

        elif self.mode == 'min' and self.threshold_mode == 'abs':
            return current < best - self.threshold

        elif self.mode == 'max' and self.threshold_mode == 'abs':
            return current > best + self.threshold

    def get_last_lr(self):
        return self._last_lr

    def plot_decision(self, detailed=False, fig_kw=None):
        import matplotlib.pyplot as plt

        fig_kw = dict(figsize=(20, 15), dpi=150) if fig_kw is None else fig_kw

        metrics = np.array(self.metrics)
        decisions = np.array(self.decisions)
        nones = np.where(decisions == self.Trend.NONE)[0]
        oscillating = np.where(decisions == self.Trend.OSCILLATING)[0]
        plateau = np.where(decisions == self.Trend.PLATEAU)[0]
        linear = np.where(decisions == self.Trend.LINEAR)[0]

        if not detailed:
            for i, el in enumerate(self._consecutive(nones)):
                plt.plot(el, metrics[el], color="k")
            for i, el in enumerate(self._consecutive(oscillating)):
                plt.plot(el, metrics[el], color="C0", label="oscillating" if i == 0 else "")
            for i, el in enumerate(self._consecutive(plateau)):
                plt.plot(el, metrics[el], color="C1", label="plateau" if i == 0 else "")
            for i, el in enumerate(self._consecutive(linear)):
                plt.plot(el, metrics[el], color="C2", label="linear" if i == 0 else "")
            plt.legend()
        else:
            from matplotlib.gridspec import GridSpec
            fig = plt.figure(constrained_layout=True, **fig_kw)
            gs = GridSpec(4, 2, figure=fig)
            ax = fig.add_subplot(gs[0, :])
            for i, el in enumerate(self._consecutive(nones)):
                ax.plot(el, metrics[el], color="k")
            for i, el in enumerate(self._consecutive(oscillating)):
                ax.plot(el, metrics[el], color="C0", label="oscillating" if i == 0 else "")
            for i, el in enumerate(self._consecutive(plateau)):
                ax.plot(el, metrics[el], color="C1", label="plateau" if i == 0 else "")
            for i, el in enumerate(self._consecutive(linear)):
                ax.plot(el, metrics[el], color="C2", label="linear" if i == 0 else "")
            ax.legend()

            ax = fig.add_subplot(gs[1, 0])
            ax.plot(self.analysis["avg"], label=f"avg")
            ax.legend()
            ax = fig.add_subplot(gs[1, 1])
            ax.plot(self.analysis["avg_abs"], label=f"avg_abs: {self.avg_step_thresh:.4f}")
            ax.hlines(self.avg_step_thresh, xmin=0, xmax=len(self.analysis["avg_abs"]), color="r")
            ax.legend()
            ax = fig.add_subplot(gs[2, 0])
            ax.plot(self.analysis["cum"], label=f"cum")
            ax.legend()
            ax = fig.add_subplot(gs[2, 1])
            ax.plot(np.abs(self.analysis["avg_cum"]), label=f"avg_cum (abs): {self.cum_disp_thresh:.4f}")
            ax.hlines(self.cum_disp_thresh, xmin=0, xmax=len(self.analysis["avg_cum"]), color="r")
            ax.legend()
            ax = fig.add_subplot(gs[3, 0])
            ax.plot(self.analysis["std"], label=f"std: {self.oscillation_thresh:.4f}")
            ax.hlines(self.oscillation_thresh, xmin=0, xmax=len(self.analysis["std"]), color="r")
            ax.legend()
            ax: plt.Axes = fig.add_subplot(gs[3, 1])
            ax.plot(np.abs(self.analysis["slope"]), label=f"slope (abs): {self.slope_thresh:.4f}")
            ax.hlines(self.slope_thresh, xmin=0, xmax=len(self.analysis["slope"]), color="r")
            ax.legend()

    def _consecutive(self, x, stepsize=1):
        return np.split(x, np.where(np.diff(x) != stepsize)[0] + 1)

    def state_dict(self):
        return {key: value for key, value in self.__dict__.items() if key != 'optimizer'}

    def load_state_dict(self, state_dict):
        self.__dict__.update(state_dict)


class AdjustLROnPlateau(object):
    """
    Same as torch.optim.lr_scheduler.ReduceLROnPlateau, but allows factor > 1 and max_lr.
    """

    def __init__(self, optimizer, mode='min', factor=0.1, patience=10,
                 threshold=1e-4, threshold_mode='rel', cooldown=0,
                 min_lr=0, max_lr=None, eps=1e-8, verbose=False):

        if factor >= 1.0 and max_lr is None:
            raise ValueError('When factor > 1, max_lr must be passed.')
        self.factor = factor

        # Attach optimizer
        if not isinstance(optimizer, Optimizer):
            raise TypeError('{} is not an Optimizer'.format(
                type(optimizer).__name__))
        self.optimizer = optimizer

        if isinstance(min_lr, list) or isinstance(min_lr, tuple):
            if len(min_lr) != len(optimizer.param_groups):
                raise ValueError("expected {} min_lrs, got {}".format(
                    len(optimizer.param_groups), len(min_lr)))
            self.min_lrs = list(min_lr)
        else:
            self.min_lrs = [min_lr] * len(optimizer.param_groups)

        max_lr = np.inf if None else max_lr
        if isinstance(max_lr, list) or isinstance(max_lr, tuple):
            if len(max_lr) != len(optimizer.param_groups):
                raise ValueError("expected {} min_lrs, got {}".format(
                    len(optimizer.param_groups), len(max_lr)))
            self.max_lrs = list(max_lr)
        else:
            self.max_lrs = [max_lr] * len(optimizer.param_groups)

        self.patience = patience
        self.verbose = verbose
        self.cooldown = cooldown
        self.cooldown_counter = 0
        self.mode = mode
        self.threshold = threshold
        self.threshold_mode = threshold_mode
        self.best = None
        self.num_bad_epochs = None
        self.mode_worse = None  # the worse value for the chosen mode
        self.eps = eps
        self.last_epoch = 0
        self._init_is_better(mode=mode, threshold=threshold,
                             threshold_mode=threshold_mode)
        self._reset()

    def _reset(self):
        """Resets num_bad_epochs counter and cooldown counter."""
        self.best = self.mode_worse
        self.cooldown_counter = 0
        self.num_bad_epochs = 0

    def step(self, metrics, epoch=None):
        # convert `metrics` to float, in case it's a zero-dim Tensor
        current = float(metrics)
        if epoch is None:
            epoch = self.last_epoch + 1
        else:
            warnings.warn(EPOCH_DEPRECATION_WARNING, UserWarning)
        self.last_epoch = epoch

        if self.is_better(current, self.best):
            self.best = current
            self.num_bad_epochs = 0
        else:
            self.num_bad_epochs += 1

        if self.in_cooldown:
            self.cooldown_counter -= 1
            self.num_bad_epochs = 0  # ignore any bad epochs in cooldown

        if self.num_bad_epochs > self.patience:
            self._reduce_lr(epoch)
            self.cooldown_counter = self.cooldown
            self.num_bad_epochs = 0

        self._last_lr = [group['lr'] for group in self.optimizer.param_groups]

    def _reduce_lr(self, epoch):
        for i, param_group in enumerate(self.optimizer.param_groups):
            old_lr = float(param_group['lr'])
            new_lr = min(max(old_lr * self.factor, self.min_lrs[i]), self.max_lrs[i])
            if old_lr - new_lr > self.eps:
                param_group['lr'] = new_lr
                if self.verbose:
                    epoch_str = ("%.2f" if isinstance(epoch, float) else
                                 "%.5d") % epoch
                    print('Epoch {}: reducing learning rate'
                          ' of group {} to {:.4e}.'.format(epoch_str, i, new_lr))

    @property
    def in_cooldown(self):
        return self.cooldown_counter > 0

    def is_better(self, a, best):
        if self.mode == 'min' and self.threshold_mode == 'rel':
            rel_epsilon = 1. - self.threshold
            return a < best * rel_epsilon

        elif self.mode == 'min' and self.threshold_mode == 'abs':
            return a < best - self.threshold

        elif self.mode == 'max' and self.threshold_mode == 'rel':
            rel_epsilon = self.threshold + 1.
            return a > best * rel_epsilon

        else:  # mode == 'max' and epsilon_mode == 'abs':
            return a > best + self.threshold

    def _init_is_better(self, mode, threshold, threshold_mode):
        if mode not in {'min', 'max'}:
            raise ValueError('mode ' + mode + ' is unknown!')
        if threshold_mode not in {'rel', 'abs'}:
            raise ValueError('threshold mode ' + threshold_mode + ' is unknown!')

        if mode == 'min':
            self.mode_worse = np.inf
        else:  # mode == 'max':
            self.mode_worse = -np.inf

        self.mode = mode
        self.threshold = threshold
        self.threshold_mode = threshold_mode

    def state_dict(self):
        return {key: value for key, value in self.__dict__.items() if key != 'optimizer'}

    def load_state_dict(self, state_dict):
        self.__dict__.update(state_dict)
        self._init_is_better(mode=self.mode, threshold=self.threshold, threshold_mode=self.threshold_mode)

EPOCH_DEPRECATION_WARNING = (
    "The epoch parameter in `scheduler.step()` was not necessary and is being "
    "deprecated where possible. Please use `scheduler.step()` to step the "
    "scheduler. During the deprecation, if epoch is different from None, the "
    "closed form is used instead of the new chainable form, where available. "
    "Please open an issue if you are unable to replicate your use case: "
    "https://github.com/pytorch/pytorch/issues/new/choose."
)
