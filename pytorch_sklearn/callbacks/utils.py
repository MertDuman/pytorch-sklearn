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
    and even then this is not the same thing as NOT updating those groups.
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
        try:
            return [group for i, group in enumerate(self._optimizer.param_groups) if i in self._groups]
        except AttributeError:
            return []

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

    def __setattr__(self, key, value):
        """ Stops PyTorch's _LRScheduler from reassigning the step function of optimizers, just to throw a warning for users..... """
        if key == "step":
            pass
        else:
            super().__setattr__(key, value)

    @property
    def _step_count(self):
        """ Yet another workaround for PyTorch's _LRScheduler. Uses _step_count to throw a warning. This just bypasses it and returns 1 every time. """
        return 1

    @_step_count.setter
    def _step_count(self, v):
        """ Yet another workaround for PyTorch's _LRScheduler. Uses _step_count to throw a warning. This just bypasses it and returns 1 every time. """
        pass

    @property
    def state(self):
        """ PyTorch optimizers call this, but it shouldn't be called for this class. """
        return self.state_dict()

    def state_dict(self):
        """ 
        For torch.save related stuff. Used as torch.save(ins.state_dict(), 'state.pth'). 

        Instead of saving/loading the object directly, the state_dict should be saved/loaded instead. The difference is that,
        in order to call state_dict or load_state_dict, the object must be instantiated, meaning __init__ was called sometime before.
        This implies that one can omit large objects from state_dict, and request them in the __init__, and not worry about saving/loading those.

        This is exactly how PyTorch does it. For instance, Optimizers have model parameters that are updated with backprop. These parameters are not
        saved/loaded with state_dict, rather they are requested in __init__. It is the user's responsibility to save the model parameters separately and
        provide it later.

        This design ensures that your code does not unexpectedly crash after loading an object. If you solely used pickle (and __get/setstate__), after loading
        an object things might crash because some variables are omitted in __get/setstate__ in order to save memory. As an example, lets say some helper class
        called History points to a huge object called NeuralNetwork, and History just tracks the progress of NeuralNetwork. History has its internal state, but it
        also has a pointer to NeuralNetwork. If you pickle saved History, it would save NeuralNetwork, which would be a huge file. If you omitted NeuralNetwork by
        overriding __get/setstate__, then when you load History, it wouldn't have a pointer to NeuralNetwork, so your code will crash later in the future.
        If you request a pointer to NeuralNetwork in the constructor, and use state_dict, load_state_dict, this issue will not happen.
        """
        return {key: value for key, value in self.__dict__.items() if key != '_optimizer'}

    def load_state_dict(self, state_dict):
        """ For torch.save related stuff. Used as ins.load_state_dict(torch.load('state.pth')). """
        self.__dict__.update(state_dict)

    def __getstate__(self):
        """ 
        For pickle. torch.save also calls this as it uses pickle (but it shouldn't, because we should save the state_dict instead).
        
        __getstate__ and __setstate__ are special, as __init__ is not called on objects that are saved and loaded with pickle. 
        pickle simply creates an empty instance using Class.__new__, and fills its __dict__ using __setstate__. 
        """
        # No need to show this warning, because PyTorch calls __getstate__ of Optimizer's by default.
        # warnings.warn(f"{type(self).__name__}: Saving the object directly is not recommended. Save and load the state_dict instead.")
        return self.state_dict()

    def __setstate__(self, state: dict) -> None:
        """ 
        For pickle. torch.load also calls this as it uses pickle. 
        
        Usually one would call the super() here as well, but this class is NOT an Optimizer, it just inherits from it to bypass some type checks.
        """
        self.__dict__.update(state)


class Tally:
    """
    Tally a given record's best state.
    """
    def __init__(self, recorded: str, mode: str, threshold: float = 0, threshold_mode = 'rel', **kwargs):
        self.__dict__.update(kwargs)
        self.recorded = recorded
        self.mode = mode
        self.threshold = threshold
        self.threshold_mode = threshold_mode
        self.best_record = -np.Inf if mode == "max" else np.Inf

    def state_dict(self):
        return self.__dict__

    def load_state_dict(self, state_dict):
        self.__dict__.update(state_dict)

    def is_better_record(self, new_record):
        if self.mode == 'min' and self.threshold_mode == 'rel':
            rel_epsilon = 1. - self.threshold
            return new_record < self.best_record * rel_epsilon

        elif self.mode == 'min' and self.threshold_mode == 'abs':
            return new_record < self.best_record - self.threshold

        elif self.mode == 'max' and self.threshold_mode == 'rel':
            rel_epsilon = self.threshold + 1.
            return new_record > self.best_record * rel_epsilon

        else:  # mode == 'max' and epsilon_mode == 'abs':
            return new_record > self.best_record + self.threshold

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
        NONE = 0
        OSCILLATING = 1
        PLATEAU = 2
        LINEAR = 3

    """
    Adjusts the learning rate based on trends of the given metric:
        1. Oscillating:
            Reduce LR
        2. Linear:
            Increase LR
        3. Plateau:
            Target Not Set:
                Reduce LR
            Target Set:
                Target Not Reached:
                    Follow Plateau Strategy
                Target Reached:
                    Target 2 Set:
                        Target 2 Reached:
                            Reduce LR to target2_lr
                        Target 2 Not Reached:
                            Reduce LR to min_lr
                    Target 2 Not Set:
                        Reduce LR to min_lr
                        
    Parameters
    ----------
    optimizer : torch.optim.Optimizer
        Determines the parameters to be adjusted.
    mode : str, ["min", "max"]
        Whether the given metric is minimized or maximized.
    down_factor : float
        A value to multiply the LR when reducing it.
    up_factor : float   
        A value to multiply the LR when increasing it.
    target : float, optional
        The given metric tries to approach target. This scheduler does different LR adjustments based on distance to target.
    target_close_thresh : float
        How we decide we are close enough to the target.
    target_distance : str or callable
        How we calculate distance to target. If linear, then abs(current - target). If callable, signature expects (current, target).
    plateau_strategy : str
        What strategy to follow on plateau when away from target. Possible options are:
            up: Increases the learning rate by up_factor.
            cos: Cosine anneals the learning rate between eta_max (or initial_lr) and eta_min.
            random: Randomly selects a lr between initial lr and max_lr. A high cooldown is preferred for this.
    cosine_params : dict
        Parameters for CosineAnnealingWarmRestarts. Additionally you can pass eta_max which would set the max lr of
        CosineAnnealingWarmRestarts, otherwise max lr is set as the default lr of the optimizer.
    """
    def __init__(self,
                 optimizer,
                 mode='min',
                 down_factor=1,
                 up_factor=1,
                 target=None,
                 target_distance="linear",
                 min_lr=0,
                 max_lr=None,
                 cooldown=0,
                 separate_cooldowns=False,
                 plateau_strategy="up",
                 plateau_params=None,
                 oscillation_strategy="down",
                 oscillation_params=None,
                 linear_strategy="up",
                 linear_params=None,
                 relevant_samples=10,
                 deciding_samples=3,
                 oscillation_thresh=0.008,
                 avg_step_thresh=0.01,
                 cum_disp_thresh=0.015,
                 slope_thresh=0.0001,
                 eps=1e-8,
                 verbose=False,
                 ):
        from collections import defaultdict

        if not isinstance(optimizer, Optimizer):
            raise TypeError(f"{type(optimizer).__name__} is not an Optimizer")

        if up_factor > 1 and max_lr is None:
            raise ValueError("When up factor > 1, max_lr must be passed.")

        if mode not in {'min', 'max'}:
            raise ValueError('mode ' + mode + ' is unknown!')

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

        if plateau_strategy == "cos":
            if not isinstance(plateau_params, dict):
                raise ValueError(f"Using cosine plateau strategy. Pass dict of parameters for torch.optim.lr_scheduler.CosineAnnealingWarmRestarts (cosine_params).")
            elif isinstance(plateau_params, dict) and "T_0" not in plateau_params:
                raise ValueError(f"T_0 must be present for cosine parameters.")

        self.optimizer = optimizer
        self.mode = mode
        self.down_factor = down_factor
        self.up_factor = up_factor
        self.target = target
        self.target_distance = target_distance
        self.cooldown = cooldown
        self.separate_cooldowns = separate_cooldowns

        self.plateau_strategy = plateau_strategy
        self.plateau_params = plateau_params
        self.oscillation_strategy = oscillation_strategy
        self.oscillation_params = oscillation_params
        self.linear_strategy = linear_strategy
        self.linear_params = linear_params

        self.relevant_samples = relevant_samples
        self.deciding_samples = deciding_samples
        self.oscillation_thresh = oscillation_thresh
        self.avg_step_thresh = avg_step_thresh
        self.cum_disp_thresh = cum_disp_thresh
        self.slope_thresh = slope_thresh
        self.eps = eps
        self.verbose = verbose

        self.metrics = []
        self.analysis = defaultdict(list)
        self.decisions = []

        self._init_plateau_strategy()
        self._init_oscillation_strategy()
        self._init_linear_strategy()

        self.cooldown_counter = [0]
        if self.separate_cooldowns:
            self.cooldown_counter = [0] * len(self.Trend)

        self.target_reached = False
        self.last_step = 0
        self._last_lr = [group['lr'] for group in self.optimizer.param_groups]

    def step(self, metric):
        # convert `metrics` to float, in case it's a zero-dim Tensor
        current = float(metric)
        self.metrics.append(current)
        self.last_step += 1

        if len(self.metrics) >= self.relevant_samples:
            self._analyze_metrics()
            decision = self._make_decision()

            if self._on_cooldown(decision):
                self._decrement_cooldown()
            else:
                if self._try_update_lr(current, decision):
                    self._reset_cooldown(decision)
        else:
            self.decisions.append(self.Trend.NONE)

        self._last_lr = [group['lr'] for group in self.optimizer.param_groups]

    def _init_plateau_strategy(self):
        self.cos_anneal = None
        if self.plateau_strategy == "cos":
            plateau_params = self.plateau_params.copy()
            cos_max_lr = plateau_params.pop("eta_max", None)
            self.cos_anneal = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(self.optimizer, **plateau_params)
            if cos_max_lr is not None:
                self.cos_anneal.base_lrs = [cos_max_lr for _ in self.optimizer.param_groups]

    def _init_oscillation_strategy(self):
        pass

    def _init_linear_strategy(self):
        pass

    def _on_cooldown(self, decision: Trend):
        index = decision.value if self.separate_cooldowns else self.Trend.NONE.value
        return self.cooldown_counter[index] > 0

    def _decrement_cooldown(self):
        for i in range(len(self.cooldown_counter)):
            if self.cooldown_counter[i] > 0:
                self.cooldown_counter[i] -= 1

    def _reset_cooldown(self, decision: Trend):
        index = decision.value if self.separate_cooldowns else self.Trend.NONE.value
        self.cooldown_counter[index] = self.cooldown

    def _try_update_lr(self, current, decision):
        if decision == self.Trend.OSCILLATING:
            self._update_lr(self.last_step, self.down_factor)
        elif decision == self.Trend.PLATEAU:
            if self._is_target_reached(current):
                self._update_lr(self.last_step, self.down_factor)
            else:
                self._plateau_update_lr(self.last_step)
        elif decision == self.Trend.LINEAR:
            self._update_lr(self.last_step, self.up_factor)
        elif decision == self.Trend.NONE:
            pass

        return decision != self.Trend.NONE

    def _update_lr(self, step, factor):
        for i, param_group in enumerate(self.optimizer.param_groups):
            old_lr = float(param_group['lr'])
            new_lr = min(max(old_lr * factor, self.min_lrs[i]), self.max_lrs[i])
            if abs(old_lr - new_lr) > self.eps:
                param_group['lr'] = new_lr
                if self.verbose:
                    print(f"Step {step}: updating learning rate of group {i} from {old_lr:.6f} to {new_lr:.6f}.")

    def _plateau_update_lr(self, step):
        if self.plateau_strategy == "up":
            self._update_lr(step, self.up_factor)
        elif self.plateau_strategy == "cos":
            self.cos_anneal.step()
        elif self.plateau_strategy == "random":
            for i, param_group in enumerate(self.optimizer.param_groups):
                old_lr = float(param_group['lr'])
                new_lr = np.random.rand() * (self.max_lrs[i] - self.optimizer.defaults["lr"]) + self.optimizer.defaults["lr"]
                if abs(old_lr - new_lr) > self.eps:
                    param_group['lr'] = new_lr
                    if self.verbose:
                        print(f"Step {step}: updating learning rate of group {i} from {old_lr:.6f} to {new_lr:.6f}.")
        else:
            raise NotImplementedError

    def _make_decision(self):
        # avg = np.mean(self.analysis["avg"][-self.deciding_samples:])
        avg_abs = np.mean(self.analysis["avg_abs"][-self.deciding_samples:])
        # cum = np.mean(self.analysis["cum"][-self.deciding_samples:])
        avg_cum = np.mean(self.analysis["avg_cum"][-self.deciding_samples:])
        std = np.mean(self.analysis["std"][-self.deciding_samples:])
        slope = np.mean(self.analysis["slope"][-self.deciding_samples:])

        decision = self.Trend.NONE

        if std > self.oscillation_thresh and avg_abs > self.avg_step_thresh:
            decision = self.Trend.OSCILLATING
        elif avg_abs < self.avg_step_thresh and np.abs(avg_cum) < self.cum_disp_thresh and np.abs(slope) < self.slope_thresh:
            decision = self.Trend.PLATEAU
        elif avg_abs < self.avg_step_thresh and np.abs(avg_cum) > self.cum_disp_thresh and np.abs(slope) > self.slope_thresh:
            decision = self.Trend.LINEAR
        # SPIKE

        self.decisions.append(decision)
        return decision

    def _analyze_metrics(self):
        metrics = np.array(self.metrics[-self.relevant_samples:])
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

    def _is_target_reached(self, current):
        distance = self._distance_to_target(current)  # target - current
        if self.mode == "min":
            # we are minimizing to get to target, distance is negative when target is not reached.
            if distance < 0:
                return False
            else:
                self.target_reached = True
                return True
        elif self.mode == "max":
            # we are maximizing to get to target, distance is positive when target is not reached.
            if distance > 0:
                return False
            else:
                self.target_reached = True
                return True

        return False

    def _distance_to_target(self, current):
        if self.target is None:
            return 0

        if self.target_distance == "linear":
            return self.target - current

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
