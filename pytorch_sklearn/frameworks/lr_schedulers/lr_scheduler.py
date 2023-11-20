import torch
import torch.optim.lr_scheduler as tlrs
import pytorch_sklearn as psk
import numpy as np




class LinearLRScheduler(psk.callbacks.LRScheduler):
    """
    Linear learning rate scheduler. This is slightly different from torch.optim.lr_scheduler.LinearLR.
    This scheduler doesn't immediately scale the learning rate by start_factor. Thus, it will take total_steps + 1 steps to reach end_factor,
    as the first step will scale the learning rate by start_factor. Nonetheless, it will still take total_steps "jumps" to reach end_factor.

    To get exactly the same behavior as torch's LinearLR, set torch_linear=True.

    Here is a comparison with LinearLR when total_steps=2, start_factor=0.2, end_factor=0, base_lr=1:
        This                            LinearLR
        start:  lr = 1                  lr = 0.2
        step 0: lr = 0.2                lr = 0.1
        step 1: lr = 0.1                lr = 0
        step 2: lr = 0                  lr = 0
        step 3: lr = 0                  lr = 0
    This scheduler also accepts an interval (steps it is active), so if interval is [2, -1):
        This                            LinearLR (equivalent to not calling step for torch's scheduler)
        start:  lr = 1                  lr = 0.2
        step 0: lr = 1                  lr = 0.2
        step 1: lr = 1                  lr = 0.2
        step 2: lr = 0.2                lr = 0.1
        step 3: lr = 0.1                lr = 0
        step 4: lr = 0                  lr = 0
        step 5: lr = 0                  lr = 0
    """
    def __init__(self, optimizer, total_steps, start_factor=1, end_factor=0.1, torch_linear=False, per_epoch=True, per_step=None, store_lrs=False, reset_on_fit_end=True, reset_on_epoch_end=False, interval=None, pass_metric=None):
        self.total_steps = total_steps
        self.start_mult = start_factor
        self.end_mult = end_factor
        self.torch_linear = torch_linear

        def get_lr_scale(step):
            if step == 0:
                return 1
            if step > total_steps:
                return end_factor
            return ((step - 1) / total_steps) * (end_factor - start_factor) + start_factor

        if torch_linear:
            lr_scheduler = tlrs.LinearLR(optimizer, start_factor, end_factor, total_steps)
        else:
            lr_scheduler = tlrs.LambdaLR(optimizer, lr_lambda=get_lr_scale)
        super().__init__(lr_scheduler, per_epoch=per_epoch, per_step=per_step, store_lrs=store_lrs, reset_on_fit_end=reset_on_fit_end, reset_on_epoch_end=reset_on_epoch_end, interval=interval, pass_metric=pass_metric)


class AttentionLRScheduler(psk.callbacks.LRScheduler):
    """
    Learning rate scheduler used in the Attention is All You Need paper.
    From: https://github.com/jadore801120/attention-is-all-you-need-pytorch/blob/master/transformer/Optim.py
    """
    def __init__(self, optimizer, d_model, warmup_steps, lr_multiplier, per_epoch=True, per_step=None, store_lrs=False, reset_on_fit_end=True, reset_on_epoch_end=False, interval=None, pass_metric=None):
        self.d_model = d_model
        self.warmup_steps = warmup_steps
        self.lr_multiplier = lr_multiplier

        def get_lr_scale(step):
            step = step + 1
            return (d_model ** -0.5) * min(step ** (-0.5), step * warmup_steps ** (-1.5)) * lr_multiplier

        lr_scheduler = tlrs.LambdaLR(optimizer, lr_lambda=get_lr_scale)
        super().__init__(lr_scheduler, per_epoch=per_epoch, per_step=per_step, store_lrs=store_lrs, reset_on_fit_end=reset_on_fit_end, reset_on_epoch_end=reset_on_epoch_end, interval=interval, pass_metric=pass_metric)


class CosineWarmupLRScheduler(psk.callbacks.LRScheduler):
    def __init__(self, optimizer, warmup_steps, lr_min, lr_max, lr_start, max_decay_steps, per_epoch=True, per_step=None, store_lrs=False, reset_on_fit_end=True, reset_on_epoch_end=False, interval=None, pass_metric=None):
        class InternalCosineWarmupLRScheduler(torch.optim.lr_scheduler._LRScheduler):
            def __init__(self, optimizer, warmup_steps, lr_min, lr_max, lr_start, max_decay_steps):
                self.warmup_steps = warmup_steps
                self.lr_start = lr_start
                self.lr_min = lr_min
                self.lr_max = lr_max
                self.max_decay_steps = max_decay_steps
                super().__init__(optimizer)

            def get_lr(self):
                return [self.get_lr_with_step(self.last_epoch) for _ in self.base_lrs]

            def get_lr_with_step(self, step):
                if step < self.warmup_steps:
                    lr = (self.lr_max - self.lr_start) / self.warmup_steps * step + self.lr_start
                    return lr
                else:
                    t = (step - self.warmup_steps) / (self.max_decay_steps - self.warmup_steps)
                    t = min(t, 1.0)
                    lr = self.lr_min + 0.5 * (self.lr_max - self.lr_min) * (1 + np.cos(t * np.pi))
                    return lr

        lr_scheduler = InternalCosineWarmupLRScheduler(optimizer, warmup_steps, lr_min, lr_max, lr_start, max_decay_steps)
        super().__init__(lr_scheduler, per_epoch=per_epoch, per_step=per_step, store_lrs=store_lrs, reset_on_fit_end=reset_on_fit_end, reset_on_epoch_end=reset_on_epoch_end, interval=interval, pass_metric=pass_metric)

