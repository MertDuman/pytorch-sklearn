import pickle
import copy
from typing import Any, Callable, Iterable, Mapping, Optional, Sequence, Tuple, Union
import warnings

import torch
import torch.nn as nn
from torch.nn.modules.loss import _Loss
from torch.optim.optimizer import Optimizer as _Optimizer
from torch.utils.data import DataLoader, Dataset
from pytorch_sklearn.callbacks.predefined import History, CycleGANHistory

import math

from pytorch_sklearn.neural_network import NeuralNetwork
from pytorch_sklearn.utils.datasets import DefaultDataset, CUDADataset
from pytorch_sklearn.callbacks import Callback
from pytorch_sklearn.utils.class_utils import set_properties_hidden
from pytorch_sklearn.utils.func_utils import to_tensor, to_safe_tensor


class DiffusionUtils:
    def __init__(self, noise_steps=1000, noise_schedule='linear', device=None):
        self.noise_steps = noise_steps
        self.noise_schedule = noise_schedule
        self.device = device if device is not None else ('cuda' if torch.cuda.is_available() else 'cpu')

        self.beta = self.get_betas(self.noise_schedule).to(self.device)
        self.alpha = 1 - self.beta
        self.alpha_hat = torch.cumprod(self.alpha, dim=0)
        self.alpha_hat_prev = torch.as_tensor([1.0, *self.alpha_hat[:-1]], device=self.device)
        self.alpha_hat_next = torch.as_tensor([*self.alpha_hat[1:], 0.0], device=self.device)
        self.sqrt_alpha_hat = torch.sqrt(self.alpha_hat)
        self.sqrt_1m_alpha_hat = torch.sqrt(1 - self.alpha_hat)

        self.log_1m_alpha_hat = torch.log(1.0 - self.alpha_hat)
        self.sqrt_recip_alpha_hat = torch.sqrt(1.0 / self.alpha_hat)
        self.sqrt_recip_m1_alpha_hat = torch.sqrt(1.0 / self.alpha_hat - 1)

        # from: https://arxiv.org/pdf/2102.09672.pdf
        # eq. 10
        self.posterior_variance = self.beta * (1.0 - self.alpha_hat_prev) / (1.0 - self.alpha_hat)
        # log calculation clipped because the posterior variance is 0 at the beginning of the diffusion chain.
        self.posterior_log_variance_clipped = torch.log(torch.as_tensor([self.posterior_variance[1], *self.posterior_variance[1:]], device=self.device))

        # eq. 11
        self.x0_coef = self.beta * torch.sqrt(self.alpha_hat_prev) / (1 - self.alpha_hat)
        self.xt_coef = torch.sqrt(self.alpha) * (1 - self.alpha_hat_prev) / (1 - self.alpha_hat)

    def get_betas(self, schedule_type):
        """
        Get the betas for the diffusion process according to the schedule type.
        """
        assert schedule_type in ['linear', 'cos'], f"schedule_type must be one of ['linear', 'cos'], got {schedule_type}"
        if schedule_type == 'linear':
            # from: https://github.com/openai/guided-diffusion/blob/main/guided_diffusion/gaussian_diffusion.py#L27
            scale = 1000 / self.noise_steps
            beta_start = scale * 0.0001
            beta_end = scale * 0.02
            return torch.linspace(beta_start, beta_end, self.noise_steps, device=self.device)
        elif schedule_type == 'cos':
            # from: https://github.com/openai/guided-diffusion/blob/main/guided_diffusion/gaussian_diffusion.py#L36
            alpha_bar = lambda t: math.cos((t + 0.008) / 1.008 * math.pi / 2) ** 2
            n = self.noise_steps
            return torch.as_tensor([min(1 - alpha_bar((i + 1) / n) / alpha_bar(i / n), max_beta := 0.999) for i in range(n)], device=self.device)

    def add_noise(self, x_0, t):
        """
        Add noise to x_0 to get x_t.
        """
        e = torch.randn_like(x_0)
        return self.sqrt_alpha_hat[t] * x_0 + self.sqrt_1m_alpha_hat[t] * e, e

    def add_noise_to_x(self, x_t, t):
        """
        Add noise to x_t to get x_{t+1}.
        """
        e = torch.randn_like(x_t)
        return torch.sqrt(self.alpha[t]) * x_t + torch.sqrt(1 - self.alpha[t]) * e, e

    def random_t(self, n):
        return torch.randint(self.noise_steps, size=(n,), device=self.device)

    def p_sample(self, model, n, shape, clip_denoised=True):
        """
        Generate samples from the model.
        """
        assert isinstance(shape, (tuple, list))

        model.eval()
        with torch.no_grad():
            x = torch.randn(n, *shape, device=self.device)
            for i in range(self.noise_steps)[::-1]:
                t = torch.full((n,), i, device=self.device).long().view(-1, *[1] * (x.ndim - 1))
                out = self.p_sample_once(model, x, t, clip_denoised=clip_denoised)
                yield out
                x = out["sample"]

    def sample_other(self, model, n):
        model.eval()
        with torch.no_grad():
            x = torch.randn(n, 3, 64, 64, device=self.device)
            for i in range(1, self.noise_steps)[::-1]:
                t = torch.full((n,), i, device=self.device).long()
                pred_e = model(x, t)
                e = torch.zeros_like(x)
                if i > 1:
                    e = torch.randn_like(x)
                alpha = self.alpha[i]
                sqrt_1m_alpha_hat = self.sqrt_1m_alpha_hat[i]
                beta = self.beta[i]
                x = 1 / torch.sqrt(alpha) * (x - (beta / sqrt_1m_alpha_hat) * pred_e) + torch.sqrt(beta) * e
        return x

    def predict_x0_from_e(self, x_t, e, t):
        """
        Predict x0 give x_t and e. e is a unit variance version of the noise that is added to x0 to get x_t.
        """
        assert x_t.shape == e.shape
        return (x_t - self.sqrt_1m_alpha_hat[t] * e) / self.sqrt_alpha_hat[t]

    def predict_x0_from_xprev(self, x_t, xprev, t):
        assert x_t.shape == xprev.shape
        return (1 / self.x0_coef[t]) * xprev - (self.xt_coef[t] / self.x0_coef[t]) * x_t

    def q_mean_variance(self, x_0, t):
        """
        Get the mean and variance of q(x_t | x_0).

        Parameters
        ----------
        x_0: the [N x C x ...] tensor of noiseless inputs.
        t: the number of diffusion steps (minus 1). Here, 0 means one step.

        Returns
        -------
        A tuple (mean, variance, log_variance), all of x_start's shape.
        """
        mean = self.sqrt_alpha_hat[t] * x_0
        variance = 1.0 - self.alpha_hat[t]
        log_variance = self.log_1m_alpha_hat[t]
        return mean, variance, log_variance

    def q_posterior_mean_variance(self, x_0, x_t, t):
        """
        Compute the mean and variance of the diffusion posterior q(x_{t-1} | x_t, x_0)
        from: https://arxiv.org/pdf/2102.09672.pdf
        """
        # eq. 11
        posterior_mean = self.x0_coef[t] * x_0 + self.xt_coef[t] * x_t
        # eq. 10
        posterior_variance = self.posterior_variance[t]
        posterior_log_variance_clipped = self.posterior_log_variance_clipped[t]
        return posterior_mean, posterior_variance, posterior_log_variance_clipped

    def p_mean_variance(self, model, x, t, clip_denoised=True):
        """
        Apply the model to get p(x_{t-1} | x_t), as well as a prediction of x_0.
        """
        N, C = x.shape[:2]
        pred_e = model(x, t.squeeze())
        pred_x0 = self.predict_x0_from_e(x_t=x, e=pred_e, t=t)
        if clip_denoised:
            pred_x0 = torch.clamp(pred_x0, -1.0, 1.0)

        mean_xprev, var_xprev, logvar_xprev = self.q_posterior_mean_variance(x_0=pred_x0, x_t=x, t=t)

        return {
            "mean": mean_xprev,
            "variance": var_xprev,
            "log_variance": logvar_xprev,
            "pred_x0": pred_x0,
        }

    def p_sample_once(self, model, x, t, clip_denoised=True):
        """
        Sample x_{t-1} from the model at the given timestep.
        """
        out = self.p_mean_variance(model, x, t, clip_denoised=clip_denoised)
        noise = torch.randn_like(x)
        nonzero_mask = (t != 0).float().view(-1, *([1] * (x.ndim - 1)))  # no noise when t == 0

        # Added noise has a variance that is added to x_{t - 1} to get x_t, but here, we add it to E[x_{t - 1}] to somehow get x_{t - 1} ?
        # forward process:  x_t         = x_{t - 1} * scale + sqrt(beta_t) * randn
        # but here:         x_{t - 1}   = E[x_{t - 1}]      + sqrt(beta_t) * randn
        # from:
        # https://arxiv.org/pdf/2006.11239.pdf right after eq. 11
        # https://github.com/openai/guided-diffusion/blob/main/guided_diffusion/gaussian_diffusion.py#L438
        # https://github.com/huggingface/diffusers/blob/main/src/diffusers/schedulers/scheduling_ddpm.py#L448
        sample = out["mean"] + nonzero_mask * torch.exp(0.5 * out["log_variance"]) * noise
        return {"sample": sample, "pred_x0": out["pred_x0"]}


class DDPM(NeuralNetwork):
    def __init__(self, module: nn.Module, optimizer: _Optimizer, criterion: _Loss, noise_steps=1000, noise_schedule='cos', clip_denoised=True):
        super().__init__(module, optimizer, criterion)
        self.noise_steps = noise_steps
        self.noise_schedule = noise_schedule
        self.clip_denoised = clip_denoised
        self.diffusion: DiffusionUtils = None

    def forward(self, X):
        X_t, t = X
        return self.module(X_t, t)

    def fit(
        self,
        train_X: Union[torch.Tensor, DataLoader, Dataset],
        train_y: Optional[torch.Tensor] = None,
        validate: bool = False,
        val_X: Optional[Union[torch.Tensor, DataLoader, Dataset]] = None,
        val_y: Optional[torch.Tensor] = None,
        max_epochs: int = 10,
        batch_size: int = 32,
        use_cuda: bool = True,
        fits_gpu: bool = False,
        callbacks: Optional[Sequence[Callback]] = None,
        metrics: Optional[Mapping[str, Callable]] = None,
    ):
        device = "cuda" if use_cuda else "cpu"
        self.diffusion = DiffusionUtils(noise_steps=self.noise_steps, noise_schedule=self.noise_schedule, device=device)
        return super().fit(train_X, train_y, validate, val_X, val_y, max_epochs, batch_size, use_cuda, fits_gpu, callbacks, metrics)

    def unpack_fit_batch(self, batch_data):
        """ Might modify it later to return labels for guided diffusion. """
        X = batch_data
        return X, None      # Expects X, y, but y is decided later. Should fix this later.

    def fit_batch(self, batch_data):
        X, _ = self.unpack_fit_batch(batch_data)
        X = X.to(self._device, non_blocking=True)
        t = self.diffusion.random_t(X.shape[0])
        t = t.view(-1, *[1] * (X.ndim - 1))

        X_t, noise = self.diffusion.add_noise(X, t)
        pred_noise = self.forward((X_t, t.squeeze()))
        loss = self.compute_loss(pred_noise, noise)

        return pred_noise, loss

    def predict_batch(self, batch_data, decision_func: Optional[Callable] = None, **decision_func_kw):
        X = batch_data  # this is noise
        X = X.to(self._device, non_blocking=True)
        for i in range(self.diffusion.noise_steps)[::-1]:
            t = torch.full((X.shape[0],), i, device=self._device).long().view(-1, *[1] * (X.ndim - 1))
            out = self.diffusion.p_sample_once(self.module, X, t, clip_denoised=self.clip_denoised)
            X = out["sample"]
        return X
