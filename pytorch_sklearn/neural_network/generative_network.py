import pickle
import copy
from typing import Callable, Iterable, Mapping, Optional, Union
import warnings

import torch
import torch.nn as nn
from torch.nn.modules.loss import _Loss
from torch.optim.optimizer import Optimizer as _Optimizer
from torch.utils.data import DataLoader, Dataset

from pytorch_sklearn.neural_network import NeuralNetwork
from pytorch_sklearn.utils.datasets import DefaultDataset, CUDADataset
from pytorch_sklearn.callbacks import CallbackManager, Callback
from pytorch_sklearn.utils.class_utils import set_properties_hidden
from pytorch_sklearn.utils.func_utils import to_tensor, to_safe_tensor


class CycleGAN(NeuralNetwork):
    """
    Implements CycleGAN from the paper: https://github.com/junyanz/CycleGAN
    Follows similar implementation to: https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix

    Parameters
    ----------
    generator_A : PyTorch module
        Generator that tries to convert class A to class B
    generator_B : PyTorch module
        Generator that tries to convert class B to class A
    discriminator_A : PyTorch module
        Discriminator that classifies input as "from class A" or "not from class A"
    discriminator_B : PyTorch module
        Discriminator that classifies input as "from class B" or "not from class B"
    optimizer_Gen : PyTorch optimizer
        Updates the weights of the generators.
    optimizer_Disc : PyTorch optimizer
        Updates the weights of the discriminators.
    criterion : PyTorch loss
        GAN loss that will be applied to discriminator outputs.
    """
    def __init__(
        self, 
        G_A: nn.Module, G_B: nn.Module, 
        D_A: nn.Module, D_B: nn.Module, 
        G_optim: _Optimizer, D_optim: _Optimizer,
        criterion: str = "lsgan",
        cycle_loss: str = "l1",
        identity_loss: str = "l1",
        lambda_cycle: float = 10.0,
        lambda_identity: float = 5.0
    ):
        # Base parameters
        self.G_A = G_A
        self.G_B = G_B
        self.D_A = D_A
        self.D_B = D_B
        self.G_optim = G_optim
        self.D_optim = D_optim
        self.lambda_cycle = lambda_cycle
        self.lambda_identity = lambda_identity
        
        # Loss
        self.implemented_gan_losses = ["lsgan", "gan"]
        assert criterion in self.implemented_gan_losses, f"Criterion {criterion} not implemented. Choose from {self.implemented_gan_losses}."
        self.G_criterion, self.D_criterion = self.build_criterion(criterion)

        self.implemented_cycle_losses = ["l1", "l2"]
        assert cycle_loss in self.implemented_cycle_losses, f"Cycle loss {cycle_loss} not implemented. Choose from {self.implemented_cycle_losses}."
        self.cycle_loss = nn.L1Loss() if cycle_loss == "l1" else nn.MSELoss()

        self.implemented_identity_losses = ["l1", "l2"]
        assert identity_loss in self.implemented_identity_losses, f"Identity loss {identity_loss} not implemented. Choose from {self.implemented_identity_losses}."
        self.identity_loss = nn.L1Loss() if identity_loss == "l1" else nn.MSELoss()

        self.cbmanager = CallbackManager()  # SAVED
        self.keep_training = True

    def fit(
        self,
        train_X: Union[torch.Tensor, DataLoader, Dataset],
        train_y: Optional[torch.Tensor] = None,
        max_epochs: int = 10,
        batch_size: int = 32,
        use_cuda: bool = True,
        fits_gpu: bool = False,
        callbacks: Optional[Iterable[Callback]] = None,
        metrics: Optional[Mapping[str, Callable]] = None,
    ):
        super().fit(
            train_X=train_X,
            train_y=train_y,
            validate=False,
            val_X=None,
            val_y=None,
            max_epochs=max_epochs,
            batch_size=batch_size,
            use_cuda=use_cuda,
            fits_gpu=fits_gpu,
            callbacks=callbacks,
            metrics=metrics,
        )

    def fit_impl(self):
        self.G_A = self.G_A.to(self._device)
        self.G_B = self.G_B.to(self._device)
        self.D_A = self.D_A.to(self._device)
        self.D_B = self.D_B.to(self._device)

        self._notify("on_fit_begin")
        self._epoch = 1
        while self._epoch < self._max_epochs + 1:
            if not self.keep_training:
                self._notify("on_fit_interrupted")
                break
            self.train()
            self.fit_epoch(self._train_loader)

            if self._epoch == self._max_epochs:
                break  # so that self._epoch == self._max_epochs when loop exits.
            self._epoch += 1
        self._notify("on_fit_end")

    def fit_epoch(self, data_loader):
        self._num_batches = len(data_loader)
        self._notify(f"on_{self._pass_type}_epoch_begin")
        for self._batch, self._batch_data in enumerate(data_loader, start=1):
            self._notify(f"on_{self._pass_type}_batch_begin")

            self._batch_out, self._batch_loss = self.fit_batch(self._batch_data)

            self._notify(f"on_{self._pass_type}_batch_end")
        self._notify(f"on_{self._pass_type}_epoch_end")

    def unpack_fit_batch(self, batch_data):
        ''' In CycleGAN setup, we have no inputs, but only two targets: A and B. '''
        A, B = batch_data
        return [], [A, B]

    def fit_batch(self, batch_data):
        _, [A, B] = self.unpack_fit_batch(batch_data)
        A = A.to(self._device, non_blocking=True)
        B = B.to(self._device, non_blocking=True)

        A2B = self.G_A(A)
        B2A = self.G_B(B)
        A2B2A = self.G_B(A2B)
        B2A2B = self.G_A(B2A)
        A2A = self.G_B(A)
        B2B = self.G_A(B)

        B2A_logits = self.D_A(B2A)
        A2B_logits = self.D_B(A2B)

        # Generator loss
        A2B_g_loss = self.G_criterion(A2B_logits)
        B2A_g_loss = self.G_criterion(B2A_logits)
        A2B2A_cycle_loss = self.cycle_loss(A2B2A, A)
        B2A2B_cycle_loss = self.cycle_loss(B2A2B, B)
        A2A_identity_loss = self.identity_loss(A2A, A)
        B2B_identity_loss = self.identity_loss(B2B, B)

        G_loss = A2B_g_loss + B2A_g_loss + \
            (A2B2A_cycle_loss + B2A2B_cycle_loss) * self.lambda_cycle + \
            (A2A_identity_loss + B2B_identity_loss) * self.lambda_identity
        
        self.backward(G_loss, self.G_optim)

        # Gradients past this point are not needed
        A2B = A2B.detach()
        B2A = B2A.detach()

        # Discriminator loss
        A_logits = self.D_A(A)
        B_logits = self.D_B(B)
        B2A_d_logits = self.D_A(B2A)
        A2B_d_logits = self.D_B(A2B)

        A_d_loss, B2A_d_loss = self.D_criterion(A_logits, B2A_d_logits)
        B_d_loss, A2B_d_loss = self.D_criterion(B_logits, A2B_d_logits)

        D_loss = A_d_loss + B_d_loss + B2A_d_loss + A2B_d_loss

        self.backward(D_loss, self.D_optim)
        
        return [A2B, B2A], G_loss + D_loss

    def build_criterion(self, criterion: str):
        """
        Builds the criterion based on the input string.

        Parameters
        ----------
        criterion : str
            Name of the criterion to use.

        Returns
        -------
        criterion : PyTorch loss
            The criterion to use.
        """
        if criterion == "lsgan":
            mse = nn.MSELoss()
            def G_criterion(f_logits):
                return mse(f_logits, torch.ones_like(f_logits))
            def D_criterion(r_logits, f_logits):
                return mse(r_logits, torch.ones_like(r_logits)), mse(f_logits, torch.zeros_like(f_logits))
            return G_criterion, D_criterion
        elif criterion == "gan":
            bce = nn.BCEWithLogitsLoss()
            def G_criterion(f_logits):
                return bce(f_logits, torch.ones_like(f_logits))
            def D_criterion(r_logits, f_logits):
                return bce(r_logits, torch.ones_like(r_logits)), bce(f_logits, torch.zeros_like(f_logits))
            return G_criterion, D_criterion
        else:
            raise ValueError(f"Criterion {criterion} not implemented.")
        
    # Model Modes
    def train(self):
        self.G_A.train()
        self.G_B.train()
        self.D_A.train()
        self.D_B.train()
        self._pass_type = "train"

    def val(self):
        self.G_A.eval()
        self.G_B.eval()
        self.D_A.eval()
        self.D_B.eval()
        self._pass_type = "val"

    def test(self):
        self.G_A.eval()
        self.G_B.eval()
        self.D_A.eval()
        self.D_B.eval()
        self._pass_type = "test"