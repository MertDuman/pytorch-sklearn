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

from pytorch_sklearn.neural_network import NeuralNetwork
from pytorch_sklearn.utils.datasets import DefaultDataset, CUDADataset
from pytorch_sklearn.callbacks import Callback
from pytorch_sklearn.utils.class_utils import set_properties_hidden
from pytorch_sklearn.utils.func_utils import to_tensor, to_safe_tensor, to_device


class CycleGAN(NeuralNetwork):
    """
    Implements CycleGAN from the paper: https://github.com/junyanz/CycleGAN
    Follows similar implementation to: https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix

    Parameters
    ----------
    G_A : PyTorch module
        Generator that tries to convert class A to class B
    G_B : PyTorch module
        Generator that tries to convert class B to class A
    D_A : PyTorch module
        Discriminator that classifies input as "from class A" or "not from class A"
    D_B : PyTorch module
        Discriminator that classifies input as "from class B" or "not from class B"
    G_optim : PyTorch optimizer
        Updates the weights of the generators.
    D_optim : PyTorch optimizer
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
        cycle_criterion: str = "l1",
        identity_criterion: str = "l1",
        lambda_cycle: float = 10.0,
        lambda_identity: float = 5.0,
        elo_training: bool = False,
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
        self.elo_training = elo_training
        
        # Loss
        self.implemented_gan_losses = ["lsgan", "gan"]
        assert criterion in self.implemented_gan_losses, f"Criterion {criterion} not implemented. Choose from {self.implemented_gan_losses}."
        self.G_criterion, self.D_criterion = self.build_criterion(criterion)

        self.implemented_cycle_losses = ["l1", "l2"]  # TODO: focal frequency loss: https://github.com/EndlessSora/focal-frequency-loss
        assert cycle_criterion in self.implemented_cycle_losses, f"Cycle loss {cycle_criterion} not implemented. Choose from {self.implemented_cycle_losses}."
        self.cycle_criterion = nn.L1Loss() if cycle_criterion == "l1" else nn.MSELoss()

        self.implemented_identity_losses = ["l1", "l2"]
        assert identity_criterion in self.implemented_identity_losses, f"Identity loss {identity_criterion} not implemented. Choose from {self.implemented_identity_losses}."
        self.identity_criterion = nn.L1Loss() if identity_criterion == "l1" else nn.MSELoss()

        # Maintenance parameters
        self._callbacks: Sequence[Callback] = [CycleGANHistory()]  # SAVED
        self._using_original = True  # SAVED
        self._original_state_dict: Optional[Mapping[str, Any]] = None # SAVED
        self.keep_training = True

        self.G_elo = 0
        self.D_elo = 0

    @property
    def history(self) -> CycleGANHistory:
        assert isinstance(self.callbacks[0], CycleGANHistory)
        return self.callbacks[0]

    def forward(self, X):
        A, B = X
        A2B = self.G_A(A)
        B2A = self.G_B(B)
        A2B2A = self.G_B(A2B)
        B2A2B = self.G_A(B2A)
        A2A = self.G_B(A)
        B2B = self.G_A(B)
        return A2B, B2A, A2B2A, B2A2B, A2A, B2B

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
        ''' Compute and return the output and loss for a batch. This method should be overridden by subclasses.
            
        The default implementation assumes that ``batch_data`` is a tuple of ``(A, B)`` and that the model
        outputs ``A2B, B2A``. The loss is a 4-tuple of ``(G_A_loss, G_B_loss, D_A_loss, D_B_loss)``.

        Parameters
        ----------
        batch_data : Any
            Batch data as returned by the dataloader provided to ``fit``.
        '''
        batch_data = to_device(batch_data, self._device)
        _, [A, B] = self.unpack_fit_batch(batch_data)

        A2B, B2A, A2B2A, B2A2B, A2A, B2B = self.forward([A, B])

        B2A_logits = self.D_A(B2A)
        A2B_logits = self.D_B(A2B)
            
        # Generator loss
        A2B_g_loss = self.G_criterion(A2B_logits)
        B2A_g_loss = self.G_criterion(B2A_logits)

        # Cycle loss
        A2B2A_cycle_loss = self.cycle_criterion(A2B2A, A)
        B2A2B_cycle_loss = self.cycle_criterion(B2A2B, B)

        # Identity loss
        A2A_identity_loss = self.identity_criterion(A2A, A)
        B2B_identity_loss = self.identity_criterion(B2B, B)

        # Total generator loss
        G_A_loss = A2B_g_loss + A2B2A_cycle_loss * self.lambda_cycle + A2A_identity_loss * self.lambda_identity
        G_B_loss = B2A_g_loss + B2A2B_cycle_loss * self.lambda_cycle + B2B_identity_loss * self.lambda_identity

        G_loss = G_A_loss + G_B_loss
        
        if self._pass_type == "train" and (not self.elo_training or self.G_elo <= self.D_elo):
            self.backward(G_loss, self.G_optim)

        # Gradients past this point are not needed
        A2B = A2B.detach()
        B2A = B2A.detach()

        A_logits = self.D_A(A)
        B_logits = self.D_B(B)
        B2A_d_logits = self.D_A(B2A)
        A2B_d_logits = self.D_B(A2B)

        # Discriminator loss
        A_d_loss, B2A_d_loss = self.D_criterion(A_logits, B2A_d_logits)
        B_d_loss, A2B_d_loss = self.D_criterion(B_logits, A2B_d_logits)

        # Total discriminator loss
        D_A_loss = A_d_loss + B2A_d_loss
        D_B_loss = B_d_loss + A2B_d_loss

        D_loss = D_A_loss + D_B_loss

        if self._pass_type == "train" and (not self.elo_training or self.G_elo >= self.D_elo):
            self.backward(D_loss, self.D_optim)

        if self.elo_training:
            with torch.no_grad():
                # Discriminator wants reals to be 1 and fakes to be 0
                # These are all fakes.
                A_preds = A_logits > 0.5  # Values above 0.5 get 1, meaning they are predicted as real. Since these are real, disc wants 1.
                B_preds = B_logits > 0.5
                B2A_preds = B2A_logits > 0.5  # Values above 0.5 get 1, meaning they are predicted as real. Since these are fake, disc wants 0.
                A2B_preds = A2B_logits > 0.5

                # Generator only rewarded for fooling discriminator
                G_wins = (B2A_preds == 1).sum().item() + (A2B_preds == 1).sum().item()
                G_losses = (B2A_preds == 0).sum().item() + (A2B_preds == 0).sum().item()

                # Discriminator rewarded for spotting fakes, but penalized for missing reals
                D_wins = (B2A_preds == 0).sum().item() + (A2B_preds == 0).sum().item()
                D_losses = (A_preds == 0).sum().item() + (B_preds == 0).sum().item()

                self.G_elo += (G_wins > 0) - (G_losses > 0)
                self.D_elo += (D_wins > 0) - (D_losses > 0)
        
        return [A2B, B2A, A2B2A, B2A2B, A2A, B2B], [G_A_loss, G_B_loss, D_A_loss, D_B_loss]
    
    def unpack_predict_batch(self, batch_data):
        ''' In CycleGAN setup, we have no inputs, but only two targets: A and B. '''
        A, B = batch_data
        return [], [A, B]

    def predict_batch(self, batch_data, decision_func: Optional[Callable] = None, **decision_func_kw):
        ''' Compute and return the output for a batch. This method should be overridden by subclasses.
        
        The default implementation assumes that ``batch_data`` is a tuple of ``(A, B)`` and that the model
        outputs a 4-tuple ``(A2B, B2A, A2B2A, B2A2B)``.

        NOTE! decision_func is not used in this implementation.
                
        Parameters
        ----------
        batch_data : Any
            Batch data as returned by the dataloader provided to ``predict`` or ``predict_generator``.
        decision_func : Optional[Callable]
            Decision function passed to ``predict`` or ``predict_generator``.. If None, the output of the model is returned.
        **decision_func_kw
            Keyword arguments passed to ``decision_func``, provided to ``predict`` or ``predict_generator``.
        '''
        batch_data = to_device(batch_data, self._device)
        _, [A, B] = self.unpack_predict_batch(batch_data)

        A2B, B2A, A2B2A, B2A2B, A2A, B2B = self.forward([A, B])
        return [A2B, B2A, A2B2A, B2A2B, A2A, B2B]
    
    def unpack_score_batch(self, batch_data):
        ''' In CycleGAN setup, we have no inputs, but only two targets: A and B. '''
        A, B = batch_data
        return [], [A, B]
    
    def score_batch(self, batch_data, score_func: Optional[Callable[[Any], Any]] = None, **score_func_kw):
        ''' Compute and return the score for a batch. This method should be overridden by subclasses.
        
        The default implementation assumes that ``batch_data`` is a tuple of ``(A, B)``.
        If ``score_func`` is None, score is the model loss.

        Parameters
        ----------
        batch_data : Any
            Batch data as returned by the dataloader provided to ``score``.
        score_func : Optional[Callable]
            Score function passed to ``score``. If None, the criterion is used by default.
            Takes network input and output as input, and returns either a scalar, tuple of scalars, tensor, or tuple of tensors.
        **score_func_kw
            Keyword arguments passed to ``score_func``, provided to ``score``.
        '''
        batch_data = to_device(batch_data, self._device)
        _, [A, B] = self.unpack_score_batch(batch_data)

        A2B, B2A, A2B2A, B2A2B, A2A, B2B = self.forward([A, B])

        if score_func is None:
            B2A_logits = self.D_A(B2A)
            A2B_logits = self.D_B(A2B)

            # Generator loss
            A2B_g_loss = self.G_criterion(A2B_logits)
            B2A_g_loss = self.G_criterion(B2A_logits)
            A2B2A_cycle_loss = self.cycle_criterion(A2B2A, A)
            B2A2B_cycle_loss = self.cycle_criterion(B2A2B, B)
            A2A_identity_loss = self.identity_criterion(A2A, A)
            B2B_identity_loss = self.identity_criterion(B2B, B)

            G_A_loss = A2B_g_loss + A2B2A_cycle_loss * self.lambda_cycle + A2A_identity_loss * self.lambda_identity
            G_B_loss = B2A_g_loss + B2A2B_cycle_loss * self.lambda_cycle + B2B_identity_loss * self.lambda_identity

            G_loss = G_A_loss + G_B_loss

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

            D_A_loss = A_d_loss + B2A_d_loss
            D_B_loss = B_d_loss + A2B_d_loss

            D_loss = D_A_loss + D_B_loss

            score = [G_A_loss.item(), G_B_loss.item(), D_A_loss.item(), D_B_loss.item()]
        else:
            score = score_func(self._to_safe_tensor([A2B, B2A, A2B2A, B2A2B, A2A, B2B]), self._to_safe_tensor(batch_data), **score_func_kw)
        return score
    
    def to_device(self, device):
        self.G_A = self.G_A.to(device)
        self.G_B = self.G_B.to(device)
        self.D_A = self.D_A.to(device)
        self.D_B = self.D_B.to(device)

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

    def get_module_weights(self):
        return {
            "G_A_state": self.G_A.state_dict(),
            "G_B_state": self.G_B.state_dict(),
            "D_A_state": self.D_A.state_dict(),
            "D_B_state": self.D_B.state_dict()
        }
    
    def load_module_weights(self, state_dict, strict=True, map_param_names: Mapping[str, str] = None):
        # Workaround for optimizer being on the wrong device. Check ``func_utils.optimizer_to`` for more info.
        checkpoint_device = state_dict["G_A_state"][next(iter(state_dict["G_A_state"]))].device
        self.to_device(checkpoint_device)

        if map_param_names is not None:
            state_dict = {map_param_names.get(k, k): v for k, v in state_dict.items()}

        self.G_A.load_state_dict(state_dict["G_A_state"], strict=strict)
        self.G_B.load_state_dict(state_dict["G_B_state"], strict=strict)
        self.D_A.load_state_dict(state_dict["D_A_state"], strict=strict)
        self.D_B.load_state_dict(state_dict["D_B_state"], strict=strict)

    def get_module_parameters(self):
        yield from self.G_A.parameters()
        yield from self.G_B.parameters()
        yield from self.D_A.parameters()
        yield from self.D_B.parameters()

    def state_dict(self):
        return {
            **self.get_module_weights(),
            "original_module_state": self._original_state_dict,
            "using_original": self._using_original,
            "G_optim_state": self.G_optim.state_dict(),
            "D_optim_state": self.D_optim.state_dict(),
            "epoch": self._epoch,
            "batch": self._batch
        }

    def load_state_dict(self, state_dict, strict=True, map_param_names: Mapping[str, str] = None):
        self.load_module_weights(state_dict, strict=strict, map_param_names=map_param_names)
        self._original_state_dict = state_dict["original_module_state"]
        self._using_original = state_dict["using_original"]
        self.G_optim.load_state_dict(state_dict["G_optim_state"])
        self.D_optim.load_state_dict(state_dict["D_optim_state"])
        self._epoch = state_dict["epoch"]
        self._batch = state_dict["batch"]



class R2CGAN(CycleGAN):
    """
    Implements CycleGAN from the paper: https://arxiv.org/abs/2209.14770
    Follows similar implementation to: https://github.com/meteahishali/R2C-GAN

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
        cycle_criterion: str = "l1",
        identity_criterion: str = "l1",
        lambda_cycle: float = 10.0,
        lambda_identity: float = 5.0,
    ):
        super().__init__(
            G_A, G_B, D_A, D_B, G_optim, D_optim, criterion, cycle_criterion, identity_criterion, lambda_cycle, lambda_identity, elo_training=False
        )

        self.classification_criterion = nn.CrossEntropyLoss()

    def unpack_fit_batch(self, batch_data):
        ''' In CycleGAN setup, we have no inputs, but only two targets: A and B. '''
        A, yA, B, yB = batch_data
        return [], [A, yA, B, yB]
        
    def fit_batch(self, batch_data):
        ''' Compute and return the output and loss for a batch. This method should be overridden by subclasses.
            
        The default implementation assumes that ``batch_data`` is a tuple of ``(A, yA, B, yB)`` and that the model
        outputs ``A2B, B2A``. The loss is a 4-tuple of ``(G_A_loss, G_B_loss, D_A_loss, D_B_loss)``.

        Parameters
        ----------
        batch_data : Any
            Batch data as returned by the dataloader provided to ``fit``.
        '''
        batch_data = to_device(batch_data, self._device)
        _, [A, yA, B, yB] = self.unpack_fit_batch(batch_data)

        A2B, yA2B = self.G_A(A)
        B2A, yB2A = self.G_B(B)
        A2B2A, yA2B2A = self.G_B(A2B)
        B2A2B, yB2A2B = self.G_A(B2A)
        A2A, yA2A = self.G_B(A)
        B2B, yB2B = self.G_A(B)

        B2A_logits = self.D_A(B2A)
        A2B_logits = self.D_B(A2B)
            
        # Generator loss
        A2B_g_loss = self.G_criterion(A2B_logits)
        B2A_g_loss = self.G_criterion(B2A_logits)
        A2B2A_cycle_loss = self.cycle_criterion(A2B2A, A)
        B2A2B_cycle_loss = self.cycle_criterion(B2A2B, B)
        A2A_identity_loss = self.identity_criterion(A2A, A)
        B2B_identity_loss = self.identity_criterion(B2B, B)

        # Classification loss
        A2B_c_loss = self.classification_criterion(yA2B, yA)
        A2B2A_c_loss = self.classification_criterion(yA2B2A, yA)
        A2A_c_loss = self.classification_criterion(yA2A, yA)
        B2A_c_loss = self.classification_criterion(yB2A, yB)
        B2A2B_c_loss = self.classification_criterion(yB2A2B, yB)
        B2B_c_loss = self.classification_criterion(yB2B, yB)

        G_A_loss = A2B_g_loss + 0.1 * A2B_c_loss + (A2B2A_cycle_loss + 0.01 * A2B2A_c_loss) * self.lambda_cycle + (A2A_identity_loss + 0.02 * A2A_c_loss) * self.lambda_identity
        G_B_loss = B2A_g_loss + 0.1 * B2A_c_loss + (B2A2B_cycle_loss + 0.01 * B2A2B_c_loss) * self.lambda_cycle + (B2B_identity_loss + 0.02 * B2B_c_loss) * self.lambda_identity

        G_loss = G_A_loss + G_B_loss
        
        if self._pass_type == "train":
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

        D_A_loss = A_d_loss + B2A_d_loss
        D_B_loss = B_d_loss + A2B_d_loss

        D_loss = D_A_loss + D_B_loss

        if self._pass_type == "train":
            self.backward(D_loss, self.D_optim)
        
        return [A2B, yA2B, B2A, yB2A, A2B2A, yA2B2A, B2A2B, yB2A2B, A2A, yA2A, B2B, yB2B], [G_A_loss, G_B_loss, D_A_loss, D_B_loss]

    def unpack_predict_batch(self, batch_data):
        ''' In CycleGAN setup, we have no inputs, but only two targets: A and B. '''
        A, yA, B, yB = batch_data
        return [], [A, yA, B, yB]

    def predict_batch(self, batch_data, decision_func: Optional[Callable] = None, **decision_func_kw):
        ''' Compute and return the output for a batch. This method should be overridden by subclasses.
        
        The default implementation assumes that ``batch_data`` is a tuple of ``(A, yA, B, yB)`` and that the model
        outputs a 4-tuple ``(A2B, B2A, A2B2A, B2A2B)``.

        NOTE! decision_func is not used in this implementation.
                
        Parameters
        ----------
        batch_data : Any
            Batch data as returned by the dataloader provided to ``predict`` or ``predict_generator``.
        decision_func : Optional[Callable]
            Decision function passed to ``predict`` or ``predict_generator``.. If None, the output of the model is returned.
        **decision_func_kw
            Keyword arguments passed to ``decision_func``, provided to ``predict`` or ``predict_generator``.
        '''
        batch_data = to_device(batch_data, self._device)
        _, [A, yA, B, yB] = self.unpack_predict_batch(batch_data)

        A2B, yA2B = self.G_A(A)
        B2A, yB2A = self.G_B(B)
        A2B2A, yA2B2A = self.G_B(A2B)
        B2A2B, yB2A2B = self.G_A(B2A)
        A2A, yA2A = self.G_B(A)
        B2B, yB2B = self.G_A(B)
        return [A2B, yA2B, B2A, yB2A, A2B2A, yA2B2A, B2A2B, yB2A2B, A2A, yA2A, B2B, yB2B]

    def unpack_score_batch(self, batch_data):
        ''' In CycleGAN setup, we have no inputs, but only two targets: A and B. '''
        A, yA, B, yB = batch_data
        return [], [A, yA, B, yB]
        
    def score_batch(self, batch_data, score_func: Optional[Callable[[Any], Any]] = None, **score_func_kw):
        ''' Compute and return the score for a batch. This method should be overridden by subclasses.
        
        The default implementation assumes that ``batch_data`` is a tuple of ``(A, yA, B, yB)``.
        If ``score_func`` is None, score is the model loss.

        Parameters
        ----------
        batch_data : Any
            Batch data as returned by the dataloader provided to ``score``.
        score_func : Optional[Callable]
            Score function passed to ``score``. If None, the criterion is used by default.
            Takes a tuple of ``(A, B, A2B, B2A, A2B2A, B2A2B, A2A, B2B)`` as input, and returns either a scalar, tuple of scalars, tensor, or tuple of tensors.
        **score_func_kw
            Keyword arguments passed to ``score_func``, provided to ``score``.
        '''
        batch_data = to_device(batch_data, self._device)
        _, [A, yA, B, yB] = self.unpack_score_batch(batch_data)

        A2B, yA2B = self.G_A(A)
        B2A, yB2A = self.G_B(B)
        A2B2A, yA2B2A = self.G_B(A2B)
        B2A2B, yB2A2B = self.G_A(B2A)
        A2A, yA2A = self.G_B(A)
        B2B, yB2B = self.G_A(B)

        if score_func is None:
            B2A_logits = self.D_A(B2A)
            A2B_logits = self.D_B(A2B)
                
            # Generator loss
            A2B_g_loss = self.G_criterion(A2B_logits)
            B2A_g_loss = self.G_criterion(B2A_logits)
            A2B2A_cycle_loss = self.cycle_criterion(A2B2A, A)
            B2A2B_cycle_loss = self.cycle_criterion(B2A2B, B)
            A2A_identity_loss = self.identity_criterion(A2A, A)
            B2B_identity_loss = self.identity_criterion(B2B, B)

            # Classification loss
            A2B_c_loss = self.classification_criterion(yA2B, yA)
            A2B2A_c_loss = self.classification_criterion(yA2B2A, yA)
            A2A_c_loss = self.classification_criterion(yA2A, yA)
            B2A_c_loss = self.classification_criterion(yB2A, yB)
            B2A2B_c_loss = self.classification_criterion(yB2A2B, yB)
            B2B_c_loss = self.classification_criterion(yB2B, yB)

            G_A_loss = A2B_g_loss + 0.1 * A2B_c_loss + (A2B2A_cycle_loss + 0.01 * A2B2A_c_loss) * self.lambda_cycle + (A2A_identity_loss + 0.02 * A2A_c_loss) * self.lambda_identity
            G_B_loss = B2A_g_loss + 0.1 * B2A_c_loss + (B2A2B_cycle_loss + 0.01 * B2A2B_c_loss) * self.lambda_cycle + (B2B_identity_loss + 0.02 * B2B_c_loss) * self.lambda_identity

            G_loss = G_A_loss + G_B_loss

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

            D_A_loss = A_d_loss + B2A_d_loss
            D_B_loss = B_d_loss + A2B_d_loss

            D_loss = D_A_loss + D_B_loss

            score = [G_A_loss.item(), G_B_loss.item(), D_A_loss.item(), D_B_loss.item()]
        else:
            score = score_func(self._to_safe_tensor([A2B, yA2B, B2A, yB2A, A2B2A, yA2B2A, B2A2B, yB2A2B, A2A, yA2A, B2B, yB2B]), self._to_safe_tensor(batch_data), **score_func_kw)
        return score