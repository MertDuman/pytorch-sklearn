# %%
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as tvt
import torch.optim.lr_scheduler as tlrs
from torch.utils.data import DataLoader, Dataset

import matplotlib.pyplot as plt

from pytorch_sklearn.neural_network.nn_base import NeuralNetworkBase
from pytorch_sklearn.neural_network.neural_network import NeuralNetwork
from pytorch_sklearn.neural_network.generative_network import CycleGAN, R2CGAN
from pytorch_sklearn.callbacks.predefined import Verbose, History, EarlyStopping
from pytorch_sklearn.utils.progress_bar import print_progress
from pytorch_sklearn.frameworks.lr_schedulers import *
from pytorch_sklearn.utils.func_utils import to_safe_tensor

from pytorch_sklearn.neural_network.diffusion_network import DiffusionUtils

from sonn.building_blocks import Downsample2d, Upsample2d
from sonn.norm_layers import LayerNormNLP2d
from sonn.superonn_final import SuperONN2d

from PIL import Image

from collections import Iterable as CIterable
from typing import Iterable, Union, List
from pytorch_sklearn.utils.func_utils import to_device

import datetime


### Neural Network ###
# %%
model = nn.Sequential(
    nn.Conv2d(3, 128, 3, padding=1),
    nn.ReLU(),
    nn.Conv2d(128, 128, 3, padding=1),
    nn.Conv2d(128, 128, 3, padding=1),
    nn.Conv2d(128, 128, 3, padding=1),
    nn.Conv2d(128, 128, 3, padding=1),
    nn.Conv2d(128, 128, 3, padding=1),
    nn.Conv2d(128, 128, 3, padding=1),
    nn.Conv2d(128, 128, 3, padding=1),
    nn.Conv2d(128, 128, 3, padding=1),
    nn.Conv2d(128, 128, 3, padding=1),
    nn.Conv2d(128, 128, 3, padding=1),
    nn.Conv2d(128, 128, 3, padding=1),
    nn.Conv2d(128, 128, 3, padding=1),
    nn.ReLU(),
    nn.Conv2d(128, 3, 3, padding=1),
)
optim = torch.optim.Adam(model.parameters(), lr=1e-3)
crit = nn.L1Loss()

net = NeuralNetwork(model, optim, crit)

X = torch.randn(10, 3, 320, 320)
X = (X - X.min()) / (X.max() - X.min())
y = torch.randn(10, 3, 320, 320)
y = (y - y.min()) / (y.max() - y.min())

# %%
net.fit(
    train_X=X,
    train_y=y,
    validate=True,
    val_X=X,
    val_y=y,
    max_epochs=2,
    batch_size=1,
    use_cuda=True,
    callbacks=[Verbose(verbose=3, per_batch=True)],
    metrics={'diff': lambda out, inp: (out - inp[1]).abs().mean()},
)

ypred = net.predict(X, use_cuda=True)

plt.subplot(1, 3, 1)
plt.imshow(to_safe_tensor(X[0]).permute(1, 2, 0))
plt.title('X')
plt.axis('off')
plt.subplot(1, 3, 2)
plt.imshow(to_safe_tensor(ypred[0]).permute(1, 2, 0))
plt.axis('off')
plt.title('ypred')
plt.subplot(1, 3, 3)
plt.imshow(to_safe_tensor(y[0]).permute(1, 2, 0))
plt.title('y')
plt.axis('off')
plt.show()


### CycleGAN ###

# %%
class AbsModule(nn.Module):
    def forward(self, x):
        return torch.abs(x)
    
G_A = nn.Sequential(
    nn.Conv2d(3, 32, 3, padding=1),
    nn.ReLU(),
    nn.Conv2d(32, 3, 3, padding=1),
    nn.ReLU(),
    nn.Conv2d(3, 3, 3, padding=1),
)
G_B = nn.Sequential(
    nn.Conv2d(3, 32, 3, padding=1),
    nn.ReLU(),
    nn.Conv2d(32, 3, 3, padding=1),
    nn.ReLU(),
    nn.Conv2d(3, 3, 3, padding=1),
)
D_A = nn.Sequential(
    AbsModule(),
    nn.Conv2d(3, 32, 3, padding=1),
    nn.MaxPool2d(4),
    nn.ReLU(),
    nn.Conv2d(32, 32, 3, padding=1),
    nn.MaxPool2d(4),
    nn.ReLU(),
    nn.Conv2d(32, 1, 3, padding=1),
    nn.MaxPool2d(2),
    nn.AdaptiveAvgPool2d(1),
    nn.Flatten(),
)
D_B = nn.Sequential(
    AbsModule(),
    nn.Conv2d(3, 32, 3, padding=1),
    nn.MaxPool2d(4),
    nn.ReLU(),
    nn.Conv2d(32, 32, 3, padding=1),
    nn.MaxPool2d(4),
    nn.ReLU(),
    nn.Conv2d(32, 1, 3, padding=1),
    nn.MaxPool2d(2),
    nn.AdaptiveAvgPool2d(1),
    nn.Flatten(),
)

G_optim = torch.optim.Adam(list(G_A.parameters()) + list(G_B.parameters()), lr=2e-4)
D_optim = torch.optim.Adam(list(D_A.parameters()) + list(D_B.parameters()), lr=2e-4)

cyclegan = CycleGAN(G_A, G_B, D_A, D_B, G_optim, D_optim)

class STDMetric(nn.Module):
    def __init__(self):
        super().__init__()
        self.stdA = []
        self.stdB = []
        self.stdA2B = []
        self.stdB2A = []

    def forward(self, batch_out, batch_data):
        A2B, B2A, *_ = batch_out
        A, B = batch_data
        self.stdA.append(A.std().item())
        self.stdB.append(B.std().item())
        self.stdA2B.append(A2B.std().item())
        self.stdB2A.append(B2A.std().item())
        return 0
    
class CycleGANDataset(Dataset):
    def __init__(self):
        self.A = torch.randn(10, 3, 32, 32) * .1
        self.B = torch.randn(10, 3, 32, 32) * .8

    def __len__(self):
        return 10
    
    def __getitem__(self, index):
        return self.A[index], self.B[index]
    
# %%
    
cyclegan.fit(
    train_X=CycleGANDataset(),
    max_epochs=700,
    use_cuda=True,
    callbacks=[Verbose(per_batch=False)],
    metrics={'std': STDMetric()},
)

# %%

A2B, B2A, *_ = cyclegan.predict(CycleGANDataset(), use_cuda=True)

print(A2B.std(), B2A.std())

plt.plot(cyclegan._metrics['std'].stdA, label='A')
plt.plot(cyclegan._metrics['std'].stdB, label='B')
plt.plot(cyclegan._metrics['std'].stdA2B, label='A2B')
plt.plot(cyclegan._metrics['std'].stdB2A, label='B2A')
plt.legend()
plt.show()


### GAN ###
# %%
nn.ConvTranspose2d( 1, 3 * 8, 4, 1, 0)(torch.randn(1, 1, 1, 1)).shape











# %%
### TRAINING TRACKER V2 PREPARING AND SAVING ###
from pytorch_sklearn.utils.training_tracker_v2 import TrainingTrackerV2
from pytorch_sklearn.callbacks import *

class TempDS(Dataset):
    def __init__(self, train=True, only_X=False, only_y=False):
        self.data = torch.randn(10, 1, 32, 32)
        self.train = train
        self.only_X = only_X
        self.only_y = only_y

    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, index):
        if self.only_X:
            return self.data[index]
        if self.only_y:
            return self.data[index]
        return self.data[index], self.data[index]

tt = TrainingTrackerV2()

tt.configure(
    folder='trainings_deldeldel',
    hyperparameters=dict(
        model='',
        model_ctor='',
        optimizer='',
        optimizer_ctor='',
        criterion='',
        criterion_ctor='',
        train_ds='',
        train_ds_ctor='',
        test_ds='',
        test_ds_ctor='',
        score_ds='',
        score_ds_ctor='',
        tot_params=0,
        tot_neurons=0,
        dataset_size=0,
        batch_size=0,
        net="",
        net_ctor="{}",
    ),
    metrics=dict(
        train_psnr=0,
        train_losses="[]",
        test_psnr=0,
        test_ssim=0,
        test_losses="[]",
    ),
    misc=dict(
        setup_id='',
        start_date='',
        end_date='',
        gpu_name='',
        total_epochs=0,
        failed='',
        args='',
        callbacks='[]',
    )
)

lr = 1e-3
train_ds_ctor = dict(train=True)
train_ds = TempDS(**train_ds_ctor)
train_dl = DataLoader(train_ds, batch_size=1, shuffle=True)

test_ds_ctor = dict(train=False)
test_ds = TempDS(**test_ds_ctor)
test_dl = DataLoader(test_ds, batch_size=1, shuffle=False)

score_ds_ctor = dict(train=False, only_X=True)
score_ds = TempDS(**score_ds_ctor)
score_dl = DataLoader(score_ds, batch_size=1, shuffle=False)

model_ctor = dict()
model = nn.Sequential(
    nn.Conv2d(1, 32, 3, padding=1),
    nn.ReLU(),
    nn.Conv2d(32, 32, 3, padding=1),
    nn.ReLU(),
    nn.Conv2d(32, 1, 3, padding=1),
)

optim_ctor = dict(lr=lr)
optim = torch.optim.Adam(model.parameters(), **optim_ctor)
crit_ctor = dict()
crit = nn.MSELoss(**crit_ctor)

net_ctor = dict()
net = NeuralNetwork(model, optim, crit, **net_ctor)

hypers=dict(
    model=type(model),
    model_ctor=model_ctor,
    optimizer=type(optim),
    optimizer_ctor=optim_ctor,
    criterion=type(crit),
    criterion_ctor=crit_ctor,
    train_ds=type(train_ds),
    train_ds_ctor=train_ds_ctor,
    test_ds=type(test_ds),
    test_ds_ctor=test_ds_ctor,
    score_ds=type(score_ds),
    score_ds_ctor=score_ds_ctor,
    tot_params=sum(p.numel() for p in model.parameters()),
    tot_neurons=sum(p.numel() for p in model.parameters() if len(p.shape) == 1),
    dataset_size=len(train_ds),
    batch_size=len(train_dl),
    net=type(net),
    net_ctor=net_ctor,
)
saveid = tt.initialize_training(
    hyperparameters=hypers,
    misc=dict(
        setup_id=0,
        start_date=datetime.datetime.now().strftime("%d/%m/%Y %H:%M:%S"),
        gpu_name=torch.cuda.get_device_name(torch.cuda.current_device()) if torch.cuda.is_available() else "CPU",
    )
)

savefolder = tt.get_savefolder()
if not os.path.exists(savefolder):
    os.makedirs(savefolder)

torch.save(
    hypers,
    osj(savefolder, 'hyperparameters.pth')
)

callbacks = [
    Verbose(), 
    WeightCheckpoint(tracked='train_loss', mode='min', savepath=osj(savefolder, 'weights.pth'), save_per_epoch=True),
    NetCheckpoint(savepath=osj(savefolder, 'net.pth'), per_epoch=1),
]

net.fit(
    train_X=train_dl,
    validate=True,
    val_X=test_dl,
    max_epochs=10,
    callbacks=callbacks,
    metrics={'train_loss': lambda out, inp: crit(out, inp[1])},
)

tt.finalize_training(
    metrics=dict(
        train_psnr=0,
        train_losses="[]",
        test_psnr=0,
        test_ssim=0,
        test_losses="[]",
    ),
    misc=dict(
        end_date=datetime.datetime.now().strftime("%d/%m/%Y %H:%M:%S"),
        total_epochs=net._epoch,
        callbacks=[type(cb) for cb in callbacks],
    )
)

# %%
### TRAINING TRACKER V2 LOADING ###
class TempDS(Dataset):
    def __init__(self, train=True, only_X=False, only_y=False):
        self.data = torch.randn(10, 1, 32, 32)
        self.train = train
        self.only_X = only_X
        self.only_y = only_y

    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, index):
        if self.only_X:
            return self.data[index]
        if self.only_y:
            return self.data[index]
        return self.data[index], self.data[index]

load_checkpoint = True
checkpoint_id = 0

tt = TrainingTrackerV2()

tt.configure(
    folder='trainings_deldeldel',
    hyperparameters=dict(
        model='',
        model_ctor='',
        optimizer='',
        optimizer_ctor='',
        criterion='',
        criterion_ctor='',
        train_ds='',
        train_ds_ctor='',
        test_ds='',
        test_ds_ctor='',
        score_ds='',
        score_ds_ctor='',
        tot_params=0,
        tot_neurons=0,
        dataset_size=0,
        batch_size=0,
        net="",
        net_ctor="{}",
    ),
    metrics=dict(
        train_psnr=0,
        train_losses="[]",
        test_psnr=0,
        test_ssim=0,
        test_losses="[]",
    ),
    misc=dict(
        setup_id='',
        start_date='',
        end_date='',
        gpu_name='',
        total_epochs=0,
        failed='',
        args='',
        callbacks='[]',
    )
)

if load_checkpoint:
    savefolder = tt.get_savefolder(checkpoint_id)
    hypers = torch.load(osj(savefolder, 'hyperparameters.pth'))

    train_ds_ctor = hypers['train_ds_ctor']
    train_ds_cls = hypers['train_ds']
    train_ds = train_ds_cls(**train_ds_ctor)
    train_dl = DataLoader(train_ds, batch_size=1, shuffle=True)

    test_ds_ctor = hypers['test_ds_ctor']
    test_ds_cls = hypers['test_ds']
    test_ds = test_ds_cls(**test_ds_ctor)
    test_dl = DataLoader(test_ds, batch_size=1, shuffle=False)

    score_ds_ctor = hypers['score_ds_ctor']
    score_ds_cls = hypers['score_ds']
    score_ds = score_ds_cls(**score_ds_ctor)
    score_dl = DataLoader(score_ds, batch_size=1, shuffle=False)

    model_ctor = hypers['model_ctor']
    model_cls = hypers['model']
    model = model_cls(
        nn.Conv2d(1, 32, 3, padding=1),
        nn.ReLU(),
        nn.Conv2d(32, 32, 3, padding=1),
        nn.ReLU(),
        nn.Conv2d(32, 1, 3, padding=1),
    )

    optim_ctor = hypers['optimizer_ctor']
    optim_cls = hypers['optimizer']
    optim = optim_cls(model.parameters(), **optim_ctor)

    crit_ctor = hypers['criterion_ctor']
    crit_cls = hypers['criterion']
    crit = crit_cls(**crit_ctor)

    net_ctor = hypers['net_ctor']
    net_cls: NeuralNetwork = hypers['net']
    net = net_cls(model, optim, crit, **net_ctor)

if not load_checkpoint:
    hypers=dict(
        model=type(model),
        model_ctor=model_ctor,
        optimizer=type(optim),
        optimizer_ctor=optim_ctor,
        criterion=type(crit),
        criterion_ctor=crit_ctor,
        train_ds=type(train_ds),
        train_ds_ctor=train_ds_ctor,
        test_ds=type(test_ds),
        test_ds_ctor=test_ds_ctor,
        score_ds=type(score_ds),
        score_ds_ctor=score_ds_ctor,
        tot_params=sum(p.numel() for p in model.parameters()),
        tot_neurons=sum(p.numel() for p in model.parameters() if len(p.shape) == 1),
        dataset_size=len(train_ds),
        batch_size=len(train_dl),
        net=type(net),
        net_ctor=net_ctor,
    )
    saveid = tt.initialize_training(
        hyperparameters=hypers,
        misc=dict(
            setup_id=0,
            start_date=datetime.datetime.now().strftime("%d/%m/%Y %H:%M:%S"),
            gpu_name=torch.cuda.get_device_name(torch.cuda.current_device()) if torch.cuda.is_available() else "CPU",
        )
    )

    savefolder = tt.get_savefolder()
    if not os.path.exists(savefolder):
        os.makedirs(savefolder)

    torch.save(
        hypers,
        osj(savefolder, 'hyperparameters.pth')
    )

else:
    tt.set_current_training_from_index(checkpoint_id)

callbacks = [
    Verbose(), 
    WeightCheckpoint(tracked='train_loss', mode='min', savepath=osj(savefolder, 'weights.pth'), save_per_epoch=True),
    NetCheckpoint(savepath=osj(savefolder, 'net.pth'), per_epoch=1),
]

if load_checkpoint:
    net_cls.load_class(net, callbacks, loadpath=osj(savefolder, 'net.pth'))

net.fit(
    train_X=train_dl,
    validate=True,
    val_X=test_dl,
    max_epochs=10,
    callbacks=callbacks,
    metrics={'train_loss': lambda out, inp: crit(out, inp[1])},
)

tt.finalize_training(
    metrics=dict(
        train_psnr=0,
        train_losses="[]",
        test_psnr=0,
        test_ssim=0,
        test_losses="[]",
    ),
    misc=dict(
        end_date=datetime.datetime.now().strftime("%d/%m/%Y %H:%M:%S"),
        total_epochs=net._epoch,
        callbacks=[type(cb) for cb in callbacks],
    )
)
















# %%
### TRAINING TRACKER V2 MERGED ###
from pytorch_sklearn.utils.training_tracker_v2 import TrainingTrackerV2
from pytorch_sklearn.callbacks import *

class TempDS(Dataset):
    def __init__(self, train=True, only_X=False, only_y=False):
        self.data = torch.randn(10, 1, 32, 32)
        self.train = train
        self.only_X = only_X
        self.only_y = only_y

    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, index):
        if self.only_X:
            return self.data[index]
        if self.only_y:
            return self.data[index]
        return self.data[index], self.data[index]
    
train = True
test = not train
load_checkpoint = False
checkpoint_id = 0

tt = TrainingTrackerV2()

tt.configure(
    folder='trainings_deldeldel',
    hyperparameters=dict(
        model='',
        model_ctor='',
        optimizer='',
        optimizer_ctor='',
        criterion='',
        criterion_ctor='',
        train_ds='',
        train_ds_ctor='',
        test_ds='',
        test_ds_ctor='',
        score_ds='',
        score_ds_ctor='',
        tot_params=0,
        tot_neurons=0,
        dataset_size=0,
        batch_size=0,
        net="",
        net_ctor="{}",
    ),
    metrics=dict(
        train_psnr=0,
        train_losses="[]",
        test_psnr=0,
        test_ssim=0,
        test_losses="[]",
    ),
    misc=dict(
        setup_id='',
        start_date='',
        end_date='',
        gpu_name='',
        total_epochs=0,
        failed='',
        args='',
        callbacks='[]',
    )
)

lr = 1e-6
hypers=dict(
    model=nn.Sequential,
    model_ctor=dict(),
    optimizer=torch.optim.Adam,
    optimizer_ctor=dict(lr=lr),
    criterion=nn.MSELoss,
    criterion_ctor=dict(),
    train_ds=TempDS,
    train_ds_ctor=dict(train=True),
    test_ds=TempDS,
    test_ds_ctor=dict(train=False),
    score_ds=TempDS,
    score_ds_ctor=dict(train=False, only_X=True),
    net=NeuralNetwork,
    net_ctor=dict(),
)

if load_checkpoint:
    tt.set_current_training_from_index(checkpoint_id)
    savefolder = tt.get_savefolder(checkpoint_id)
    hypers = torch.load(osj(savefolder, 'hyperparameters.pth'))

train_ds_ctor = hypers['train_ds_ctor']
train_ds_cls = hypers['train_ds']
train_ds = train_ds_cls(**train_ds_ctor)
train_dl = DataLoader(train_ds, batch_size=1, shuffle=True)

test_ds_ctor = hypers['test_ds_ctor']
test_ds_cls = hypers['test_ds']
test_ds = test_ds_cls(**test_ds_ctor)
test_dl = DataLoader(test_ds, batch_size=1, shuffle=False)

score_ds_ctor = hypers['score_ds_ctor']
score_ds_cls = hypers['score_ds']
score_ds = score_ds_cls(**score_ds_ctor)
score_dl = DataLoader(score_ds, batch_size=1, shuffle=False)

model_ctor = hypers['model_ctor']
model_cls = hypers['model']
model = model_cls(
    nn.Conv2d(1, 32, 3, padding=1),
    nn.ReLU(),
    nn.Conv2d(32, 32, 3, padding=1),
    nn.ReLU(),
    nn.Conv2d(32, 1, 3, padding=1),
)

optim_ctor = hypers['optimizer_ctor']
optim_cls = hypers['optimizer']
optim = optim_cls(model.parameters(), **optim_ctor)

crit_ctor = hypers['criterion_ctor']
crit_cls = hypers['criterion']
crit = crit_cls(**crit_ctor)

net_ctor = hypers['net_ctor']
net_cls: NeuralNetwork = hypers['net']
net = net_cls(model, optim, crit, **net_ctor)

hypers.update(
    tot_params=sum(p.numel() for p in model.parameters()),
    tot_neurons=sum(p.numel() for p in model.parameters() if len(p.shape) == 1),
    dataset_size=len(train_ds),
    batch_size=len(train_dl),
)

if not load_checkpoint:
    saveid = tt.initialize_training(
        hyperparameters=hypers,
        misc=dict(
            setup_id=0,
            start_date=datetime.datetime.now().strftime("%d/%m/%Y %H:%M:%S"),
            gpu_name=torch.cuda.get_device_name(torch.cuda.current_device()) if torch.cuda.is_available() else "CPU",
        )
    )
    savefolder = tt.get_savefolder()
    if not os.path.exists(savefolder):
        os.makedirs(savefolder)

torch.save(
    hypers,
    osj(savefolder, 'hyperparameters.pth')
)

callbacks = [
    Verbose(), 
    WeightCheckpoint(tracked='train_loss', mode='min', savepath=osj(savefolder, 'weights.pth'), save_per_epoch=True),
    NetCheckpoint(savepath=osj(savefolder, 'net.pth'), per_epoch=1),
]

if load_checkpoint:
    net_cls.load_class(net, callbacks, loadpath=osj(savefolder, 'net.pth'))

if train:
    net.fit(
        train_X=train_dl,
        validate=True,
        val_X=test_dl,
        max_epochs=10,
        use_cuda=torch.cuda.is_available(),
        callbacks=callbacks,
        metrics={'train_loss': lambda out, inp: crit(out, inp[1])},
    )

    metrics = dict(
        train_psnr=0,
        train_losses="[]",
        test_psnr=0,
        test_ssim=0,
        test_losses="[]",
    )
else:
    out_gen = net.predict_generator(
        score_dl,
        use_cuda=torch.cuda.is_available(),
    )

    for i, out in enumerate(out_gen):
        if i >= 5:
            out_gen.close()
            break

        print(out.shape)

    metrics = dict(
        train_psnr=0,
        train_losses="[]",
        test_psnr=0,
        test_ssim=0,
        test_losses="[]",
    )


tt.finalize_training(
    metrics=metrics,
    misc=dict(
        end_date=datetime.datetime.now().strftime("%d/%m/%Y %H:%M:%S"),
        total_epochs=net._epoch,
        callbacks=[type(cb) for cb in callbacks],
    )
)