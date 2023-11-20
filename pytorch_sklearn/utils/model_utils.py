import torch
import torch.nn as nn
import torch.autograd as AG
import numpy as np
import matplotlib.pyplot as plt
from typing import Union, Iterable


def get_num_params(model):
    ''' Counts all parameters in a model. '''
    return sum(p.numel() for p in model.parameters())


def get_num_layers(model, layer_types: Iterable):
    ''' Counts all layers of a certain type in a model. '''
    return sum(isinstance(m, tuple(layer_types)) for m in model.modules())


def print_model_param_info(model, cell_len=30, cell_algn="<", in_filters=None, ex_filters=None, ex_zero=False, ex_container=False, clip=True):
    ''' Prints model parameter information. '''
    in_filters = None if in_filters is None else set(in_filters)
    ex_filters = None if ex_filters is None else set(ex_filters)
    table_width = cell_len * 3 + 2

    # First pass to calculate total params
    total_params = 0
    total_layers = 0
    for n, m in model.named_modules():
        m_params = get_num_params(m)
        c_params = sum(get_num_params(c) for c in m.children())
        param_diff = m_params - c_params

        # Is container if it has no parameters but has children
        is_container = param_diff == 0 and c_params != 0
        
        if ex_container and is_container:
            continue

        if ex_zero and m_params == 0:
            continue

        type_name = type(m).__name__
        if in_filters is not None and (type_name not in in_filters and type(m) not in in_filters):
            continue
        if ex_filters is not None and (type_name in ex_filters or type(m) in ex_filters):
            continue
            
        total_params += param_diff
        total_layers += 1

    # Second pass to print info
    has_seperator = False
    separated_now = False
    print(f'{"Layer":{cell_algn}{cell_len}} {"Structure":{cell_algn}{cell_len}} {"Params":{cell_algn}{cell_len}}')
    print(f"{type(model).__name__:{cell_algn}{cell_len}} {total_layers:{cell_algn}{cell_len}} {total_params:{cell_algn}{cell_len}}")
    for n, m in model.named_modules():
        separated_now = False

        m_params = get_num_params(m)
        c_params = sum(get_num_params(c) for c in m.children())
        param_diff = m_params - c_params

        # Is container if it has no parameters but has children
        is_container = param_diff == 0 and c_params != 0

        # Print separator if no dot in name and we haven't printed a separator yet
        if n.count('.') == 0 and not has_seperator:
            print("-" * table_width)
            has_seperator = True
            separated_now = True

        if ex_container and param_diff == 0 and m_params != 0:
            continue

        if ex_zero and m_params == 0: 
            continue

        type_name = type(m).__name__
        if in_filters is not None and (type_name not in in_filters and type(m) not in in_filters):
            continue
        if ex_filters is not None and (type_name in ex_filters or type(m) in ex_filters):
            continue

        # If we passed all filters, and this is not the iteration that printed a separator, we no longer have a separator
        if not separated_now:
            has_seperator = False
        
        if clip and len(n) >= cell_len:
            n = n[:cell_len-3] + '...'
        if clip and len(type_name) >= cell_len: 
            type_name = type_name[:cell_len-3] + '...'

        # Containers
        final_params = param_diff
        if is_container:
            final_params = f"{m_params} - C"
        print(f"{type_name:{cell_algn}{cell_len}} {n:{cell_algn}{cell_len}} {final_params:{cell_algn}{cell_len}}")


def get_receptive_field(x: torch.Tensor, model: torch.nn.Module, target_output: int = None, absnorm=False, oneminus=False, target_pixels: torch.Tensor = None):
    '''
    Calculates the receptive field of a model by calculating the derivative of the center output pixel w.r.t input x.

    Parameters
    ----------
    x
        Input tensor of shape (1, C, H, W) where C, H, W are suitable for the model. This makes most sense when x is an image from the model's dataset.
        If unsure, pass torch.ones(1, C, H, W) * 0.01.
    model
        The model to calculate the receptive field for.
    target_idx
        If the model has multiple outputs, this specifies which output to calculate the receptive field for.
    absnorm
        If True, the gradient is normalized to [0, 1] by taking the absolute value and normalizing it.
    oneminus
        If True, returns 1 - normalized gradient. This sets absnorm to True as well.
    target_pixels
        By default, the receptive field is calculated for the center pixel of the output. If target_pixels is specified, the receptive field is calculated 
        for these pixels instead. target_pixels must match the exact shape of the model output. This lets you specify multiple pixels as well.
    '''
    assert x.ndim == 4, "x must be of shape (1, C, H, W) where C, H, W are suitable for the model"
    
    if oneminus:
        absnorm = True

    # Input requires grad
    x = x.requires_grad_(True)
    # y has grad_fn and requires_grad
    y = model(x)

    if target_output is not None:
        y = y[target_output]

    if target_pixels is None:
        grd = torch.zeros_like(y)
        grd[:, :, y.shape[2] // 2, y.shape[3] // 2] = 1
    else:
        grd = target_pixels
    
    # grad is calculated, but model weights are untouched, their grad fields are not filled, and y's computation graph is freed
    grad = AG.grad(y, x, grd)[0]

    if absnorm:
        grad = torch.abs(grad)
        grad = (grad - grad.amin(dim=(2,3), keepdim=True)) / (grad.amax(dim=(2,3), keepdim=True) - grad.amin(dim=(2,3), keepdim=True))

    if oneminus:
        grad = 1 - grad
    return grad


def plot_receptive_fields(*models: torch.nn.Module, x: torch.Tensor, **kwargs):
    '''
    Streamlines plotting receptive fields for multiple models. Plots the receptive field of each model in a grid.

    Parameters
    ----------
    models
        The models to plot the receptive fields for.
    x
        Input tensor of shape (1, C, H, W) where C, H, W are suitable for the models. This makes most sense when x is an image from the models' dataset.
    kwargs
        Keyword arguments to pass to get_receptive_field.
    '''
    num_models = len(models)
    # Find best grid size by finding the largest number that divides num_models. If there is no such number, use the largest square that fits and delete the extra axes.
    rows, cols = find_optimal_grid_size(num_models, fit_rect_if_above=2)
    fig, axs = plt.subplots(rows, cols, figsize=(7 * cols, 5 * rows), squeeze=False)

    for i in range(rows * cols):
        if i >= num_models:
            fig.delaxes(axs[i // cols, i % cols])
            continue
        model = models[i]
        rf = get_receptive_field(x, model, **kwargs)
        ax = axs[i // cols, i % cols]
        ax.imshow(rf[0].permute(1, 2, 0).cpu(), interpolation='none')
        ax.set_title(type(model).__name__)
    return fig, axs


def find_optimal_grid_size(N, aspect_ratio: Union[int, tuple] = 1, fit_rect_if_above=None):
    '''
    Finds the grid size that best matches the aspect_ratio. By default, the aspect ratio is 1, which means the grid is as square as possible. If you're
    plotting patches of an image, for example, you can set the aspect ratio to the aspect ratio of the image to get a grid that matches the image.

    Parameters
    ----------
    N
        Number of elements to fit in the grid.
    aspect_ratio
        A float for the aspect ratio or a 2-tuple for (height, width).
    fit_rect_if_above
        If the aspect ratio of the best fit is above this value, it will be changed to sqrt(N) x sqrt(N) + 1. 
        This will give you unused axes, but will stop the grid from being too long or too tall.
        The aspect ratio is always converted to > 1 before this check is done.
        Defaults to None (or inf), which means this behavior is off.
    '''
    fit_rect_if_above = float('inf') if fit_rect_if_above is None else fit_rect_if_above
    
    if isinstance(aspect_ratio, tuple):
        aspect_ratio = aspect_ratio[1] / aspect_ratio[0]

    if fit_rect_if_above < 1:
        fit_rect_if_above = 1 / fit_rect_if_above

    did_flip = False
    if aspect_ratio < 1:
        did_flip = True
        aspect_ratio = 1 / aspect_ratio

    best_fit = float('inf')
    best_ratio = float('inf')
    for height, width in ((r, N // r) for r in range(1, int(np.sqrt(N)) + 1) if N % r == 0):
        cur_aspect_ratio = width / height

        fit = abs(cur_aspect_ratio - aspect_ratio)
        
        if fit <= best_fit:
            best_fit = fit
            best_ratio = cur_aspect_ratio
            best_height, best_width = height, width
            if did_flip:
                best_height, best_width = width, height
    
    if best_ratio > fit_rect_if_above and N != 1:
        best_height = int(np.sqrt(N))
        best_width = N // best_height + 1
    return best_height, best_width