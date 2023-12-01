import torch
import torch.nn as nn
import torch.nn.functional as F


def get_linear_unit(unit: str = 'relu'):
    """
    Get linear unit from string.
    
    Parameters
    ----------
    unit: one of [relu, leaky_relu, gelu, elu, celu, selu, prelu, rrelu, none].
    """
    unit = unit.lower()
    if unit == 'relu':
        return nn.ReLU
    elif unit == 'leaky_relu':
        return nn.LeakyReLU
    elif unit == 'gelu':
        return nn.GELU
    elif unit == 'elu':
        return nn.ELU
    elif unit == 'celu':
        return nn.CELU
    elif unit == 'selu':
        return nn.SELU
    elif unit == 'prelu':
        return nn.PReLU
    elif unit == 'rrelu':
        return nn.RReLU
    elif unit == 'none':
        return nn.Identity
    else:
        raise ValueError(f'Unknown unit: {unit}, must be one of [relu, leaky_relu, gelu, elu, celu, selu, prelu, rrelu, none]')
