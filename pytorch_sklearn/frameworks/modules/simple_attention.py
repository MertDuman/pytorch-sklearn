import torch
import torch.nn as nn
import torch.nn.functional as F


class AvgChannelAttention2d(nn.Module):
    """
    The average channel attention module from https://arxiv.org/abs/1807.06521.
    The output is a channel attention map that is generally multiplied with the input.
    """
    def __init__(self, dim, bias=True):
        super().__init__()

        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.conv = nn.Conv2d(dim, dim, kernel_size=1, padding=0, bias=bias)

    def forward(self, x):
        return self.conv(self.avgpool(x))

# An alias for AvgChannelAttention2d
SimpleChannelAttention2d = AvgChannelAttention2d


class MaxChannelAttention2d(nn.Module):
    """
    The max channel attention module from https://arxiv.org/abs/1807.06521.
    The output is a channel attention map that is generally multiplied with the input.
    """
    def __init__(self, dim, bias=True):
        super().__init__()

        self.maxpool = nn.AdaptiveMaxPool2d(1)
        self.conv = nn.Conv2d(dim, dim, kernel_size=1, padding=0, bias=bias)

    def forward(self, x):
        return self.conv(self.maxpool(x))


class AvgMaxChannelAttention2d(nn.Module):
    """
    The average + max channel attention module from https://arxiv.org/abs/1807.06521.
    The output is a channel attention map that is generally multiplied with the input.
    """
    def __init__(self, dim, bias=True):
        super().__init__()

        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.maxpool = nn.AdaptiveMaxPool2d(1)
        self.conv = nn.Conv2d(dim, dim, kernel_size=1, padding=0, bias=bias)

    def forward(self, x):
        return self.conv(self.avgpool(x) + self.maxpool(x))


class AvgSpatialAttention2d(nn.Module):
    """
    The average spatial attention module from https://arxiv.org/abs/1807.06521.
    The output is a spatial attention map that is generally multiplied with the input.
    """
    def __init__(self, dim, bias=True):
        super().__init__()

        self.conv = nn.Conv2d(dim, 1, kernel_size=3, padding=1, bias=bias)

    def forward(self, x):
        return self.conv(torch.mean(x, dim=1, keepdim=True))


class MaxSpatialAttention2d(nn.Module):
    """
    The max spatial attention module from https://arxiv.org/abs/1807.06521.
    The output is a spatial attention map that is generally multiplied with the input.
    """
    def __init__(self, dim, bias=True):
        super().__init__()

        self.conv = nn.Conv2d(dim, 1, kernel_size=3, padding=1, bias=bias)

    def forward(self, x):
        return self.conv(torch.amax(x, dim=1, keepdim=True))


class AvgMaxSpatialAttention2d(nn.Module):
    """
    The average + max spatial attention module from https://arxiv.org/abs/1807.06521.
    The output is a spatial attention map that is generally multiplied with the input.
    """
    def __init__(self, dim, bias=True):
        super().__init__()

        self.conv = nn.Conv2d(dim, 1, kernel_size=3, padding=1, bias=bias)

    def forward(self, x):
        return self.conv(torch.mean(x, dim=1, keepdim=True) + torch.amax(x, dim=1, keepdim=True))