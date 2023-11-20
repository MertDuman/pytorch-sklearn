import torch
import torch.nn as nn


class LayerNormNLP2d(nn.Module):
    """
    The name NLP implies that this layer norm does not normalize the C x H x W dimensions (default), but only the C dimension (NLP).
    https://i.stack.imgur.com/1JdN6.png

    This is useful for NLP tasks, where the input is a sequence of word embeddings, and the C dimension is the embedding dimension.
    But it has also been used in image tasks, where the C dimension is the number of channels.
    """
    def __init__(self, channels, affine=True):
        super().__init__()
        self.layer_norm = nn.LayerNorm(channels, elementwise_affine=affine)

    def forward(self, x):
        x = x.permute(0, 2, 3, 1)  # n c h w -> n h w c
        x = self.layer_norm(x)
        x = x.permute(0, 3, 1, 2)  # n h w c -> n c h w
        return x


class LayerNormNLP1d(nn.Module):
    """
    The name NLP implies that this layer norm does not normalize the C x H x W dimensions (default), but only the C dimension (NLP).
    https://i.stack.imgur.com/1JdN6.png

    This is useful for NLP tasks, where the input is a sequence of word embeddings, and the C dimension is the embedding dimension.
    But it has also been used in image tasks, where the C dimension is the number of channels.
    """
    def __init__(self, channels, affine=True):
        super().__init__()
        self.layer_norm = nn.LayerNorm(channels, elementwise_affine=affine)

    def forward(self, x):
        x = x.permute(0, 2, 1)  # n c d -> n d c
        x = self.layer_norm(x)
        x = x.permute(0, 2, 1)  # n d c -> n c d
        return x