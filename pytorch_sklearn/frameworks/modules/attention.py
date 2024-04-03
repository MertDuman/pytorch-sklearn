import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange

from pytorch_sklearn.frameworks.modules.nonlinearity import SimpleGate
from pytorch_sklearn.frameworks.modules.norm_layers import LayerNormNLP2d


## Restormer Style ##
class DWConvBlock(nn.Module):
    def __init__(self, in_c, dim, bias, nonlinearity='gelu'):
        super().__init__()

        self.embed = nn.Conv2d(in_c, dim * 2, kernel_size=1, bias=bias)
        self.dwconv = nn.Conv2d(dim * 2, dim * 2, kernel_size=3, padding=1, groups=dim * 2, bias=bias)
        self.deembed = nn.Conv2d(dim, in_c, kernel_size=1, bias=bias)
        self.simple_gate = SimpleGate(nonlinearity)

    def forward(self, x):
        x = self.embed(x)
        x = self.dwconv(x)
        x = self.simple_gate(x)
        x = self.deembed(x)
        return x


class MultiHeadAttention2d(nn.Module):
    def __init__(self, dim, num_heads, bias):
        super().__init__()
        self.num_heads = num_heads
        self.temperature = nn.Parameter(torch.ones(num_heads, 1, 1))

        self.qkv = nn.Conv2d(dim, dim * 3, kernel_size=1, bias=bias)
        self.qkv_dwconv = nn.Conv2d(dim * 3, dim * 3, kernel_size=3, stride=1, padding=1, groups=dim * 3, bias=bias)
        self.project_out = nn.Conv2d(dim, dim, kernel_size=1, bias=bias)

    def forward(self, x):
        b, c, h, w = x.shape

        qkv = self.qkv_dwconv(self.qkv(x))
        q, k, v = qkv.chunk(3, dim=1)

        q = rearrange(q, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        k = rearrange(k, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        v = rearrange(v, 'b (head c) h w -> b head c (h w)', head=self.num_heads)

        q = torch.nn.functional.normalize(q, dim=-1)
        k = torch.nn.functional.normalize(k, dim=-1)

        attn = (q @ k.transpose(-2, -1)) * self.temperature
        attn = attn.softmax(dim=-1)

        out = (attn @ v)

        out = rearrange(out, 'b head c (h w) -> b (head c) h w', head=self.num_heads, h=h, w=w)

        out = self.project_out(out)
        return out


class TransformerBlock(nn.Module):
    def __init__(self, dim, num_heads, dwdim, bias):
        super().__init__()

        self.norm1 = LayerNormNLP2d(dim)
        self.attn = MultiHeadAttention2d(dim, num_heads, bias)
        self.norm2 = LayerNormNLP2d(dim)
        self.dwconv = DWConvBlock(dim, dwdim, bias)

    def forward(self, x):
        x = x + self.attn(self.norm1(x))
        x = x + self.dwconv(self.norm2(x))
        return x


## Stable Diffusion Style ##
class SelfAttention2d(nn.Module):
    def __init__(self, dim, num_heads, bias):
        super().__init__()
        self.num_heads = num_heads
        self.mha = nn.MultiheadAttention(dim, num_heads, bias=bias)  # NOT batch first ???
        self.ln = nn.LayerNorm(dim)
        self.ff_self = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, dim),
            nn.GELU(),
            nn.Linear(dim, dim),
        )

    def forward(self, x):
        N, C, H, W = x.shape
        x = rearrange(x, 'n c h w -> n (h w) c')
        x = self.ln(x)
        attn, _ = self.mha(x, x, x)
        attn = attn + x
        attn = self.ff_self(attn) + attn
        attn = rearrange(attn, 'n (h w) c -> n c h w', h=H, w=W)
        return attn
    
