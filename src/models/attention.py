"""Attention / MLP block used in the from-scratch U-Net bottleneck."""

import torch.nn as nn
import torch.nn.functional as F


class SelfAttention(nn.Module):
    """Multi-head self-attention over a [B, L, C] sequence (uses fused SDPA)."""

    def __init__(self, dim: int, num_heads: int = 4, attn_p: float = 0.0, proj_p: float = 0.0):
        super().__init__()
        assert dim % num_heads == 0, "dim must be divisible by num_heads"
        self.num_heads = num_heads
        self.head_dim = dim // num_heads

        self.qkv = nn.Linear(dim, dim * 3)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_p)
        self.attn_p = attn_p

    def forward(self, x):
        b, n, c = x.shape
        qkv = self.qkv(x).reshape(b, n, 3, self.num_heads, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)            # [3, B, heads, N, head_dim]
        q, k, v = qkv[0], qkv[1], qkv[2]
        x = F.scaled_dot_product_attention(q, k, v, dropout_p=self.attn_p if self.training else 0.0)
        x = x.transpose(1, 2).reshape(b, n, c)
        return self.proj_drop(self.proj(x))


class MLP(nn.Module):
    def __init__(self, dim: int, mlp_ratio: int = 2, p: float = 0.0):
        super().__init__()
        self.fc1 = nn.Linear(dim, dim * mlp_ratio)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(dim * mlp_ratio, dim)
        self.drop = nn.Dropout(p)

    def forward(self, x):
        return self.drop(self.fc2(self.act(self.fc1(x))))


class ImageTransformerBlock(nn.Module):
    """Pre-norm transformer block applied over the flattened spatial dimension."""

    def __init__(self, channels: int, num_heads: int = 4, mlp_ratio: int = 2):
        super().__init__()
        self.norm1 = nn.LayerNorm(channels, eps=1e-6)
        self.attn = SelfAttention(channels, num_heads=num_heads)
        self.norm2 = nn.LayerNorm(channels, eps=1e-6)
        self.mlp = MLP(channels, mlp_ratio=mlp_ratio)

    def forward(self, x):
        b, c, h, w = x.shape
        x = x.reshape(b, c, h * w).permute(0, 2, 1)   # [B, HW, C]
        x = x + self.attn(self.norm1(x))
        x = x + self.mlp(self.norm2(x))
        return x.permute(0, 2, 1).reshape(b, c, h, w)
