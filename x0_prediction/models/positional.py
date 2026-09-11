"""Fourier positional encodings used to entangle *location* with *content*.

Frozen CNN backbones (ConvNeXt, ResNet, ...) are largely translation-equivariant:
a feature describes appearance but not absolute position. The denoiser must move
each point to a precise location, so we inject explicit positional information:

* `FourierFeatures`  - NeRF-style sin/cos embedding of 2D point coordinates. Used
  both to embed the points the denoiser is moving, and to tag each *sampled*
  guidance feature with the exact coordinate it was read from (alignment by
  construction, since the coordinate is known).
* `PositionalGrid2D` - a 2D Fourier coordinate grid projected into the channel
  space of a feature map and added to it, so the guidance map itself is
  position-aware.
"""

import math

import torch
import torch.nn as nn


class FourierFeatures(nn.Module):
    """Map coordinates `[..., in_dim]` -> `[..., out_dim]` sin/cos features."""

    def __init__(self, in_dim=2, num_bands=6, max_freq=32.0, include_input=True):
        super().__init__()
        # Log-spaced frequencies in [1, max_freq] * pi. `max_freq` is bounded so
        # the highest band stays below the spatial Nyquist limit of the data; with
        # unbounded 2**arange the top bands become pure aliasing noise that the
        # model overfits to (causes val collapse).
        exps = torch.linspace(0.0, math.log2(max_freq), num_bands)
        freqs = (2.0 ** exps) * math.pi
        self.register_buffer("freqs", freqs)
        self.include_input = include_input
        self.in_dim = in_dim
        self.out_dim = in_dim * (2 * num_bands) + (in_dim if include_input else 0)

    def forward(self, x):
        if self.freqs.numel() == 0:           # num_bands=0 -> PE disabled, raw coords
            return x
        proj = x[..., None] * self.freqs                       # [..., in_dim, B]
        emb = torch.cat([proj.sin(), proj.cos()], dim=-1)      # [..., in_dim, 2B]
        emb = emb.flatten(-2)                                  # [..., in_dim*2B]
        if self.include_input:
            emb = torch.cat([x, emb], dim=-1)
        return emb


class PositionalGrid2D(nn.Module):
    """Add a projected 2D Fourier coordinate grid to a feature map `[B, C, H, W]`."""

    def __init__(self, out_channels, num_bands=5, max_freq=16.0):
        super().__init__()
        self.num_bands = num_bands
        # Bounded log-spaced frequencies (see FourierFeatures). The guidance map is
        # low-resolution (stride 4), so cap the top band well below its Nyquist.
        exps = torch.linspace(0.0, math.log2(max_freq), num_bands)
        freqs = (2.0 ** exps) * math.pi
        self.register_buffer("freqs", freqs)
        self.proj = nn.Conv2d(4 * num_bands, out_channels, 1)
        self._cache = {}                                       # (H, W, device) -> grid

    def _grid(self, h, w, device, dtype):
        key = (h, w, device)
        pe = self._cache.get(key)
        if pe is None:
            ys = torch.linspace(-1, 1, h, device=device)
            xs = torch.linspace(-1, 1, w, device=device)
            gy, gx = torch.meshgrid(ys, xs, indexing="ij")
            coords = torch.stack([gx, gy], dim=0)             # [2, H, W]
            proj = coords[..., None] * self.freqs             # [2, H, W, B]
            emb = torch.cat([proj.sin(), proj.cos()], dim=-1) # [2, H, W, 2B]
            pe = emb.permute(0, 3, 1, 2).reshape(4 * self.num_bands, h, w)
            pe = pe.unsqueeze(0)                              # [1, 4B, H, W]
            self._cache[key] = pe
        return pe.to(dtype)

    def forward(self, x):
        pe = self._grid(x.shape[-2], x.shape[-1], x.device, x.dtype)
        return x + self.proj(pe)
