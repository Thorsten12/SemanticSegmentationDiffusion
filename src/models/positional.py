"""Shared spatial encodings and point samplers.

Image coordinates always use ``(x, y)`` in ``[-1, 1]`` with
``align_corners=True``.  Diffusion itself is allowed to live in an unbounded
latent space; the denoiser converts its latent state to bounded image
coordinates before anything in this module is called.
"""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class SharedCoordinatePE(nn.Module):
    """Fourier encoding shared by 2-D feature grids and contour queries."""

    def __init__(self, num_bands=6, max_freq=32.0, include_input=True):
        super().__init__()
        self.num_bands = int(num_bands)
        self.max_freq = float(max_freq)
        self.include_input = bool(include_input)
        if self.num_bands <= 0:
            freqs = torch.zeros(0)
        else:
            exps = torch.linspace(0.0, math.log2(self.max_freq), self.num_bands)
            freqs = (2.0 ** exps) * math.pi
        self.register_buffer("freqs", freqs)
        self.out_dim = 2 * (2 * self.num_bands) + (2 if include_input else 0)

    def encode_points(self, xy, height=None):
        """Encode ``xy`` [...,2] in the image coordinate frame."""
        if self.freqs.numel() == 0:
            return xy
        proj = xy[..., None] * self.freqs
        if height is not None:
            mask = self._band_mask(height, xy.device, proj.dtype)
            proj = proj * mask
        emb = torch.cat([proj.sin(), proj.cos()], dim=-1).flatten(-2)
        if self.include_input:
            emb = torch.cat([xy, emb], dim=-1)
        return emb

    def encode_grid(self, height, width, device, dtype, nyquist=True):
        ys = torch.linspace(-1.0, 1.0, height, device=device, dtype=dtype)
        xs = torch.linspace(-1.0, 1.0, width, device=device, dtype=dtype)
        gy, gx = torch.meshgrid(ys, xs, indexing="ij")
        coords = torch.stack([gx, gy], dim=-1)
        pe = self.encode_points(coords, height=height if nyquist else None)
        return pe.permute(2, 0, 1).unsqueeze(0)

    def _band_mask(self, height, device, dtype):
        nyquist = math.pi * max(int(height), 2) / 2.0
        return (self.freqs <= nyquist).to(device=device, dtype=dtype)


def sample_aligned(feat, points, padding_mode="border"):
    """Bilinear sample ``feat`` [B,C,H,W] at [B,N,2] image coordinates.

    ``points`` should already be bounded.  A final clamp is intentionally kept
    as a numerical guard for deformable offsets that may step just outside the
    field of view; unlike the old decoder it is *not* used to turn Gaussian
    diffusion coordinates into image coordinates.
    """
    grid = points.clamp(-1.0, 1.0).unsqueeze(1)
    sampled = F.grid_sample(
        feat, grid, mode="bilinear", padding_mode=padding_mode,
        align_corners=True,
    )
    return sampled.squeeze(2).transpose(1, 2)


def sample_deformable(feat, locations, padding_mode="border"):
    """Sample K locations per contour vertex.

    Parameters
    ----------
    feat: [B,C,H,W]
    locations: [B,N,K,2] in image coordinates

    Returns
    -------
    [B,N,K,C]
    """
    b, n, k, _ = locations.shape
    grid = locations.clamp(-1.0, 1.0).reshape(b, n * k, 1, 2)
    sampled = F.grid_sample(
        feat, grid, mode="bilinear", padding_mode=padding_mode,
        align_corners=True,
    )
    sampled = sampled.squeeze(-1).transpose(1, 2)
    return sampled.reshape(b, n, k, feat.shape[1])
