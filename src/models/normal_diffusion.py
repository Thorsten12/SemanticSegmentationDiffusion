"""Sparse 1-D normal-residual utilities for P2SDiff V6.

The V6 decoder keeps the 2-D image encoder, but the generative state after the
Fourier proposal is one scalar per contour vertex.  Each scalar moves its vertex
only along the proposal normal.  Conditioning is also sparse: a short feature
profile is sampled along each normal from a few backbone levels.
"""

from __future__ import annotations

from typing import Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F

from .positional import sample_deformable


def contour_normals(points: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """Return unit normals for a closed ordered contour [B,N,2]."""
    prev_p = torch.roll(points, 1, dims=1)
    next_p = torch.roll(points, -1, dims=1)
    tangent = next_p - prev_p
    tangent = tangent / torch.linalg.vector_norm(tangent, dim=-1, keepdim=True).clamp_min(eps)
    return torch.stack([-tangent[..., 1], tangent[..., 0]], dim=-1)


def decode_normal_state(
    proposal: torch.Tensor,
    state: torch.Tensor,
    scale: float,
    normals: torch.Tensor | None = None,
) -> torch.Tensor:
    """Decode [B,N,1] scalar state into 2-D contour points."""
    if state.ndim != 3 or state.shape[-1] != 1:
        raise ValueError(f"normal residual state must be [B,N,1], got {tuple(state.shape)}")
    if normals is None:
        normals = contour_normals(proposal)
    return (proposal + float(scale) * state * normals).clamp(-0.999, 0.999)


def _cross2(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    return a[..., 0] * b[..., 1] - a[..., 1] * b[..., 0]


@torch.no_grad()
def normal_intersection_target(
    proposal: torch.Tensor,
    gt_points: torch.Tensor,
    scale: float,
    max_abs_state: float = 4.0,
    eps: float = 1e-7,
) -> torch.Tensor:
    """Signed target displacement from each proposal vertex to the GT polygon.

    For every proposal vertex we cast the infinite line through its contour
    normal, intersect it with all GT polygon segments, and choose the closest
    valid intersection.  This removes the artificial index-to-index/tangential
    component that an XY residual target contains.

    A nearest-GT projection fallback handles the rare case where numerical
    parallelism produces no valid line/segment intersection.
    """
    if proposal.ndim != 3 or proposal.shape[-1] != 2:
        raise ValueError("proposal must be [B,N,2]")
    if gt_points.ndim != 3 or gt_points.shape[-1] != 2:
        raise ValueError("gt_points must be [B,M,2]")
    if float(scale) <= 0:
        raise ValueError("scale must be positive")

    p = proposal.float()
    gt = gt_points.float()
    normals = contour_normals(p)                                  # [B,N,2]
    a = gt                                                        # [B,M,2]
    b = torch.roll(gt, shifts=-1, dims=1)
    seg = b - a                                                   # [B,M,2]

    # Broadcast proposal lines against all GT segments.
    r = normals[:, :, None, :]                                    # [B,N,1,2]
    s = seg[:, None, :, :]                                        # [B,1,M,2]
    ap = a[:, None, :, :] - p[:, :, None, :]                      # [B,N,M,2]
    denom = _cross2(r, s)                                         # [B,N,M]
    safe = denom.abs() > eps
    denom_safe = torch.where(safe, denom, torch.ones_like(denom))

    # p + d*r = a + u*s
    d = _cross2(ap, s) / denom_safe
    u = _cross2(ap, r) / denom_safe
    valid = safe & (u >= -1e-5) & (u <= 1.0 + 1e-5)

    inf = torch.full_like(d, float("inf"))
    score = torch.where(valid, d.abs(), inf)
    idx = score.argmin(dim=-1)                                    # [B,N]
    best_d = d.gather(-1, idx.unsqueeze(-1)).squeeze(-1)
    has_valid = valid.any(dim=-1)

    # Fallback: nearest GT vertex, projected onto proposal normal.
    dist = torch.cdist(p, gt, p=2)
    near_idx = dist.argmin(dim=-1)
    near = gt.gather(1, near_idx.unsqueeze(-1).expand(-1, -1, 2))
    fallback = ((near - p) * normals).sum(dim=-1)
    best_d = torch.where(has_valid, best_d, fallback)

    state = (best_d / float(scale)).clamp(-float(max_abs_state), float(max_abs_state))
    return state.unsqueeze(-1).to(dtype=proposal.dtype)


class SparseNormalProfileEncoder(nn.Module):
    """Encode short multi-scale feature profiles sampled along contour normals.

    Complexity is O(B*N*K*levels) sampled feature vectors. No dense decoder or
    full-resolution projection is created.
    """

    def __init__(
        self,
        scale_channels: Sequence[int],
        hidden_dim: int = 128,
        levels: int = 2,
        samples: int = 11,
        radius_min: float = 0.015,
        radius_max: float = 0.18,
    ):
        super().__init__()
        self.scale_channels = list(scale_channels)
        self.levels = max(1, min(int(levels), len(self.scale_channels)))
        self.samples = max(3, int(samples))
        if self.samples % 2 == 0:
            self.samples += 1
        self.radius_min = float(radius_min)
        self.radius_max = float(radius_max)
        self.hidden_dim = int(hidden_dim)

        self.proj = nn.ModuleList([
            nn.Sequential(
                nn.Linear(self.scale_channels[i], hidden_dim), nn.GELU(),
                nn.Linear(hidden_dim, hidden_dim),
            )
            for i in range(self.levels)
        ])
        self.alpha_mlp = nn.Sequential(
            nn.Linear(1, hidden_dim), nn.GELU(), nn.Linear(hidden_dim, hidden_dim)
        )
        self.score = nn.Linear(hidden_dim, 1)
        nn.init.zeros_(self.score.weight)
        nn.init.zeros_(self.score.bias)
        self.level_logits = nn.Parameter(torch.zeros(self.levels))
        self.out = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim), nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
        )

    def forward(
        self,
        maps: Sequence[torch.Tensor],
        points: torch.Tensor,
        normals: torch.Tensor,
        t_norm: torch.Tensor,
    ):
        b, n, _ = points.shape
        # High noise -> wider inspection window; low noise -> precise profile.
        radius = self.radius_min + (self.radius_max - self.radius_min) * t_norm
        radius = radius.view(b, 1, 1)
        alpha_unit = torch.linspace(
            -1.0, 1.0, self.samples, device=points.device, dtype=points.dtype,
        )
        offsets = radius.unsqueeze(-1) * alpha_unit.view(1, 1, self.samples, 1)
        locations = points[:, :, None, :] + normals[:, :, None, :] * offsets
        alpha_tokens = self.alpha_mlp(alpha_unit.view(1, 1, self.samples, 1))

        pooled_levels = []
        entropy = []
        for i in range(self.levels):
            sampled = sample_deformable(maps[i], locations)       # [B,N,K,C]
            h = self.proj[i](sampled) + alpha_tokens
            w = torch.softmax(self.score(torch.tanh(h)).squeeze(-1), dim=-1)
            pooled_levels.append((h * w.unsqueeze(-1)).sum(dim=2))
            entropy.append((-(w * (w + 1e-8).log()).sum(dim=-1).mean()).detach())

        level_w = torch.softmax(self.level_logits, dim=0)
        fused = sum(level_w[i] * pooled_levels[i] for i in range(self.levels))
        return self.out(fused), {
            "normal_profile_radius": radius.detach().mean(),
            "normal_profile_entropy": torch.stack(entropy).mean() if entropy else points.new_zeros(()),
            "normal_profile_level0": level_w[0].detach(),
        }
