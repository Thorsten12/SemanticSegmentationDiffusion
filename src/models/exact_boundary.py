"""Sparse confidence-gated exact boundary correction for P2SDiff V5.2.

This module is intentionally *not* a dense pixel decoder.  It samples a short
normal profile from the existing encoder feature pyramid at N contour vertices,
communicates along the closed contour with a tiny circular/relative-attention
block, and predicts only two scalars per vertex:

* signed normal offset, and
* confidence that an exact local correction is appropriate.

The core design rule is asymmetric:

* clear / nearby boundary -> allow precise high-frequency correction;
* ambiguous / distant boundary -> suppress the local correction and leave the
  global diffusion model in control.

There is no candidate-distribution entropy target and no post-snapper.
"""

from __future__ import annotations

import math
from typing import Dict, Sequence, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from .positional import sample_deformable


def contour_frame(points: torch.Tensor, eps: float = 1e-6) -> Tuple[torch.Tensor, torch.Tensor]:
    """Return unit tangent and normal for a closed ordered contour [B,N,2]."""
    prev_p = torch.roll(points, shifts=1, dims=1)
    next_p = torch.roll(points, shifts=-1, dims=1)
    tangent = next_p - prev_p
    tangent = tangent / torch.norm(tangent, dim=-1, keepdim=True).clamp(min=eps)
    normal = torch.stack([-tangent[..., 1], tangent[..., 0]], dim=-1)
    return tangent, normal


def curvature_scalar(points: torch.Tensor) -> torch.Tensor:
    """Simple local second-difference magnitude [B,N,1]."""
    prev_p = torch.roll(points, shifts=1, dims=1)
    next_p = torch.roll(points, shifts=-1, dims=1)
    return torch.norm(prev_p - 2.0 * points + next_p, dim=-1, keepdim=True)


class CyclicRelativeBlock(nn.Module):
    """Tiny ring-aware communication block.

    The attention bias is fixed and depends only on cyclic distance, so there is
    no learnable absolute vertex ID.  A small circular convolution provides fast
    neighbour exchange; attention still remains global over all N vertices.
    """

    def __init__(self, dim: int, num_heads: int, n_points: int, bias_strength: float = 0.12):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.conv = nn.Conv1d(dim, dim, 3, padding=1, padding_mode="circular")
        self.norm2 = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(dim, num_heads, batch_first=True, dropout=0.0)
        self.norm3 = nn.LayerNorm(dim)
        self.ff = nn.Sequential(
            nn.Linear(dim, dim * 2), nn.GELU(),
            nn.Linear(dim * 2, dim),
        )

        idx = torch.arange(n_points, dtype=torch.float32)
        d = (idx[:, None] - idx[None, :]).abs()
        d = torch.minimum(d, float(n_points) - d)
        d = d / max(float(n_points) / 2.0, 1.0)
        # Mild prior only: close ring neighbours get a small positive bias, but
        # global attention is still permitted.  This is topology awareness, not
        # a smoothing penalty.
        bias = float(bias_strength) * (1.0 - 2.0 * d)
        self.register_buffer("relative_bias", bias, persistent=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.norm1(x)
        y = self.conv(y.transpose(1, 2)).transpose(1, 2)
        x = x + F.gelu(y)
        y = self.norm2(x)
        y, _ = self.attn(y, y, y, attn_mask=self.relative_bias.to(dtype=y.dtype), need_weights=False)
        x = x + y
        x = x + self.ff(self.norm3(x))
        return x


class SparseExactBoundaryCorrector(nn.Module):
    """Predict exact local normal correction + confidence from sparse profiles."""

    def __init__(
        self,
        scale_channels: Sequence[int],
        n_points: int = 100,
        levels: int = 2,
        n_samples: int = 11,
        radius: float = 0.10,
        profile_dim: int = 20,
        hidden_dim: int = 64,
        ring_bands: int = 4,
        num_heads: int = 4,
        relative_bias_strength: float = 0.12,
        confidence_power: float = 2.0,
        use_rgb: bool = True,
    ):
        super().__init__()
        self.scale_channels = list(scale_channels)
        self.levels = max(1, min(int(levels), len(self.scale_channels)))
        self.n_points = int(n_points)
        self.n_samples = max(5, int(n_samples))
        if self.n_samples % 2 == 0:
            self.n_samples += 1
        self.radius = float(radius)
        self.profile_dim = int(profile_dim)
        self.hidden_dim = int(hidden_dim)
        self.confidence_power = max(float(confidence_power), 1.0)
        self.use_rgb = bool(use_rgb)

        self.sample_proj = nn.ModuleList([
            nn.Sequential(
                nn.Linear(c, self.profile_dim), nn.GELU(),
                nn.Linear(self.profile_dim, self.profile_dim),
            )
            for c in self.scale_channels[: self.levels]
        ])
        self.rgb_proj = nn.Sequential(
            nn.Linear(3, self.profile_dim), nn.GELU(),
            nn.Linear(self.profile_dim, self.profile_dim),
        ) if self.use_rgb else None
        # Sparse multi-scale profile -> one contour token per point.
        self.profile_mlp = nn.Sequential(
            nn.Linear(self.n_samples * self.profile_dim, self.hidden_dim), nn.GELU(),
            nn.Linear(self.hidden_dim, self.hidden_dim),
        )

        phi = 2.0 * math.pi * torch.arange(self.n_points, dtype=torch.float32) / self.n_points
        ring = []
        for k in range(1, max(int(ring_bands), 1) + 1):
            ring.extend([(k * phi).sin(), (k * phi).cos()])
        ring = torch.stack(ring, dim=-1).unsqueeze(0)
        self.register_buffer("ring_features", ring, persistent=True)
        self.ring_proj = nn.Sequential(
            nn.Linear(ring.shape[-1], self.hidden_dim), nn.GELU(),
            nn.Linear(self.hidden_dim, self.hidden_dim),
        )

        # point xy + tangent xy + normal xy + curvature + normalized radius cue
        self.geom_mlp = nn.Sequential(
            nn.Linear(2 + 2 + 2 + 1 + 1, self.hidden_dim), nn.GELU(),
            nn.Linear(self.hidden_dim, self.hidden_dim),
        )
        self.ring_block = CyclicRelativeBlock(
            self.hidden_dim, num_heads=num_heads, n_points=self.n_points,
            bias_strength=relative_bias_strength,
        )
        self.offset_head = nn.Sequential(
            nn.LayerNorm(self.hidden_dim),
            nn.Linear(self.hidden_dim, self.hidden_dim // 2), nn.GELU(),
            nn.Linear(self.hidden_dim // 2, 1),
        )
        self.conf_head = nn.Sequential(
            nn.LayerNorm(self.hidden_dim),
            nn.Linear(self.hidden_dim, self.hidden_dim // 2), nn.GELU(),
            nn.Linear(self.hidden_dim // 2, 1),
        )
        # Start as identity: exact branch initially applies no correction.
        nn.init.zeros_(self.offset_head[-1].weight)
        nn.init.zeros_(self.offset_head[-1].bias)
        nn.init.zeros_(self.conf_head[-1].weight)
        nn.init.zeros_(self.conf_head[-1].bias)

        offsets = torch.linspace(-self.radius, self.radius, self.n_samples)
        self.register_buffer("profile_offsets", offsets, persistent=True)
        self.last_stats: Dict[str, torch.Tensor] = {}

    def forward(self, points: torch.Tensor, maps: Sequence[torch.Tensor], image: torch.Tensor | None = None):
        if points.shape[1] != self.n_points:
            raise ValueError(f"expected {self.n_points} points, got {points.shape[1]}")
        # Deliberately detach the geometric query frame.  The branch learns from
        # encoder evidence and does not destabilize the successful V5 coarse path
        # through grid-sampling coordinate gradients.
        p = points.detach()
        tangent, normal = contour_frame(p)
        offsets = self.profile_offsets.to(device=p.device, dtype=p.dtype)
        loc = p[:, :, None, :] + normal[:, :, None, :] * offsets.view(1, 1, -1, 1)
        loc = loc.clamp(-1.0, 1.0)

        profile = None
        for level in range(self.levels):
            raw = sample_deformable(maps[level], loc)                  # [B,N,K,C]
            emb = self.sample_proj[level](raw)                          # [B,N,K,D]
            profile = emb if profile is None else profile + emb
        profile = profile / float(self.levels)
        if self.rgb_proj is not None and image is not None:
            # Raw RGB is sampled sparsely at exactly the same normal locations;
            # this restores pixel-level edge evidence without any dense RGB decoder.
            rgb = sample_deformable(image, loc)
            profile = profile + self.rgb_proj(rgb)
        token = self.profile_mlp(profile.flatten(-2))

        center = p.mean(dim=1, keepdim=True)
        radial = torch.norm(p - center, dim=-1, keepdim=True)
        scale = radial.mean(dim=1, keepdim=True).clamp(min=1e-4)
        radial_norm = radial / scale
        curv = curvature_scalar(p)
        geom = torch.cat([p, tangent, normal, curv, radial_norm], dim=-1)
        token = token + self.geom_mlp(geom)
        token = token + self.ring_proj(self.ring_features.to(dtype=token.dtype)).expand(p.shape[0], -1, -1)
        token = self.ring_block(token)

        offset = self.radius * torch.tanh(self.offset_head(token).squeeze(-1))
        conf_logit = self.conf_head(token).squeeze(-1)
        conf_prob = torch.sigmoid(conf_logit)
        gate = conf_prob.pow(self.confidence_power)
        correction = gate * offset

        with torch.no_grad():
            self.last_stats = {
                "exact_offset_abs": offset.detach().abs().mean(),
                "exact_conf_mean": conf_prob.detach().mean(),
                "exact_gate_mean": gate.detach().mean(),
                "exact_correction_abs": correction.detach().abs().mean(),
            }
        return correction, offset, conf_logit, normal


def nearest_boundary_targets(
    base_points: torch.Tensor,
    target_points: torch.Tensor,
    normal: torch.Tensor,
    confidence_radius: float = 0.060,
    tangent_tolerance: float = 0.040,
    max_offset: float = 0.10,
):
    """Build simple hard confidence + signed offset targets.

    A point is "confident" only if the nearest GT boundary vertex is nearby and
    requires little tangential motion.  This avoids forcing the local exact branch
    to solve large/global correspondence errors.
    """
    with torch.no_grad():
        d = torch.cdist(base_points.float(), target_points.float(), p=2)
        min_dist, idx = d.min(dim=2)
        b = torch.arange(base_points.shape[0], device=base_points.device)[:, None]
        nearest = target_points[b, idx]
        delta = nearest - base_points
        tangent = torch.stack([normal[..., 1], -normal[..., 0]], dim=-1)
        signed_n = (delta * normal).sum(dim=-1)
        signed_t = (delta * tangent).sum(dim=-1).abs()
        conf = (
            (min_dist <= float(confidence_radius))
            & (signed_t <= float(tangent_tolerance))
            & (signed_n.abs() <= float(max_offset))
        ).to(base_points.dtype)
        offset = signed_n.clamp(-float(max_offset), float(max_offset))
    return offset, conf, min_dist, signed_t


def exact_boundary_supervision_loss(
    offset_pred: torch.Tensor,
    conf_logit: torch.Tensor,
    base_points: torch.Tensor,
    normal: torch.Tensor,
    target_points: torch.Tensor,
    confidence_radius: float,
    tangent_tolerance: float,
    max_offset: float,
):
    target_offset, target_conf, min_dist, tangent_dist = nearest_boundary_targets(
        base_points.detach(), target_points.detach(), normal.detach(),
        confidence_radius=confidence_radius,
        tangent_tolerance=tangent_tolerance,
        max_offset=max_offset,
    )
    # Balanced confidence BCE so the safe negative class cannot dominate.
    pos = target_conf.sum().clamp(min=1.0)
    neg = (target_conf.numel() - target_conf.sum()).clamp(min=1.0)
    pos_weight = (neg / pos).clamp(1.0, 12.0)
    loss_conf = F.binary_cross_entropy_with_logits(conf_logit, target_conf, pos_weight=pos_weight)

    w = target_conf
    if float(w.sum().item()) > 0:
        loss_offset = F.smooth_l1_loss(
            offset_pred / max(float(max_offset), 1e-6),
            target_offset / max(float(max_offset), 1e-6),
            reduction="none",
        )
        loss_offset = (loss_offset * w).sum() / w.sum().clamp(min=1.0)
    else:
        loss_offset = offset_pred.new_zeros(())

    parts = {
        "exact_loss_offset": loss_offset.detach(),
        "exact_loss_conf": loss_conf.detach(),
        "exact_target_conf_rate": target_conf.detach().mean(),
        "exact_target_dist": min_dist.detach().mean(),
        "exact_target_tangent": tangent_dist.detach().mean(),
    }
    return loss_offset, loss_conf, parts
