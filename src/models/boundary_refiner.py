"""Lightweight boundary-specific modules for P2SDiff V3.

V2 localized lesions well but produced visibly over-smoothed contours. V3 keeps
all expensive representation learning inside the existing encoder and adds only
small boundary-specific modules:

1. ``BoundaryFeatureHead`` fuses a few encoder levels into a compact explicit
   boundary feature/logit map.
2. ``NormalBoundaryRefiner`` searches a short 1-D line along each contour normal
   and snaps each vertex to the most plausible lesion boundary.

The refiner is sparse.  For N vertices and K candidates it reads O(N*K) feature
vectors.  It does not introduce a second dense segmentation decoder.
"""

from __future__ import annotations

from typing import Dict, Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F

from .positional import sample_deformable


def _gn_groups(channels: int, max_groups: int = 8) -> int:
    for g in range(min(max_groups, channels), 0, -1):
        if channels % g == 0:
            return g
    return 1


def contour_normals(points: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """Unit normals for a closed ordered polygon, shape [B,N,2]."""
    prev_p = torch.roll(points, shifts=1, dims=1)
    next_p = torch.roll(points, shifts=-1, dims=1)
    tangent = next_p - prev_p
    tangent = tangent / torch.norm(tangent, dim=-1, keepdim=True).clamp(min=eps)
    return torch.stack([-tangent[..., 1], tangent[..., 0]], dim=-1)


def perturb_along_normals(
    points: torch.Tensor,
    max_offset: float = 0.10,
    smooth_passes: int = 2,
) -> torch.Tensor:
    """Create realistic coarse contours for teacher-training the snap refiner."""
    normals = contour_normals(points)
    displacement = torch.empty(
        points.shape[0], points.shape[1], 1,
        device=points.device, dtype=points.dtype,
    ).uniform_(-float(max_offset), float(max_offset))
    for _ in range(max(int(smooth_passes), 0)):
        displacement = (
            torch.roll(displacement, 1, 1) + 2.0 * displacement
            + torch.roll(displacement, -1, 1)
        ) / 4.0
    return (points + displacement * normals).clamp(-0.995, 0.995)


class BoundaryFeatureHead(nn.Module):
    """Fuse finest encoder levels into a compact full-resolution boundary map.

    Parameter count is tiny compared with the encoder: projections are 1x1 and
    the only spatial convolution is depth-wise.  The feature map is reused by
    every DDIM step and every snap iteration.
    """

    def __init__(
        self,
        scale_channels: Sequence[int],
        edge_dim: int = 32,
        levels: int = 3,
    ):
        super().__init__()
        self.scale_channels = list(scale_channels)
        self.levels = max(1, min(int(levels), len(self.scale_channels)))
        self.edge_dim = int(edge_dim)

        self.proj = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(c, self.edge_dim, kernel_size=1, bias=False),
                nn.GroupNorm(_gn_groups(self.edge_dim), self.edge_dim),
                nn.SiLU(),
            )
            for c in self.scale_channels[: self.levels]
        ])
        in_fuse = self.edge_dim * self.levels
        self.fuse = nn.Sequential(
            nn.Conv2d(in_fuse, self.edge_dim, kernel_size=1, bias=False),
            nn.GroupNorm(_gn_groups(self.edge_dim), self.edge_dim),
            nn.SiLU(),
            nn.Conv2d(
                self.edge_dim, self.edge_dim, kernel_size=3, padding=1,
                groups=self.edge_dim, bias=False,
            ),
            nn.GroupNorm(_gn_groups(self.edge_dim), self.edge_dim),
            nn.SiLU(),
            nn.Conv2d(self.edge_dim, self.edge_dim, kernel_size=1),
            nn.SiLU(),
        )
        self.logit = nn.Conv2d(self.edge_dim, 1, kernel_size=1)
        nn.init.constant_(self.logit.bias, -2.0)

    def forward(self, maps):
        target_hw = maps[0].shape[-2:]
        pieces = []
        for i in range(self.levels):
            x = self.proj[i](maps[i])
            if x.shape[-2:] != target_hw:
                x = F.interpolate(x, size=target_hw, mode="bilinear", align_corners=True)
            pieces.append(x)
        feat = self.fuse(torch.cat(pieces, dim=1))
        logits = self.logit(feat)
        return feat, logits


class NormalBoundaryRefiner(nn.Module):
    """Iteratively snap contour vertices to the lesion boundary along normals.

    Besides the candidate feature itself, the scorer reads a *normal profile*:
    features a small distance on the two sides of every candidate.  This makes
    it respond to a true inside/outside transition rather than merely to a dark
    hair or an internal texture edge.  The extra computation is still sparse.
    """

    def __init__(
        self,
        edge_dim: int = 32,
        n_candidates: int = 15,
        iterations: int = 3,
        radius_max: float = 0.16,
        radius_min: float = 0.015,
        temperature: float = 0.16,
        profile_radius: float = 0.018,
    ):
        super().__init__()
        n_candidates = max(3, int(n_candidates))
        if n_candidates % 2 == 0:
            n_candidates += 1
        self.n_candidates = n_candidates
        self.iterations = max(1, int(iterations))
        self.radius_max = float(radius_max)
        self.radius_min = float(radius_min)
        self.temperature = max(float(temperature), 1e-3)
        self.profile_radius = max(float(profile_radius), 1e-4)

        # center feature + |feature(+n)-feature(-n)| + raw edge logit +
        # edge-profile contrast + signed/absolute candidate displacement.
        score_in = 2 * edge_dim + 5
        inner = max(edge_dim, 24)
        self.score_mlp = nn.Sequential(
            nn.Linear(score_in, inner), nn.GELU(),
            nn.Linear(inner, max(inner // 2, 16)), nn.GELU(),
            nn.Linear(max(inner // 2, 16), 1),
        )
        nn.init.zeros_(self.score_mlp[-1].weight)
        nn.init.zeros_(self.score_mlp[-1].bias)
        self.edge_logit_scale = nn.Parameter(torch.tensor(1.0))
        self.profile_scale = nn.Parameter(torch.tensor(0.5))
        self.last_stats: Dict[str, torch.Tensor] = {}

    def _radius(self, iteration: int) -> float:
        if self.iterations <= 1:
            return self.radius_min
        frac = iteration / (self.iterations - 1)
        return self.radius_max * ((self.radius_min / self.radius_max) ** frac)

    def forward(self, points, edge_feat, edge_logits, hard: bool = False):
        p = points
        delta_stats, entropy_stats, profile_stats = [], [], []

        for it in range(self.iterations):
            normal = contour_normals(p)
            radius = self._radius(it)
            offsets = torch.linspace(
                -radius, radius, self.n_candidates,
                device=p.device, dtype=p.dtype,
            )
            candidates = p[:, :, None, :] + normal[:, :, None, :] * offsets[None, None, :, None]

            valid = (candidates.abs() <= 1.0).all(dim=-1)
            sample_locs = candidates.clamp(-1.0, 1.0)
            feat_c = sample_deformable(edge_feat, sample_locs)          # [B,N,K,C]
            logit_c = sample_deformable(edge_logits, sample_locs)[..., 0:1]

            # Sparse two-sided profile around each candidate.  Keep the probe
            # small enough to remain a local boundary cue at 224x224.
            probe = min(self.profile_radius, max(radius * 0.50, self.profile_radius * 0.5))
            n4 = normal[:, :, None, :]
            loc_minus = (candidates - probe * n4).clamp(-1.0, 1.0)
            loc_plus = (candidates + probe * n4).clamp(-1.0, 1.0)
            feat_minus = sample_deformable(edge_feat, loc_minus)
            feat_plus = sample_deformable(edge_feat, loc_plus)
            feat_contrast = (feat_plus - feat_minus).abs()
            logit_minus = sample_deformable(edge_logits, loc_minus)[..., 0:1]
            logit_plus = sample_deformable(edge_logits, loc_plus)[..., 0:1]
            profile_contrast = (logit_plus - logit_minus).abs()

            rel = (offsets / max(radius, 1e-8)).view(1, 1, -1, 1)
            rel = rel.expand(p.shape[0], p.shape[1], -1, -1)
            scorer_in = torch.cat([
                feat_c,
                feat_contrast,
                logit_c,
                profile_contrast,
                rel,
                rel.abs(),
                torch.full_like(rel, float(radius / max(self.radius_max, 1e-8))),
            ], dim=-1)
            scores = self.score_mlp(scorer_in).squeeze(-1)
            scores = scores + self.edge_logit_scale * logit_c.squeeze(-1)
            scores = scores + self.profile_scale * profile_contrast.squeeze(-1)
            scores = scores.masked_fill(~valid, -1e4)

            if hard:
                idx = scores.argmax(dim=-1)
                chosen = offsets[idx]
                weights = F.one_hot(idx, num_classes=self.n_candidates).to(p.dtype)
            else:
                weights = torch.softmax(scores / self.temperature, dim=-1)
                chosen = (weights * offsets.view(1, 1, -1)).sum(dim=-1)

            p = (p + normal * chosen.unsqueeze(-1)).clamp(-0.999, 0.999)
            delta_stats.append(chosen.detach().abs().mean())
            entropy_stats.append(
                (-(weights.detach() * (weights.detach() + 1e-8).log()).sum(dim=-1).mean())
            )
            profile_stats.append(profile_contrast.detach().mean())

        self.last_stats = {
            "snap_delta_abs": torch.stack(delta_stats).mean(),
            "snap_entropy": torch.stack(entropy_stats).mean(),
            "snap_profile_contrast": torch.stack(profile_stats).mean(),
            "edge_logit_scale": self.edge_logit_scale.detach(),
            "profile_scale": self.profile_scale.detach(),
        }
        return p


def make_boundary_target(mask: torch.Tensor, thickness: int = 1) -> torch.Tensor:
    """Binary morphological boundary band; outside image is background."""
    mask = (mask.float() > 0.5).float()
    r = max(int(thickness), 1)
    k = 2 * r + 1
    dil = F.max_pool2d(F.pad(mask, (r, r, r, r), value=0.0), k, stride=1)
    eroded = -F.max_pool2d(F.pad(-mask, (r, r, r, r), value=0.0), k, stride=1)
    return (dil - eroded).clamp(0.0, 1.0)


def boundary_head_loss(
    edge_logits: torch.Tensor,
    masks: torch.Tensor,
    thickness: int = 1,
):
    """Class-balanced BCE + soft Dice for explicit boundary supervision."""
    masks_r = F.interpolate(masks.float(), size=edge_logits.shape[-2:], mode="nearest")
    target = make_boundary_target(masks_r, thickness=thickness)

    with torch.no_grad():
        pos = target.sum().clamp(min=1.0)
        neg = (target.numel() - pos).clamp(min=1.0)
        pos_weight = (neg / pos).clamp(1.0, 20.0)
    bce = F.binary_cross_entropy_with_logits(edge_logits, target, pos_weight=pos_weight)

    prob = torch.sigmoid(edge_logits)
    inter = (prob * target).sum(dim=(1, 2, 3))
    denom = prob.sum(dim=(1, 2, 3)) + target.sum(dim=(1, 2, 3))
    dice = (2.0 * inter + 1e-6) / (denom + 1e-6)
    dice_loss = (1.0 - dice).mean()
    return bce + dice_loss, {
        "boundary_bce": bce.detach(),
        "boundary_dice_loss": dice_loss.detach(),
    }
