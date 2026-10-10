"""Confidence-gated normal-offset boundary snapper.

A lightweight post-processing module applied once after DDIM sampling. The
diffusion model recovers the global contour shape well, but exact boundary
locations may drift locally. The snapper refines these local details and is
deliberately decoupled from the diffusion forward pass.

Method
------
1. Profile sampling: for each of the N contour points, a 1D profile is sampled
   along the local normal from the raw backbone feature maps (multiple encoder
   scales, optionally raw RGB).
2. Ring communication: a small conv/attention block lets the N points exchange
   information along the closed contour (cyclic neighborhood, no learned
   per-index embeddings).
3. Prediction heads: a signed normal offset and a confidence logit. The offset
   is applied with weight sigmoid(conf)^p, so low-confidence points are left
   (almost) untouched. This prevents the snapper from attempting to correct
   large, global errors it is not designed for.

Training
--------
The snapper is trained with a separate teacher loss on artificially perturbed
ground-truth contours rather than on on-policy DDIM intermediates. Early
rollouts are often far from the ground truth, which would drive the confidence
head to collapse to "always uncertain". Perturbed ground truth instead yields
realistic near-boundary inputs that require only fine adjustments, independent
of the current quality of the diffusion model.

The loss combines offset regression (`loss_offset`), confidence classification
(`loss_conf`) and an optional BoundaryIoU term (`loss_biou`). The latter
rasterizes the contour that `forward()` would output (same confidence gate,
via `_apply_gate`) and compares it to the ground-truth mask, providing a
mask-level geometric signal.

Public API
----------
    snapper = BoundarySnapper(scale_channels=encoder.feature_channels, n_points=100)
    corrected = snapper(points, maps, image=None)                # inference
    loss, parts = snapper.teacher_loss(gt_points, maps, image)   # training
"""

from __future__ import annotations

from typing import Dict, Optional, Sequence, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..utils.rasterize import soft_boundary_iou_loss


# --------------------------------------------------------------------------
# Contour geometry utilities
# --------------------------------------------------------------------------

def contour_frame(points: torch.Tensor, eps: float = 1e-6) -> Tuple[torch.Tensor, torch.Tensor]:
    """Tangent and normal of a closed, ordered contour.

    Args:
        points: Contour vertices, shape [B, N, 2].
    Returns:
        (tangent, normal), each [B, N, 2] and unit length. Tangents use central
        differences over the cyclic neighbors.
    """
    prev_p = torch.roll(points, shifts=1, dims=1)
    next_p = torch.roll(points, shifts=-1, dims=1)
    tangent = next_p - prev_p
    tangent = tangent / torch.norm(tangent, dim=-1, keepdim=True).clamp(min=eps)
    normal = torch.stack([-tangent[..., 1], tangent[..., 0]], dim=-1)
    return tangent, normal


def curvature_scalar(points: torch.Tensor) -> torch.Tensor:
    """Curvature proxy via the norm of the second difference, shape [B, N, 1]."""
    prev_p = torch.roll(points, shifts=1, dims=1)
    next_p = torch.roll(points, shifts=-1, dims=1)
    return torch.norm(prev_p - 2.0 * points + next_p, dim=-1, keepdim=True)


def perturb_along_normals(
    points: torch.Tensor,
    max_offset: float = 0.08,
    smooth_passes: int = 2,
) -> torch.Tensor:
    """Generate slightly erroneous contours from ground truth (teacher inputs).

    A uniform random offset is drawn per vertex and applied along the normal.
    The offsets are smoothed over the ring neighbors ([1, 2, 1] / 4 kernel,
    `smooth_passes` times) so that the perturbation resembles a plausible,
    spatially continuous inaccuracy rather than i.i.d. vertex noise.
    """
    _, normal = contour_frame(points)
    displacement = torch.empty(
        points.shape[0], points.shape[1], 1,
        device=points.device, dtype=points.dtype,
    ).uniform_(-float(max_offset), float(max_offset))
    for _ in range(max(int(smooth_passes), 0)):
        displacement = (
            torch.roll(displacement, 1, 1) + 2.0 * displacement
            + torch.roll(displacement, -1, 1)
        ) / 4.0
    return (points + displacement * normal).clamp(-1.0, 1.0)


def sample_at_points(feat_map: torch.Tensor, locations: torch.Tensor) -> torch.Tensor:
    """Bilinearly sample a feature map at arbitrary locations.

    Args:
        feat_map: [B, C, H, W].
        locations: [B, N, K, 2] in normalized coordinates [-1, 1].
    Returns:
        Samples of shape [B, N, K, C].
    """
    # grid_sample expects [B, H_out, W_out, 2]; N*K is folded into H_out.
    b, n, k, _ = locations.shape
    grid = locations.view(b, n * k, 1, 2)
    sampled = F.grid_sample(feat_map, grid, mode="bilinear",
                             padding_mode="border", align_corners=True)  # [B, C, N*K, 1]
    sampled = sampled.squeeze(-1).transpose(1, 2)                       # [B, N*K, C]
    return sampled.view(b, n, k, -1)


# --------------------------------------------------------------------------
# Ring communication block
# --------------------------------------------------------------------------

class _CyclicRelativeBlock(nn.Module):
    """Ring-aware communication block over the N contour points.

    Combines a circular 1D convolution, multi-head self-attention and a
    feed-forward layer (pre-norm residual structure). Attention uses a fixed,
    non-learned bias that depends on cyclic distance. This acts as a mild
    locality prior (nearby neighbors slightly favored) without hard masking,
    and avoids per-index embeddings, keeping the module independent of a
    particular vertex numbering.
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

        # Attention bias: linear in normalized cyclic distance, in
        # [-bias_strength, +bias_strength] (maximal for self, minimal for the
        # antipodal vertex).
        idx = torch.arange(n_points, dtype=torch.float32)
        d = (idx[:, None] - idx[None, :]).abs()
        d = torch.minimum(d, float(n_points) - d)
        d = d / max(float(n_points) / 2.0, 1.0)
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


# --------------------------------------------------------------------------
# Boundary snapper
# --------------------------------------------------------------------------

class BoundarySnapper(nn.Module):
    """Confidence-gated normal-offset snapper (post-DDIM refinement).

    Args:
        scale_channels: Channel counts of the raw backbone maps, identical to
            `encoder.feature_channels` of the denoiser.
        n_points: Number of contour points (must match the diffusion model).
        levels: Number of (finest) feature levels to sample.
        n_samples: Number of profile samples along the normal. Forced to be
            odd so that one sample lies exactly on the vertex.
        radius: Maximum search radius along the normal, in normalized [-1, 1]
            image coordinates.
        profile_dim: Per-sample embedding dimension.
        hidden_dim: Token dimension of the ring block and heads.
        ring_bands: Number of sin/cos harmonics of the fixed ring-position feature.
        num_heads: Attention heads in the ring block.
        relative_bias_strength: Amplitude of the cyclic attention bias.
        confidence_power: Exponent p of the confidence gate sigmoid(conf)^p.
        use_rgb: Additionally sample raw RGB at the profile locations.
    """

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

        # Per-level projection of sampled features to a common profile dimension.
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

        # Aggregates the K-sample profile into one token per vertex.
        self.profile_mlp = nn.Sequential(
            nn.Linear(self.n_samples * self.profile_dim, self.hidden_dim), nn.GELU(),
            nn.Linear(self.hidden_dim, self.hidden_dim),
        )

        # Fixed ring-position feature (sin/cos over several harmonics);
        # no learned per-index embedding.
        import math
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

        # Per-vertex geometry: xy, tangent, normal, curvature, normalized radial distance.
        self.geom_mlp = nn.Sequential(
            nn.Linear(2 + 2 + 2 + 1 + 1, self.hidden_dim), nn.GELU(),
            nn.Linear(self.hidden_dim, self.hidden_dim),
        )
        self.ring_block = _CyclicRelativeBlock(
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
        # Zero-init the output layers: zero offset at initialization, so the
        # snapper starts as an identity mapping.
        nn.init.zeros_(self.offset_head[-1].weight)
        nn.init.zeros_(self.offset_head[-1].bias)
        nn.init.zeros_(self.conf_head[-1].weight)
        nn.init.zeros_(self.conf_head[-1].bias)

        offsets = torch.linspace(-self.radius, self.radius, self.n_samples)
        self.register_buffer("profile_offsets", offsets, persistent=True)
        self.last_stats: Dict[str, torch.Tensor] = {}

    # -- Shared forward path: sample profile -> predict offset and confidence --

    def _predict(self, points: torch.Tensor, maps: Sequence[torch.Tensor],
                 image: Optional[torch.Tensor] = None):
        """Predict per-vertex normal offsets and confidence logits.

        Returns:
            offset: [B, N], bounded to (-radius, radius).
            conf_logit: [B, N].
            normal: [B, N, 2], unit normals of the input contour.
        """
        if points.shape[1] != self.n_points:
            raise ValueError(f"expected {self.n_points} points, got {points.shape[1]}")
        tangent, normal = contour_frame(points)

        # Profile locations: K equidistant samples along the normal of each vertex.
        offsets = self.profile_offsets.to(device=points.device, dtype=points.dtype)
        loc = points[:, :, None, :] + normal[:, :, None, :] * offsets.view(1, 1, -1, 1)
        loc = loc.clamp(-1.0, 1.0)

        # Feature profile: mean of the projected per-level samples, plus RGB.
        profile = None
        for level in range(self.levels):
            raw = sample_at_points(maps[level], loc)              # [B, N, K, C]
            emb = self.sample_proj[level](raw)
            profile = emb if profile is None else profile + emb
        profile = profile / float(self.levels)
        if self.rgb_proj is not None and image is not None:
            rgb = sample_at_points(image, loc)
            profile = profile + self.rgb_proj(rgb)
        token = self.profile_mlp(profile.flatten(-2))

        # Geometry features; the radial distance is normalized by its mean
        # for scale invariance.
        center = points.mean(dim=1, keepdim=True)
        radial = torch.norm(points - center, dim=-1, keepdim=True)
        scale = radial.mean(dim=1, keepdim=True).clamp(min=1e-4)
        radial_norm = radial / scale
        curv = curvature_scalar(points)
        geom = torch.cat([points, tangent, normal, curv, radial_norm], dim=-1)
        token = token + self.geom_mlp(geom)
        token = token + self.ring_proj(self.ring_features.to(dtype=token.dtype)).expand(points.shape[0], -1, -1)
        token = self.ring_block(token)

        offset = self.radius * torch.tanh(self.offset_head(token).squeeze(-1))
        conf_logit = self.conf_head(token).squeeze(-1)
        return offset, conf_logit, normal

    # -- Shared reconstruction of the corrected contour from (offset, conf) ----

    def _apply_gate(self, points: torch.Tensor, offset: torch.Tensor,
                     conf_logit: torch.Tensor, normal: torch.Tensor) -> torch.Tensor:
        """Apply the confidence-gated correction: p + n * sigmoid(c)^power * offset.

        Factored out of `forward()` so that `teacher_loss` reconstructs exactly
        the contour produced at inference for the BoundaryIoU term.
        """
        conf_prob = torch.sigmoid(conf_logit)
        gate = conf_prob.pow(self.confidence_power)
        correction = gate * offset
        return (points + normal * correction.unsqueeze(-1)).clamp(-1.0, 1.0)

    # -- Public inference API --------------------------------------------------

    def forward(self, points: torch.Tensor, maps: Sequence[torch.Tensor],
                image: Optional[torch.Tensor] = None, hard: bool = True) -> torch.Tensor:
        """Refine a finished contour (e.g. the DDIM sampling output).

        Args:
            points: Contour to refine, [B, N, 2] in [-1, 1].
            maps: Raw backbone feature maps, finest level first.
            image: Optional RGB image, [B, 3, H, W].
            hard: Kept for API compatibility; the gate is always applied in full.
        Returns:
            Corrected contour, [B, N, 2]. Diagnostics are stored in `last_stats`.
        """
        offset, conf_logit, normal = self._predict(points, maps, image=image)
        corrected = self._apply_gate(points, offset, conf_logit, normal)

        with torch.no_grad():
            conf_prob = torch.sigmoid(conf_logit)
            gate = conf_prob.pow(self.confidence_power)
            correction = gate * offset
            self.last_stats = {
                "snap_offset_abs": offset.detach().abs().mean(),
                "snap_conf_mean": conf_prob.detach().mean(),
                "snap_gate_mean": gate.detach().mean(),
                "snap_correction_abs": correction.detach().abs().mean(),
            }
        return corrected

    # -- Teacher loss for standalone snapper training --------------------------

    def teacher_loss(
        self,
        gt_points: torch.Tensor,
        maps: Sequence[torch.Tensor],
        image: Optional[torch.Tensor] = None,
        perturb_max_offset: float = 0.08,
        perturb_smooth_passes: int = 2,
        confidence_radius: float = 0.060,
        tangent_tolerance: float = 0.040,
        masks: Optional[torch.Tensor] = None,
        lambda_biou: float = 0.0,
        biou_size: int = 64,
    ):
        """Teacher loss on normal-perturbed ground-truth contours.

        Inputs are ground-truth contours perturbed along their normals, which
        gives the confidence head sufficient "close to ground truth" positives
        from the start, independent of the diffusion model's training state.

        Targets:
            confidence: 1 if the nearest GT point is within `confidence_radius`,
                its tangential displacement is within `tangent_tolerance` and
                its normal displacement is within the search radius; else 0.
            offset: signed normal displacement to the nearest GT point,
                clamped to [-radius, radius].

        Loss terms:
            loss_conf:   BCE on confidence logits with a clamped positive weight.
            loss_offset: smooth-L1 on radius-normalized offsets, restricted to
                         confident targets.
            loss_biou:   optional BoundaryIoU between the gated corrected
                         contour (identical to the inference output) and the
                         GT mask; active only if `lambda_biou > 0` and `masks`
                         is given. With `lambda_biou == 0` the loss reduces to
                         offset + confidence regression.

        Returns:
            (total_loss, parts), where `parts` holds detached scalars for logging.
            Usage:
                snap_loss, snap_parts = snapper.teacher_loss(points, cond_maps_raw)
                loss = diffusion_loss + lambda_snap * snap_loss
        """
        teacher_points = perturb_along_normals(
            gt_points.detach(), max_offset=perturb_max_offset,
            smooth_passes=perturb_smooth_passes,
        )
        offset_pred, conf_logit, normal = self._predict(teacher_points, maps, image=image)

        # Targets via nearest-neighbor correspondence to the GT contour.
        with torch.no_grad():
            d = torch.cdist(teacher_points.float(), gt_points.float(), p=2)
            min_dist, idx = d.min(dim=2)
            b_idx = torch.arange(teacher_points.shape[0], device=teacher_points.device)[:, None]
            nearest = gt_points[b_idx, idx]
            delta = nearest - teacher_points
            tangent = torch.stack([normal[..., 1], -normal[..., 0]], dim=-1)
            signed_n = (delta * normal).sum(dim=-1)
            signed_t = (delta * tangent).sum(dim=-1).abs()
            target_conf = (
                (min_dist <= float(confidence_radius))
                & (signed_t <= float(tangent_tolerance))
                & (signed_n.abs() <= self.radius)
            ).to(teacher_points.dtype)
            target_offset = signed_n.clamp(-self.radius, self.radius)

        # Confidence: class-balanced BCE (positive weight clamped to [1, 12]).
        pos = target_conf.sum().clamp(min=1.0)
        neg = (target_conf.numel() - target_conf.sum()).clamp(min=1.0)
        pos_weight = (neg / pos).clamp(1.0, 12.0)
        loss_conf = F.binary_cross_entropy_with_logits(conf_logit, target_conf, pos_weight=pos_weight)

        # Offset: supervised only where the target is confident.
        w = target_conf
        if float(w.sum().item()) > 0:
            loss_offset = F.smooth_l1_loss(
                offset_pred / max(self.radius, 1e-6),
                target_offset / max(self.radius, 1e-6),
                reduction="none",
            )
            loss_offset = (loss_offset * w).sum() / w.sum().clamp(min=1.0)
        else:
            loss_offset = offset_pred.new_zeros(())

        # Optional mask-level term on the gated corrected contour.
        if masks is not None and lambda_biou > 0:
            corrected = self._apply_gate(teacher_points, offset_pred, conf_logit, normal)
            loss_biou = soft_boundary_iou_loss(corrected, masks, size=biou_size)
        else:
            loss_biou = offset_pred.new_zeros(())

        total = loss_offset + loss_conf + lambda_biou * loss_biou
        parts = {
            "snap_teacher_loss_offset": loss_offset.detach(),
            "snap_teacher_loss_conf": loss_conf.detach(),
            "snap_teacher_conf_rate": target_conf.detach().mean(),
            "snap_teacher_target_dist": min_dist.detach().mean(),
            "snap_teacher_loss_biou": loss_biou.detach(),
        }
        return total, parts