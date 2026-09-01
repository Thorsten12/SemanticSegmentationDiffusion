"""P2SDiff V5.2: V5 Fourier + XY residual diffusion with exact local correction.

The successful V5 coarse/diffusion path is preserved.  A tiny sparse boundary
module is active only at low noise and predicts a signed normal offset together
with a directly supervised confidence score.  There is no dense boundary head
and no post-snapper; the low-t correction is part of the denoiser x0 prediction.
"""

from __future__ import annotations

import math
from typing import Dict, List, Union

import torch
import torch.nn as nn
import torch.nn.functional as F

from .exact_boundary import (
    SparseExactBoundaryCorrector,
    exact_boundary_supervision_loss,
    contour_frame,
)
from .contour_decoder import GlobalLocalContourDecoder
from .positional import SharedCoordinatePE


def timestep_embedding(t: torch.Tensor, dim: int, max_period: int = 10000) -> torch.Tensor:
    half = dim // 2
    freqs = torch.exp(
        -math.log(max_period)
        * torch.arange(half, device=t.device, dtype=torch.float32)
        / max(half, 1)
    )
    args = t.float()[:, None] * freqs[None]
    emb = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
    if dim % 2:
        emb = F.pad(emb, (0, 1))
    return emb


def stable_atanh(x: torch.Tensor, eps: float = 1e-4) -> torch.Tensor:
    x = x.clamp(-1.0 + eps, 1.0 - eps)
    return 0.5 * (torch.log1p(x) - torch.log1p(-x))


class CircularMixerBlock(nn.Module):
    """Closed-contour local mixing followed by global self-attention."""

    def __init__(self, hidden_dim: int, num_heads: int):
        super().__init__()
        self.norm1 = nn.LayerNorm(hidden_dim)
        self.conv = nn.Conv1d(
            hidden_dim, hidden_dim, kernel_size=3, padding=1,
            padding_mode="circular", groups=1,
        )
        self.norm2 = nn.LayerNorm(hidden_dim)
        self.attn = nn.MultiheadAttention(
            hidden_dim, num_heads, batch_first=True, dropout=0.0,
        )
        self.norm3 = nn.LayerNorm(hidden_dim)
        self.ff = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 3), nn.GELU(),
            nn.Linear(hidden_dim * 3, hidden_dim),
        )

    def forward(self, x):
        y = self.norm1(x)
        y = self.conv(y.transpose(1, 2)).transpose(1, 2)
        x = x + F.gelu(y)
        y = self.norm2(x)
        y, _ = self.attn(y, y, y, need_weights=False)
        x = x + y
        x = x + self.ff(self.norm3(x))
        return x


Condition = Dict[str, Union[List[torch.Tensor], torch.Tensor]]


def resample_closed_contour_arclength(points: torch.Tensor) -> torch.Tensor:
    """Piecewise-differentiable uniform arc-length resampling of closed polygons.

    The discrete segment lookup is intentionally non-differentiable, while the
    interpolation within the selected segment retains gradients to the proposal.
    The first output vertex stays at the original phase-zero vertex.
    """
    if points.ndim != 3 or points.shape[-1] != 2:
        raise ValueError(f"expected [B,N,2] contour points, got {tuple(points.shape)}")
    n = points.shape[1]
    if n < 3:
        raise ValueError("a closed contour needs at least three points")

    next_points = torch.roll(points, shifts=-1, dims=1)
    segments = next_points - points
    lengths = torch.linalg.vector_norm(segments, dim=-1).clamp_min(1e-8)
    cumulative = torch.cat([
        torch.zeros_like(lengths[:, :1]),
        lengths.cumsum(dim=1),
    ], dim=1)
    fractions = torch.arange(n, device=points.device, dtype=points.dtype) / float(n)
    targets = cumulative[:, -1:] * fractions.unsqueeze(0)

    # searchsorted has no useful gradient through the integer segment choice.
    # Detaching only that choice leaves the piecewise-linear interpolation and
    # total-perimeter dependence differentiable.
    indices = torch.searchsorted(
        cumulative.detach().contiguous(), targets.detach().contiguous(), right=True,
    ) - 1
    indices = indices.clamp_(0, n - 1)
    starts = cumulative[:, :-1].gather(1, indices)
    selected_lengths = lengths.gather(1, indices)
    alpha = ((targets - starts) / selected_lengths).clamp(0.0, 1.0)
    gather_index = indices.unsqueeze(-1).expand(-1, -1, 2)
    p0 = points.gather(1, gather_index)
    p1 = next_points.gather(1, gather_index)
    return p0 + alpha.unsqueeze(-1) * (p1 - p0)


class ContourDenoiser(nn.Module):
    """Predict clean latent contour z0 and refine it against explicit boundaries."""

    def __init__(
        self,
        n_points: int = 100,
        hidden_dim: int = 128,
        num_layers: int = 4,
        num_heads: int = 4,
        scale_channels=(64,),
        coord_fourier_bands: int = 6,
        tag_maps: bool = True,
        tag_mode: str = "concat",
        timesteps: int = 1000,
        deformable_samples: int = 5,
        local_radius_min: float = 0.025,
        local_radius_max: float = 0.22,
        global_levels: int = 3,
        global_grid: int = 14,
        contour_phase_bands: int = 4,
        boundary_dim: int = 32,
        boundary_levels: int = 3,
        snap_candidates: int = 15,
        snap_iterations: int = 3,
        snap_radius_max: float = 0.16,
        snap_radius_min: float = 0.015,
        snap_temperature: float = 0.16,
        snap_profile_radius: float = 0.018,
        proposal_type: str = "ellipse",
        diffusion_target: str = "absolute",
        fourier_harmonics: int = 4,
        proposal_arclength_resample: bool = False,
        residual_scale: float = 1.0,
        exact_boundary_enabled: bool = True,
        exact_boundary_levels: int = 2,
        exact_boundary_samples: int = 11,
        exact_boundary_radius: float = 0.10,
        exact_boundary_profile_dim: int = 20,
        exact_boundary_hidden: int = 64,
        exact_boundary_ring_bands: int = 4,
        exact_boundary_relative_bias: float = 0.12,
        exact_boundary_confidence_power: float = 2.0,
        exact_boundary_low_t_fraction: float = 0.30,
        exact_confidence_radius: float = 0.060,
        exact_tangent_tolerance: float = 0.040,
        exact_use_rgb: bool = True,
    ):
        super().__init__()
        del tag_maps, tag_mode
        self.n_points = int(n_points)
        self.hidden_dim = int(hidden_dim)
        self.timesteps = int(timesteps)
        self.boundary_dim = int(scale_channels[0])  # raw finest skip; no dense boundary head.
        self.diffusion_target = str(diffusion_target)
        if self.diffusion_target not in ("absolute", "residual"):
            raise ValueError("diffusion_target must be 'absolute' or 'residual'")
        self.proposal_arclength_resample = bool(proposal_arclength_resample)
        self.residual_scale = float(residual_scale)
        if self.residual_scale <= 0:
            raise ValueError("residual_scale must be positive")
        self.last_stats = {}
        self.exact_boundary_enabled = bool(exact_boundary_enabled)
        self.exact_boundary_low_t_fraction = float(exact_boundary_low_t_fraction)
        self.exact_confidence_radius = float(exact_confidence_radius)
        self.exact_tangent_tolerance = float(exact_tangent_tolerance)
        self._last_exact_aux = None

        self.time_mlp = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim), nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.coord_pe = SharedCoordinatePE(num_bands=coord_fourier_bands)
        self.coord_mlp = nn.Sequential(
            nn.Linear(self.coord_pe.out_dim, hidden_dim), nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.latent_mlp = nn.Sequential(
            nn.Linear(2, hidden_dim), nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
        )

        phi = (-0.5 * math.pi
               + 2.0 * math.pi * torch.arange(self.n_points, dtype=torch.float32) / self.n_points)
        phase = torch.stack([phi.cos(), phi.sin()], dim=-1).unsqueeze(0)
        self.register_buffer("phase", phase, persistent=True)
        phase_feats = []
        for k in range(1, max(int(contour_phase_bands), 1) + 1):
            phase_feats.extend([(k * phi).cos(), (k * phi).sin()])
        phase_feats = torch.stack(phase_feats, dim=-1).unsqueeze(0)
        self.register_buffer("phase_features", phase_feats, persistent=True)
        self.phase_mlp = nn.Sequential(
            nn.Linear(phase_feats.shape[-1], hidden_dim), nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
        )

        self.decoder = GlobalLocalContourDecoder(
            scale_channels=list(scale_channels),
            hidden_dim=hidden_dim,
            time_dim=hidden_dim,
            num_heads=num_heads,
            coord_fourier_bands=coord_fourier_bands,
            deformable_samples=deformable_samples,
            local_radius_min=local_radius_min,
            local_radius_max=local_radius_max,
            global_levels=global_levels,
            global_grid=global_grid,
            timesteps=timesteps,
            boundary_dim=self.boundary_dim,
            proposal_type=proposal_type,
            fourier_harmonics=fourier_harmonics,
        )
        self.mixers = nn.ModuleList([
            CircularMixerBlock(hidden_dim, num_heads) for _ in range(num_layers)
        ])
        self.final_norm = nn.LayerNorm(hidden_dim)
        self.residual_head = nn.Sequential(
            nn.Linear(hidden_dim + 4, hidden_dim), nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim // 2), nn.GELU(),
            nn.Linear(hidden_dim // 2, 2),
        )
        # Preserve V2's robust initialization behavior.
        nn.init.zeros_(self.residual_head[-1].weight)
        nn.init.zeros_(self.residual_head[-1].bias)

        self.exact_corrector = SparseExactBoundaryCorrector(
            scale_channels=scale_channels,
            n_points=self.n_points,
            levels=exact_boundary_levels,
            n_samples=exact_boundary_samples,
            radius=exact_boundary_radius,
            profile_dim=exact_boundary_profile_dim,
            hidden_dim=exact_boundary_hidden,
            ring_bands=exact_boundary_ring_bands,
            num_heads=num_heads,
            relative_bias_strength=exact_boundary_relative_bias,
            confidence_power=exact_boundary_confidence_power,
            use_rgb=exact_use_rgb,
        ) if self.exact_boundary_enabled else None

    def latent_to_points(self, z):
        return torch.tanh(z)

    def prepare_condition(self, cond_maps, image=None) -> Condition:
        """Cache the encoder pyramid and optional raw image for sparse RGB profiles."""
        if isinstance(cond_maps, dict):
            return cond_maps
        maps = list(cond_maps)
        memory, global_context, proposal = self.decoder.prepare_global(maps, self.phase)
        if self.proposal_arclength_resample:
            proposal = resample_closed_contour_arclength(proposal)
        return {
            "maps": maps,
            "image": image,
            "global_memory": memory, "global_context": global_context,
            "proposal_points": proposal,
        }

    def drop_condition(self, condition: Condition, keep: torch.Tensor) -> Condition:
        """Per-sample classifier-free drop while retaining a reusable structure."""
        maps = [m * keep for m in condition["maps"]]
        memory = condition["global_memory"] * keep.flatten(2).transpose(1, 2)
        context = condition["global_context"] * keep.flatten(1)
        # Residual diffusion uses the deterministic proposal as its coordinate
        # frame. Keep that frame under CFG dropout and drop only residual-detail
        # conditioning; otherwise conditional/unconditional states are defined
        # relative to different origins.
        if self.diffusion_target == "residual":
            proposal = condition["proposal_points"]
        else:
            proposal = self.decoder.propose_from_global(memory, context, self.phase)
            if self.proposal_arclength_resample:
                proposal = resample_closed_contour_arclength(proposal)
        image = condition.get("image")
        if image is not None:
            image = image * keep
        return {
            "maps": maps,
            "image": image,
            "global_memory": memory,
            "global_context": context,
            "proposal_points": proposal,
        }

    def proposal_points(self, condition: Condition):
        return self.prepare_condition(condition)["proposal_points"]

    def refine_points(self, points, condition: Condition, hard: bool = False):
        # Public compatibility: V5.2 has no post-snapper.  The final contour is
        # already corrected inside low-t diffusion.
        del condition, hard
        return points

    def exact_boundary_loss(self, target_points, condition, teacher_points=None):
        """Supervise exact local regression on the on-policy and teacher paths."""
        if (not self.exact_boundary_enabled) or self._last_exact_aux is None:
            z = target_points.new_zeros(())
            return z, z, {
                "exact_loss_offset": z.detach(), "exact_loss_conf": z.detach(),
                "exact_target_conf_rate": z.detach(), "exact_target_dist": z.detach(),
                "exact_target_tangent": z.detach(), "exact_teacher_loss_offset": z.detach(),
                "exact_teacher_loss_conf": z.detach(), "exact_teacher_conf_rate": z.detach(),
            }

        aux = self._last_exact_aux
        loss_off, loss_conf, parts = exact_boundary_supervision_loss(
            aux["offset"], aux["conf_logit"], aux["base_points"], aux["normal"],
            target_points, confidence_radius=self.exact_confidence_radius,
            tangent_tolerance=self.exact_tangent_tolerance,
            max_offset=self.exact_corrector.radius,
        )

        # Teacher branch provides abundant positive close-boundary examples so the
        # confidence head cannot collapse to "uncertain everywhere" early on.
        if teacher_points is not None:
            corr_t, off_t, conf_t, normal_t = self.exact_corrector(
                teacher_points, condition["maps"], image=condition.get("image")
            )
            del corr_t
            t_off, t_conf, t_parts = exact_boundary_supervision_loss(
                off_t, conf_t, teacher_points.detach(), normal_t, target_points,
                confidence_radius=max(self.exact_confidence_radius, self.exact_corrector.radius),
                tangent_tolerance=max(self.exact_tangent_tolerance, self.exact_corrector.radius),
                max_offset=self.exact_corrector.radius,
            )
        else:
            t_off = loss_off.new_zeros(())
            t_conf = loss_conf.new_zeros(())
            t_parts = {"exact_target_conf_rate": loss_off.new_zeros(())}
        parts = dict(parts)
        parts.update({
            "exact_teacher_loss_offset": t_off.detach(),
            "exact_teacher_loss_conf": t_conf.detach(),
            "exact_teacher_conf_rate": t_parts["exact_target_conf_rate"].detach(),
        })
        return loss_off + t_off, loss_conf + t_conf, parts

    def forward(self, latent_points, t, cond_maps):
        if latent_points.shape[1] != self.n_points:
            raise ValueError(
                f"denoiser built for {self.n_points} points, got {latent_points.shape[1]}"
            )

        condition = self.prepare_condition(cond_maps)
        maps = condition["maps"]
        proposal = condition["proposal_points"]
        if self.diffusion_target == "residual":
            bounded = (proposal + self.residual_scale * latent_points).clamp(-0.999, 0.999)
        else:
            bounded = self.latent_to_points(latent_points)
        t_vec = self.time_mlp(timestep_embedding(t, self.hidden_dim))
        query_pe = self.coord_pe.encode_points(bounded)

        phase_tokens = self.phase_mlp(self.phase_features).expand(latent_points.shape[0], -1, -1)
        tokens = (
            self.coord_mlp(query_pe)
            + self.latent_mlp(latent_points)
            + phase_tokens
            + t_vec.unsqueeze(1)
        )

        tokens, prior_points, stats = self.decoder(
            maps, bounded, tokens, t_vec, t=t, phase=self.phase,
            boundary_feat=maps[0],  # sparse low-t read from the raw finest skip
            global_memory=condition["global_memory"],
            global_context=condition["global_context"],
            prior_points=proposal,
        )
        for block in self.mixers:
            tokens = block(tokens)
        tokens = self.final_norm(tokens)

        residual = self.residual_head(
            torch.cat([tokens, bounded, latent_points], dim=-1)
        )
        if self.diffusion_target == "residual":
            pred_z0 = residual
            base_points = (proposal + self.residual_scale * pred_z0).clamp(-0.999, 0.999)
            pred_points_for_stats = base_points
            exact_stats = {}
            self._last_exact_aux = None
            if self.exact_boundary_enabled and self.exact_corrector is not None:
                t_norm = (t.float() / max(self.timesteps - 1, 1)).clamp(0.0, 1.0)
                low = max(self.exact_boundary_low_t_fraction, 1e-6)
                low_gate = ((low - t_norm) / low).clamp(0.0, 1.0).view(-1, 1)
                # During inference the exact branch is evaluated only in the
                # low-noise part of DDIM.  Training evaluates it every batch so
                # the auxiliary confidence/offset supervision stays well sampled.
                if self.training or bool((low_gate > 0).any().item()):
                    # Query geometry is detached inside the corrector.  The successful
                    # V5 coarse path therefore remains protected while corrected
                    # geometry is still optimized end-to-end through the additive x0.
                    correction, offset, conf_logit, normal = self.exact_corrector(
                        base_points, condition["maps"], image=condition.get("image")
                    )
                else:
                    correction = base_points.new_zeros(base_points.shape[:2])
                    offset = correction
                    conf_logit = correction
                    _, normal = contour_frame(base_points.detach())
                applied = low_gate * correction
                corrected = (base_points + normal * applied.unsqueeze(-1)).clamp(-0.999, 0.999)
                pred_z0 = (corrected - proposal) / float(self.residual_scale)
                pred_points_for_stats = corrected
                self._last_exact_aux = {
                    "base_points": base_points.detach(),
                    "offset": offset,
                    "conf_logit": conf_logit,
                    "normal": normal.detach(),
                    "low_gate": low_gate.detach(),
                }
                exact_stats = dict(getattr(self.exact_corrector, "last_stats", {}) or {})
                exact_stats["exact_low_t_gate"] = low_gate.detach().mean()
                exact_stats["exact_applied_abs"] = applied.detach().abs().mean()
        else:
            pred_z0 = stable_atanh(prior_points) + residual
            pred_points_for_stats = torch.tanh(pred_z0)
            exact_stats = {}
            self._last_exact_aux = None

        with torch.no_grad():
            stats = dict(stats)
            stats.update({
                "residual_abs": (
                    residual.detach().abs().mean() * self.residual_scale
                    if self.diffusion_target == "residual"
                    else residual.detach().abs().mean()
                ),
                "residual_state_abs": residual.detach().abs().mean(),
                "pred_radius": torch.norm(
                    pred_points_for_stats - pred_points_for_stats.mean(dim=1, keepdim=True), dim=-1
                ).mean(),
                "bounded_abs": bounded.detach().abs().mean(),
            })
            stats.update(exact_stats)
        self.last_stats = stats
        return pred_z0
