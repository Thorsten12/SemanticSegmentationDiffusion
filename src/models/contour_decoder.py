"""Global-to-local contour decoder for sparse geometric diffusion.

V2 fixes the core high-noise failure mode of the original DP2Seg decoder:
Gaussian diffusion variables are never used directly as image coordinates.
The denoiser first maps its latent state to bounded coordinates.  The decoder
then combines:

* global cross-attention to a compact coarse 2-D memory (localization), and
* sparse deformable sampling around N contour vertices (boundary refinement).

The local path is intentionally *point sparse*: it samples raw backbone maps
first and projects only the N x K sampled vectors.  It does not run a dense
pixel decoder or a hidden-dimensional 1x1 projection over the full-resolution
feature map.
"""

from __future__ import annotations

import math
from typing import Dict, Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F

from .positional import SharedCoordinatePE, sample_deformable


def _time_scale_prior(t_norm: torch.Tensor, n_scales: int, sigma: float = 0.75):
    """Maps are finest-first: t=0 prefers fine, t=1 prefers coarse."""
    if n_scales <= 1:
        return torch.zeros(t_norm.shape[0], 1, device=t_norm.device, dtype=t_norm.dtype)
    idx = torch.arange(n_scales, device=t_norm.device, dtype=t_norm.dtype)
    pref = t_norm[:, None] * (n_scales - 1)
    return -0.5 * ((idx[None] - pref) / sigma) ** 2


class DeformablePointSampler(nn.Module):
    """Read K learned locations from one raw backbone map for every vertex."""

    def __init__(self, in_ch: int, hidden_dim: int, n_samples: int = 5):
        super().__init__()
        self.n_samples = max(int(n_samples), 1)
        n_learned = max(self.n_samples - 1, 0)
        self.offset_head = nn.Linear(hidden_dim, n_learned * 2)
        self.weight_head = nn.Linear(hidden_dim, self.n_samples)
        self.sample_proj = nn.Sequential(
            nn.Linear(in_ch, hidden_dim), nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.out = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim), nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        # Start as an aligned sampler (all learned offsets zero, uniform K weights).
        nn.init.zeros_(self.offset_head.weight)
        nn.init.zeros_(self.offset_head.bias)
        nn.init.zeros_(self.weight_head.weight)
        nn.init.zeros_(self.weight_head.bias)

    def forward(self, feat, points, tokens, radius):
        b, n, _ = points.shape
        if self.n_samples == 1:
            offsets = points.new_zeros(b, n, 1, 2)
        else:
            learned = self.offset_head(tokens).view(b, n, self.n_samples - 1, 2)
            learned = torch.tanh(learned) * radius.unsqueeze(2)
            zero = learned.new_zeros(b, n, 1, 2)
            offsets = torch.cat([zero, learned], dim=2)
        locations = points.unsqueeze(2) + offsets
        samples = sample_deformable(feat, locations)                 # [B,N,K,C]
        samples = self.sample_proj(samples)                           # sparse projection
        weights = torch.softmax(self.weight_head(tokens), dim=-1)    # [B,N,K]
        fused = (samples * weights.unsqueeze(-1)).sum(dim=2)
        return self.out(fused), offsets, weights


class GlobalContourPrior(nn.Module):
    """Low-capacity image-conditioned ellipse for robust high-noise localization."""

    def __init__(self, hidden_dim: int):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim), nn.GELU(),
            nn.Linear(hidden_dim, 5),
        )
        nn.init.zeros_(self.mlp[-1].weight)
        nn.init.zeros_(self.mlp[-1].bias)

    def forward(self, global_context, phase):
        raw = self.mlp(global_context)
        center = 0.70 * torch.tanh(raw[:, 0:2])
        radii = 0.10 + 0.56 * torch.sigmoid(raw[:, 2:4])
        angle = math.pi * torch.tanh(raw[:, 4:5])

        c, s = phase[..., 0], phase[..., 1]
        rx = radii[:, None, 0]
        ry = radii[:, None, 1]
        ca, sa = angle.cos(), angle.sin()
        x = center[:, None, 0] + rx * c * ca - ry * s * sa
        y = center[:, None, 1] + rx * c * sa + ry * s * ca
        return torch.stack([x, y], dim=-1).clamp(-0.97, 0.97)


class SpatialFourierContourProposal(nn.Module):
    """Predict a low-frequency closed contour from spatial global memory.

    Unlike the legacy ellipse prior, the learned shape query attends to tokens
    that already contain 2-D positional encoding.  Four Fourier harmonics need
    only 18 outputs (center + 4 coefficients per harmonic) while supporting
    asymmetric, elongated, mildly concave, and near-border proposals.
    """

    def __init__(self, hidden_dim: int, num_heads: int, harmonics: int = 4):
        super().__init__()
        self.harmonics = max(1, int(harmonics))
        self.query = nn.Parameter(torch.zeros(1, 1, hidden_dim))
        nn.init.normal_(self.query, std=0.02)
        self.query_norm = nn.LayerNorm(hidden_dim)
        self.memory_norm = nn.LayerNorm(hidden_dim)
        self.attn = nn.MultiheadAttention(
            hidden_dim, num_heads, batch_first=True, dropout=0.0,
        )
        self.mlp = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim), nn.GELU(),
            nn.Linear(hidden_dim, 2 + 4 * self.harmonics),
        )
        nn.init.zeros_(self.mlp[-1].weight)
        nn.init.zeros_(self.mlp[-1].bias)

    def forward(self, memory, phase):
        b = memory.shape[0]
        q = self.query.expand(b, -1, -1)
        shape_token, _ = self.attn(
            self.query_norm(q), self.memory_norm(memory), self.memory_norm(memory),
            need_weights=False,
        )
        raw = self.mlp((q + shape_token)[:, 0])
        center = 0.85 * torch.tanh(raw[:, :2])
        coeff = raw[:, 2:].view(b, self.harmonics, 2, 2)
        coeff = 0.55 * torch.tanh(coeff)

        # Stable initialization: a radius-0.35 circle in the first harmonic.
        base = torch.zeros_like(coeff)
        base[:, 0, 0, 0] = 0.35  # x <- cos(theta)
        base[:, 0, 1, 1] = 0.35  # y <- sin(theta)
        coeff = coeff + base

        theta = torch.atan2(phase[..., 1], phase[..., 0])
        ks = torch.arange(
            1, self.harmonics + 1, device=memory.device, dtype=memory.dtype,
        ).view(1, self.harmonics, 1)
        angles = ks * theta.unsqueeze(1)
        basis = torch.stack([angles.cos(), angles.sin()], dim=-1)  # [1,K,N,2]
        xy = torch.einsum("bkdc,bknc->bnd", coeff, basis)
        return (center[:, None, :] + xy).clamp(-0.995, 0.995)


class GlobalLocalContourDecoder(nn.Module):
    """Bridge a dense 2-D feature pyramid to an ordered 1-D contour sequence."""

    def __init__(
        self,
        scale_channels: Sequence[int],
        hidden_dim: int = 128,
        time_dim: int = 128,
        num_heads: int = 4,
        coord_fourier_bands: int = 6,
        deformable_samples: int = 5,
        local_radius_min: float = 0.025,
        local_radius_max: float = 0.22,
        global_levels: int = 3,
        global_grid: int = 14,
        timesteps: int = 1000,
        boundary_dim: int = 0,
        proposal_type: str = "ellipse",
        fourier_harmonics: int = 4,
    ):
        super().__init__()
        self.scale_channels = list(scale_channels)
        self.n_scales = len(self.scale_channels)
        self.hidden_dim = int(hidden_dim)
        self.timesteps = max(int(timesteps), 1)
        self.local_radius_min = float(local_radius_min)
        self.local_radius_max = float(local_radius_max)
        self.global_levels = max(1, min(int(global_levels), self.n_scales))
        self.global_grid = max(int(global_grid), 2)
        self.boundary_dim = max(int(boundary_dim), 0)
        self.proposal_type = str(proposal_type)
        if self.proposal_type not in ("ellipse", "fourier"):
            raise ValueError("proposal_type must be 'ellipse' or 'fourier'")

        # Sparse local path: project only sampled C-vectors, not entire HxW maps.
        self.samplers = nn.ModuleList([
            DeformablePointSampler(c, hidden_dim, deformable_samples)
            for c in self.scale_channels
        ])

        # Compact global path: only selected coarse maps are pooled and flattened.
        self.global_proj = nn.ModuleList([
            nn.Linear(c, hidden_dim) for c in self.scale_channels
        ])
        self.coord_pe = SharedCoordinatePE(num_bands=coord_fourier_bands)
        self.spatial_pe_proj = nn.Linear(self.coord_pe.out_dim, hidden_dim, bias=False)
        self.level_emb = nn.Parameter(torch.zeros(self.n_scales, hidden_dim))
        nn.init.normal_(self.level_emb, std=0.02)

        self.scale_delta = nn.Linear(time_dim, self.n_scales)
        nn.init.zeros_(self.scale_delta.weight)
        nn.init.zeros_(self.scale_delta.bias)

        self.cross_norm_q = nn.LayerNorm(hidden_dim)
        self.cross_norm_m = nn.LayerNorm(hidden_dim)
        self.global_attn = nn.MultiheadAttention(
            hidden_dim, num_heads, batch_first=True, dropout=0.0,
        )
        self.global_ff = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim * 2), nn.GELU(),
            nn.Linear(hidden_dim * 2, hidden_dim),
        )

        self.local_norm = nn.LayerNorm(hidden_dim)
        self.local_fuse = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim), nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        if self.boundary_dim > 0:
            self.boundary_sampler = DeformablePointSampler(
                self.boundary_dim, hidden_dim, max(3, deformable_samples)
            )
            self.boundary_fuse = nn.Sequential(
                nn.LayerNorm(hidden_dim),
                nn.Linear(hidden_dim, hidden_dim), nn.GELU(),
                nn.Linear(hidden_dim, hidden_dim),
            )
        else:
            self.boundary_sampler = None
            self.boundary_fuse = None
        if self.proposal_type == "ellipse":
            self.prior = GlobalContourPrior(hidden_dim)
        else:
            self.prior = SpatialFourierContourProposal(
                hidden_dim, num_heads=num_heads, harmonics=fourier_harmonics,
            )
        self.last_stats: Dict[str, torch.Tensor] = {}

    def _make_global_memory(self, maps):
        memories, contexts = [], []
        start = self.n_scales - self.global_levels
        for level in range(start, self.n_scales):
            m = maps[level]
            h, w = m.shape[-2:]
            if max(h, w) > self.global_grid:
                m = F.adaptive_avg_pool2d(m, (self.global_grid, self.global_grid))
                h, w = m.shape[-2:]

            raw = m.flatten(2).transpose(1, 2)                         # [B,L,C]
            feat_tokens = self.global_proj[level](raw)                 # [B,L,H]
            pe = self.coord_pe.encode_grid(h, w, m.device, m.dtype)
            pe = pe.flatten(2).transpose(1, 2)                         # [1,L,P]
            pe_tokens = self.spatial_pe_proj(pe).expand(m.shape[0], -1, -1)
            tagged = feat_tokens + pe_tokens + self.level_emb[level].view(1, 1, -1)
            memories.append(tagged)
            contexts.append(feat_tokens.mean(dim=1))

        memory = torch.cat(memories, dim=1)
        global_context = torch.stack(contexts, dim=0).mean(dim=0)
        return memory, global_context

    def propose_from_global(self, memory, global_context, phase):
        if self.proposal_type == "fourier":
            return self.prior(memory, phase)
        return self.prior(global_context, phase)

    def prepare_global(self, maps, phase):
        memory, global_context = self._make_global_memory(maps)
        proposal = self.propose_from_global(memory, global_context, phase)
        return memory, global_context, proposal

    def forward(self, maps, points, tokens, t_vec, t, phase, boundary_feat=None,
                global_memory=None, global_context=None, prior_points=None):
        if len(maps) != self.n_scales:
            raise ValueError(f"expected {self.n_scales} feature maps, got {len(maps)}")
        t_norm = (t.float() / max(self.timesteps - 1, 1)).clamp(0.0, 1.0)

        # ----- global branch -------------------------------------------------
        if global_memory is None or global_context is None:
            memory, global_context = self._make_global_memory(maps)
        else:
            memory, global_context = global_memory, global_context
        q = self.cross_norm_q(tokens)
        m = self.cross_norm_m(memory)
        global_update, _ = self.global_attn(q, m, m, need_weights=False)
        global_gate = (0.20 + 0.80 * t_norm).view(-1, 1, 1)
        tokens = tokens + global_gate * global_update
        tokens = tokens + global_gate * self.global_ff(tokens)

        # ----- sparse local branch ------------------------------------------
        scale_logits = _time_scale_prior(t_norm, self.n_scales) + self.scale_delta(t_vec)
        scale_gates = torch.softmax(scale_logits, dim=-1)
        radius = self.local_radius_min + (
            self.local_radius_max - self.local_radius_min
        ) * t_norm
        radius = radius.view(-1, 1, 1)

        local_sum = torch.zeros_like(tokens)
        offset_abs, offset_entropy = [], []
        for i, (feat, sampler) in enumerate(zip(maps, self.samplers)):
            local_i, offsets, weights = sampler(feat, points, tokens, radius)
            local_sum = local_sum + scale_gates[:, i].view(-1, 1, 1) * local_i
            offset_abs.append(offsets.detach().abs().mean())
            ent = -(weights.detach() * (weights.detach() + 1e-8).log()).sum(dim=-1).mean()
            offset_entropy.append(ent)

        local_gate = (0.15 + 0.85 * (1.0 - t_norm)).view(-1, 1, 1)
        local_update = self.local_fuse(torch.cat([self.local_norm(tokens), local_sum], dim=-1))
        tokens = tokens + local_gate * local_update

        # Explicit boundary-conditioned branch.  It is intentionally strongest
        # near t=0, when the evolving vertices are close enough for a local edge
        # search to be meaningful.
        boundary_gate = t_norm.new_zeros(t_norm.shape[0], 1, 1)
        boundary_offset = t_norm.new_zeros(())
        if self.boundary_sampler is not None and boundary_feat is not None:
            boundary_radius = (0.55 * radius).clamp(min=self.local_radius_min * 0.5)
            boundary_local, b_offsets, _ = self.boundary_sampler(
                boundary_feat, points, tokens, boundary_radius
            )
            boundary_gate = (0.90 * (1.0 - t_norm).pow(2)).view(-1, 1, 1)
            tokens = tokens + boundary_gate * self.boundary_fuse(boundary_local)
            boundary_offset = b_offsets.detach().abs().mean()

        if prior_points is None:
            prior_points = self.propose_from_global(memory, global_context, phase)
        stats = {
            "t_mean": t_norm.detach().mean(),
            "global_gate": global_gate.detach().mean(),
            "local_gate": local_gate.detach().mean(),
            "boundary_gate": boundary_gate.detach().mean(),
            "boundary_offset_abs": boundary_offset,
            "scale_fine": scale_gates[:, 0].detach().mean(),
            "scale_coarse": scale_gates[:, -1].detach().mean(),
            "scale_entropy": (-(scale_gates.detach() * (scale_gates.detach() + 1e-8).log())
                              .sum(dim=-1).mean()),
            "offset_abs": torch.stack(offset_abs).mean() if offset_abs else t_norm.new_zeros(()),
            "offset_entropy": (torch.stack(offset_entropy).mean()
                               if offset_entropy else t_norm.new_zeros(())),
            "memory_tokens": t_norm.new_tensor(float(memory.shape[1])),
            "prior_radius": torch.norm(
                prior_points - prior_points.mean(dim=1, keepdim=True), dim=-1
            ).mean().detach(),
        }
        for i in range(self.n_scales):
            stats[f"scale_s{i}"] = scale_gates[:, i].detach().mean()
        self.last_stats = stats
        return tokens, prior_points, stats


# Compatibility with the old public symbol.
DP2Seg = GlobalLocalContourDecoder


def stats_to_float(stats):
    out = {}
    for k, v in stats.items():
        if torch.is_tensor(v):
            out[k] = float(v.detach().float().cpu().mean())
        else:
            out[k] = float(v)
    return out
