"""Boundary-point denoiser with per-point, time-gated multi-scale conditioning.

Given the current (noisy) point set, the diffusion timestep, and the raw
multi-scale feature pyramid from the image backbone, predict the clean
points x0.

Conditioning combines two complementary signals:

(1) Local, per-point evidence via a multi-scale point sampler. Every scale
    of the backbone pyramid is queried at the point's own (x, y), with a
    forced cosine time-gate (coarse scales dominate near pure noise, fine
    scales dominate near the end of sampling) plus a small bounded learned
    offset. Three interface-compatible sampler variants are available,
    selected via `sampler_type` (see MultiScalePointSampler /
    DeformableScalePointSampler / AttentionScalePointSampler below).

(2) Global re-localization via cross-attention to a compact, spatially
    tagged memory built from the coarsest backbone scales (GlobalMemory-
    CrossAttention). The local sampler only ever reads at or near the
    point's current position; it has no mechanism to correct a badly
    mislocalized point. The global branch closes this gap by letting every
    point attend over the whole image at once, gated to be strongest at
    high t (where localization error is largest) and fade out at low t
    (where the local branch is already precise). Optional, off by default.

Sequence model (closed ordered contour of N points): sinusoidal timestep
embedding + fixed positional embedding over point index; circular Conv1d
(the contour wraps around) -> Transformer encoder -> Conv1d -> MLP head ->
2D coordinate prediction. Coordinates enter only via an additive Fourier
path, kept out of the content fusion to avoid memorizing tiny datasets.

`residual_target` controls what the zero-initialized head output means:
  - "input": legacy, out = points + head_out (residual on the noisy
    diffusion sample). Used when there is no external initial guess.
  - "zero": out = head_out alone; the caller adds a deterministic coarse-
    contour proposal on top (x0 = proposal + out), so training starts
    exactly at the proposal and the model only learns the local correction.
  - "none": out = head_out, not zero-init (full x0 prediction).
"""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from .positional import FourierFeatures


def timestep_embedding(t: torch.Tensor, dim: int, max_period: int = 10000) -> torch.Tensor:
    """Sinusoidal embedding of integer timesteps -> [B, dim]."""
    half = dim // 2
    freqs = torch.exp(
        -math.log(max_period) * torch.arange(half, device=t.device, dtype=torch.float32) / half
    )
    args = t.float()[:, None] * freqs[None]
    emb = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
    if dim % 2:
        emb = F.pad(emb, (0, 1))
    return emb


def sample_deformable(feat_map: torch.Tensor, locations: torch.Tensor) -> torch.Tensor:
    """Bilinear sample a [B,C,H,W] map at [B,N,K,2] locations (range [-1,1]).

    Returns [B,N,K,C].
    """
    b, n, k, _ = locations.shape
    grid = locations.view(b, n * k, 1, 2)
    sampled = F.grid_sample(feat_map, grid, mode="bilinear",
                             padding_mode="border", align_corners=True)
    sampled = sampled.squeeze(-1).transpose(1, 2)
    return sampled.view(b, n, k, -1)


def _local_grid_offsets(window: int) -> torch.Tensor:
    """[(window*window), 2] grid of integer (dy, dx) shifts centered at
    (0, 0), e.g. window=5 -> shifts from -2..+2 along both axes (25
    candidates)."""
    r = window // 2
    ys, xs = torch.meshgrid(
        torch.arange(-r, r + 1), torch.arange(-r, r + 1), indexing="ij"
    )
    return torch.stack([ys.reshape(-1), xs.reshape(-1)], dim=-1).float()


def _gather_topk_scale_feats(feats: list, w: torch.Tensor, k: int) -> torch.Tensor:
    """Select, per sample, the k scales with the strongest time-gate weight
    and gather their (already time-weighted) features, sorted by scale
    index so slot position stays consistent with relative scale order.

    feats: list of S tensors [B,N,proj_dim].
    w: [B,S] scale-gate weights (determines which k scales are active).
    Returns [B,N,k,proj_dim]. All S scales are still computed in full
    (no compute skip) -- only the final fusion MLP sees k instead of S
    blocks.
    """
    _, topk_idx = torch.topk(w, k=k, dim=-1)
    topk_idx, _ = torch.sort(topk_idx, dim=-1)

    stacked = torch.stack(feats, dim=-2)
    b, n, s, d = stacked.shape
    idx_exp = topk_idx.view(b, 1, k, 1).expand(-1, n, -1, d)
    gathered = torch.gather(stacked, dim=2, index=idx_exp)
    return gathered


def _contour_frame(points: torch.Tensor, eps: float = 1e-6):
    """Tangent + normal for a closed, ordered contour [B,N,2]."""
    prev_p = torch.roll(points, shifts=1, dims=1)
    next_p = torch.roll(points, shifts=-1, dims=1)
    tangent = next_p - prev_p
    tangent = tangent / torch.norm(tangent, dim=-1, keepdim=True).clamp(min=eps)
    normal = torch.stack([-tangent[..., 1], tangent[..., 0]], dim=-1)
    return tangent, normal


class MultiScalePointSampler(nn.Module):
    """Per-point query of a multi-scale feature pyramid, gated by the timestep.

    The gate over scales is a forced cosine schedule (coarse -> fine as t
    goes T -> 0), plus a small learned offset bounded to +-`offset_scale`.
    A fully free `sigmoid(Linear(t_vec))` gate was found to collapse to a
    flat, time-independent band across all scales and never specialize;
    forcing the prior fixes that while still allowing a bounded correction.
    """

    def __init__(self, scale_channels, proj_dim=64, out_dim=128, time_dim=128,
                 offset_scale: float = 0.15, timesteps: int = 1000,
                 top_k_scales: int = None):
        super().__init__()
        self.n_scales = len(scale_channels)
        self.offset_scale = offset_scale
        # Clamp top_k_scales against the ACTUAL number of scales this sampler
        # sees (which may already be reduced by local_drop_coarsest before
        # scale_channels ever reaches here). Without this, a "take all
        # scales" sentinel value (e.g. 999) that was valid against the full
        # pyramid silently diverges from n_scales once the local sampler's
        # input pyramid is truncated -- effective_scales below would use the
        # unclamped sentinel for the MLP's input dimension, while forward()
        # concatenates only the true (smaller) n_scales feature blocks,
        # causing a shape mismatch at the first forward call.
        if top_k_scales is not None:
            top_k_scales = min(int(top_k_scales), self.n_scales)
        self.top_k_scales = top_k_scales
        self.register_buffer("_t_max", torch.tensor(float(max(timesteps - 1, 1))))

        self.projs = nn.ModuleList(nn.Linear(c, proj_dim) for c in scale_channels)
        self.scale_offset = nn.Linear(time_dim, self.n_scales)
        nn.init.zeros_(self.scale_offset.weight)
        nn.init.zeros_(self.scale_offset.bias)

        effective_scales = self.top_k_scales if self.top_k_scales is not None else self.n_scales
        self.mlp = nn.Sequential(
            nn.Linear(effective_scales * proj_dim + time_dim, out_dim * 2), nn.GELU(),
            nn.Linear(out_dim * 2, out_dim),
        )

    def _cosine_prior(self, t: torch.Tensor) -> torch.Tensor:
        """Forced local->global cosine schedule -> [B, n_scales], in [0, 1].

        Scales ordered local/fine (index 0) -> global/coarse (index
        n_scales - 1). At t = T: fine ~0, coarse ~1. At t = 0: fine ~1,
        coarse ~0.
        """
        progress = (t.float() / self._t_max).clamp(0.0, 1.0)
        if self.n_scales == 1:
            return torch.ones_like(progress).unsqueeze(-1)

        centers = torch.linspace(0.0, 1.0, self.n_scales, device=t.device)
        width = 1.0 / max(self.n_scales - 1, 1)
        dist = (progress.unsqueeze(-1) - centers.unsqueeze(0)).abs()
        x = (dist / (2 * width)).clamp(0.0, 1.0)
        return 0.5 * (1.0 + torch.cos(math.pi * x))

    def forward(self, maps, points, t_vec, t_raw=None, return_weights: bool = False):
        """maps: list of [B,C_s,H,W]; points [B,N,2]; t_vec [B,time_dim];
        t_raw: integer timesteps [B]. -> [B,N,out_dim] (+ gate weights [B,S])."""
        if t_raw is None:
            raise ValueError("MultiScalePointSampler.forward requires t_raw.")
        n = points.shape[1]
        grid = torch.clamp(points, -1.0, 1.0).unsqueeze(1)

        prior = self._cosine_prior(t_raw)
        offset = self.offset_scale * torch.tanh(self.scale_offset(t_vec))
        w = (prior + offset).clamp(0.0, 1.0)

        feats = []
        for i, (proj, m) in enumerate(zip(self.projs, maps)):
            s = F.grid_sample(m, grid, mode="bilinear",
                              padding_mode="border", align_corners=True)
            s = s.squeeze(2).transpose(1, 2)
            s = proj(s)
            s = s * w[:, i][:, None, None]
            feats.append(s)

        if self.top_k_scales is not None and self.top_k_scales < self.n_scales:
            gathered = _gather_topk_scale_feats(feats, w, self.top_k_scales)
            cat = gathered.flatten(-2)
        else:
            cat = torch.cat(feats, dim=-1)

        t_exp = t_vec.unsqueeze(1).expand(-1, n, -1)
        out = self.mlp(torch.cat([cat, t_exp], dim=-1))
        if return_weights:
            return out, w
        return out


class DeformableScalePointSampler(nn.Module):
    """Drop-in replacement for MultiScalePointSampler with per-point,
    per-scale deformable (multi-location, learned-offset) reads instead of
    a single aligned grid_sample per scale.

    Interface: forward(maps, points, t_vec, t_raw=t, return_weights=bool)
        -> guidance [B,N,out_dim]  (+ scale_w [B,S] if return_weights).

    The forced cosine time-gate prior is duplicated byte-for-byte from
    MultiScalePointSampler rather than shared, so toggling the sampling
    strategy never also changes the time-gating behavior: how each scale
    is sampled and how much each scale is weighted stay independently
    ablatable.

    Per-scale search radius is derived at the first forward() call from
    each map's actual resolution, as radius_cells / (H - 1) in the [-1,1]
    coordinate space the points live in -- "N grid cells" then means the
    same relative thing at every resolution, and no encoder-specific
    config is needed at construction time.

    deform_n_samples: total reads per point per scale, including the
        aligned (zero-offset) sample itself.
    deform_min_scale_res: scales with resolution >= this use deformable
        multi-sample reads; below it, a single aligned grid_sample is used
        (matches MultiScalePointSampler for that scale), since very coarse
        maps have little sub-grid neighborhood to exploit.
    radius_cells: search radius, in grid cells at each scale's own
        resolution.
    """

    def __init__(self, scale_channels, proj_dim=64, out_dim=128, time_dim=128,
                 offset_scale: float = 0.15, timesteps: int = 1000,
                 deform_n_samples: int = 4, deform_min_scale_res: int = 28,
                 radius_cells: float = 2.5, coord_fourier_bands: int = 4,
                 top_k_scales: int = None):
        super().__init__()
        self.n_scales = len(scale_channels)
        self.offset_scale = offset_scale
        # See MultiScalePointSampler for why this clamp is necessary.
        if top_k_scales is not None:
            top_k_scales = min(int(top_k_scales), self.n_scales)
        self.top_k_scales = top_k_scales
        self.register_buffer("_t_max", torch.tensor(float(max(timesteps - 1, 1))))

        self.deform_n_samples = max(int(deform_n_samples), 1)
        self.deform_min_scale_res = int(deform_min_scale_res)
        self.radius_cells = float(radius_cells)
        self._deform_mask = None
        self._scale_radius = None

        # Bootstrap token (Fourier(xy) + t_vec) drives only the offset/weight
        # heads below; it is self-contained and shares no weights with
        # ContourDenoiser's own coord_mlp/pos_emb.
        self.token_ff = FourierFeatures(in_dim=2, num_bands=coord_fourier_bands)
        self.token_mlp = nn.Sequential(
            nn.Linear(self.token_ff.out_dim + time_dim, time_dim), nn.GELU(),
            nn.Linear(time_dim, time_dim),
        )

        self.scale_offset = nn.Linear(time_dim, self.n_scales)
        nn.init.zeros_(self.scale_offset.weight)
        nn.init.zeros_(self.scale_offset.bias)

        n_learned = max(self.deform_n_samples - 1, 0)
        self.offset_heads = nn.ModuleList(nn.Linear(time_dim, n_learned * 2) for _ in scale_channels)
        self.weight_heads = nn.ModuleList(nn.Linear(time_dim, self.deform_n_samples) for _ in scale_channels)
        for oh, wh in zip(self.offset_heads, self.weight_heads):
            # Zero-init: starts at "average a few samples right around the
            # point" rather than a random jump.
            nn.init.zeros_(oh.weight); nn.init.zeros_(oh.bias)
            nn.init.zeros_(wh.weight); nn.init.zeros_(wh.bias)

        self.projs = nn.ModuleList(nn.Linear(c, proj_dim) for c in scale_channels)

        effective_scales = self.top_k_scales if self.top_k_scales is not None else self.n_scales
        self.mlp = nn.Sequential(
            nn.Linear(effective_scales * proj_dim + time_dim, out_dim * 2), nn.GELU(),
            nn.Linear(out_dim * 2, out_dim),
        )

    def _cosine_prior(self, t: torch.Tensor) -> torch.Tensor:
        progress = (t.float() / self._t_max).clamp(0.0, 1.0)
        if self.n_scales == 1:
            return torch.ones_like(progress).unsqueeze(-1)
        centers = torch.linspace(0.0, 1.0, self.n_scales, device=t.device)
        width = 1.0 / max(self.n_scales - 1, 1)
        dist = (progress.unsqueeze(-1) - centers.unsqueeze(0)).abs()
        x = (dist / (2 * width)).clamp(0.0, 1.0)
        return 0.5 * (1.0 + torch.cos(math.pi * x))

    def _resolve_scale_geometry(self, maps):
        """Derive per-scale deform-eligibility + radius from actual map
        resolutions on the first call. Assumes square maps."""
        if self._deform_mask is not None:
            return
        resolutions = [m.shape[-1] for m in maps]
        self._deform_mask = [res >= self.deform_min_scale_res for res in resolutions]
        self._scale_radius = [self.radius_cells / max(res - 1, 1) for res in resolutions]

    def forward(self, maps, points, t_vec, t_raw=None, return_weights: bool = False):
        if t_raw is None:
            raise ValueError("DeformableScalePointSampler.forward requires t_raw.")
        self._resolve_scale_geometry(maps)
        b, n, _ = points.shape

        token_in = torch.cat([self.token_ff(points), t_vec.unsqueeze(1).expand(-1, n, -1)], dim=-1)
        tokens = self.token_mlp(token_in)

        prior = self._cosine_prior(t_raw)
        offset_t = self.offset_scale * torch.tanh(self.scale_offset(t_vec))
        w = (prior + offset_t).clamp(0.0, 1.0)

        feats = []
        for i, (proj, m) in enumerate(zip(self.projs, maps)):
            if self._deform_mask[i] and self.deform_n_samples > 1:
                radius_i = self._scale_radius[i]
                learned = self.offset_heads[i](tokens).view(b, n, self.deform_n_samples - 1, 2)
                learned = torch.tanh(learned) * radius_i
                zero = learned.new_zeros(b, n, 1, 2)
                offsets = torch.cat([zero, learned], dim=2)
            else:
                offsets = points.new_zeros(b, n, 1, 2)

            locations = (points.unsqueeze(2) + offsets).clamp(-1.0, 1.0)
            samples = sample_deformable(m, locations)
            samples = proj(samples)

            if offsets.shape[2] > 1:
                weights = torch.softmax(self.weight_heads[i](tokens)[..., :offsets.shape[2]], dim=-1)
                fused = (samples * weights.unsqueeze(-1)).sum(dim=2)
            else:
                fused = samples[:, :, 0, :]

            fused = fused * w[:, i][:, None, None]
            feats.append(fused)

        if self.top_k_scales is not None and self.top_k_scales < self.n_scales:
            gathered = _gather_topk_scale_feats(feats, w, self.top_k_scales)
            cat = gathered.flatten(-2)
        else:
            cat = torch.cat(feats, dim=-1)

        t_exp = t_vec.unsqueeze(1).expand(-1, n, -1)
        out = self.mlp(torch.cat([cat, t_exp], dim=-1))
        if return_weights:
            return out, w
        return out


class AttentionScalePointSampler(nn.Module):
    """Drop-in replacement with per-point, per-scale fixed-neighborhood,
    content-based attention reads.

    Contrast with DeformableScalePointSampler:
      * Candidate locations are fixed (an attn_window x attn_window grid
        around the point's own cell, keeping the deform_n_samples nearest
        cells) rather than learned offsets -- same candidate budget K for
        a fair comparison.
      * Fusion weights come from real scaled-dot-product attention (Query
        from the point/time token, Key/Value from the sampled features),
        not a token-blind softmax -- the model decides how much to trust a
        candidate after seeing its content.

    Everything else (time-gate prior, deform_min_scale_res eligibility,
    interface) is duplicated byte-for-byte from the other two samplers, so
    a three-way ablation isolates exactly one axis: how each scale's local
    evidence is aggregated.
    """

    def __init__(self, scale_channels, proj_dim=64, out_dim=128, time_dim=128,
                 offset_scale: float = 0.15, timesteps: int = 1000,
                 deform_n_samples: int = 4, deform_min_scale_res: int = 28,
                 attn_window: int = 5, coord_fourier_bands: int = 4,
                 n_attn_heads: int = 4, top_k_scales: int = None):
        super().__init__()
        self.n_scales = len(scale_channels)
        self.offset_scale = offset_scale
        # See MultiScalePointSampler for why this clamp is necessary.
        if top_k_scales is not None:
            top_k_scales = min(int(top_k_scales), self.n_scales)
        self.top_k_scales = top_k_scales
        self.register_buffer("_t_max", torch.tensor(float(max(timesteps - 1, 1))))

        self.deform_n_samples = max(int(deform_n_samples), 1)
        self.deform_min_scale_res = int(deform_min_scale_res)
        self.attn_window = int(attn_window)
        assert self.attn_window * self.attn_window >= self.deform_n_samples, (
            f"attn_window={attn_window} ({attn_window**2} candidates) must provide at least "
            f"deform_n_samples={deform_n_samples} candidates"
        )

        self._deform_mask = None
        self._cell_size = None
        self.register_buffer("_grid_offsets", _local_grid_offsets(self.attn_window))

        self.token_ff = FourierFeatures(in_dim=2, num_bands=coord_fourier_bands)
        self.token_mlp = nn.Sequential(
            nn.Linear(self.token_ff.out_dim + time_dim, time_dim), nn.GELU(),
            nn.Linear(time_dim, time_dim),
        )

        self.scale_offset = nn.Linear(time_dim, self.n_scales)
        nn.init.zeros_(self.scale_offset.weight)
        nn.init.zeros_(self.scale_offset.bias)

        self.projs = nn.ModuleList(nn.Linear(c, proj_dim) for c in scale_channels)

        assert proj_dim % n_attn_heads == 0, "proj_dim must be divisible by n_attn_heads"
        self.n_attn_heads = n_attn_heads
        self.head_dim = proj_dim // n_attn_heads
        self.query_heads = nn.ModuleList(nn.Linear(time_dim, proj_dim) for _ in scale_channels)
        self.key_heads = nn.ModuleList(nn.Linear(proj_dim, proj_dim) for _ in scale_channels)
        self.value_heads = nn.ModuleList(nn.Linear(proj_dim, proj_dim) for _ in scale_channels)
        # Zero-init Query: attention starts ~uniform over the K candidates.
        for qh in self.query_heads:
            nn.init.zeros_(qh.weight)
            nn.init.zeros_(qh.bias)

        effective_scales = self.top_k_scales if self.top_k_scales is not None else self.n_scales
        self.mlp = nn.Sequential(
            nn.Linear(effective_scales * proj_dim + time_dim, out_dim * 2), nn.GELU(),
            nn.Linear(out_dim * 2, out_dim),
        )

    def _cosine_prior(self, t: torch.Tensor) -> torch.Tensor:
        progress = (t.float() / self._t_max).clamp(0.0, 1.0)
        if self.n_scales == 1:
            return torch.ones_like(progress).unsqueeze(-1)
        centers = torch.linspace(0.0, 1.0, self.n_scales, device=t.device)
        width = 1.0 / max(self.n_scales - 1, 1)
        dist = (progress.unsqueeze(-1) - centers.unsqueeze(0)).abs()
        x = (dist / (2 * width)).clamp(0.0, 1.0)
        return 0.5 * (1.0 + torch.cos(math.pi * x))

    def _resolve_scale_geometry(self, maps):
        if self._deform_mask is not None:
            return
        resolutions = [m.shape[-1] for m in maps]
        self._deform_mask = [res >= self.deform_min_scale_res for res in resolutions]
        self._cell_size = [2.0 / max(res - 1, 1) for res in resolutions]

    def _nearest_k_offsets(self, device):
        """The deform_n_samples cells nearest to the center, out of the
        fixed window x window grid. Constant, no learnable parameters."""
        grid = self._grid_offsets.to(device)
        dist = grid.pow(2).sum(-1)
        k = min(self.deform_n_samples, grid.shape[0])
        idx = torch.topk(-dist, k=k).indices
        return grid[idx]

    def forward(self, maps, points, t_vec, t_raw=None, return_weights: bool = False):
        if t_raw is None:
            raise ValueError("AttentionScalePointSampler.forward requires t_raw.")
        self._resolve_scale_geometry(maps)
        b, n, _ = points.shape
        device = points.device

        token_in = torch.cat([self.token_ff(points), t_vec.unsqueeze(1).expand(-1, n, -1)], dim=-1)
        tokens = self.token_mlp(token_in)

        prior = self._cosine_prior(t_raw)
        offset_t = self.offset_scale * torch.tanh(self.scale_offset(t_vec))
        w = (prior + offset_t).clamp(0.0, 1.0)

        cell_offsets = self._nearest_k_offsets(device)

        feats = []
        for i, (proj, m) in enumerate(zip(self.projs, maps)):
            if self._deform_mask[i]:
                cell_size = self._cell_size[i]
                # (dy, dx) cell units -> (x, y) coordinate offsets for grid_sample.
                coord_offsets = torch.stack(
                    [cell_offsets[:, 1], cell_offsets[:, 0]], dim=-1
                ) * cell_size
                offsets = coord_offsets.view(1, 1, -1, 2).expand(b, n, -1, -1)
                locations = (points.unsqueeze(2) + offsets).clamp(-1.0, 1.0)

                sampled = sample_deformable(m, locations)
                sampled = proj(sampled)

                q = self.query_heads[i](tokens)
                k_ = self.key_heads[i](sampled)
                v_ = self.value_heads[i](sampled)

                H, D = self.n_attn_heads, self.head_dim
                bb, nn_, kk = sampled.shape[0], sampled.shape[1], sampled.shape[2]
                q = q.view(bb, nn_, H, D)
                k_ = k_.view(bb, nn_, kk, H, D)
                v_ = v_.view(bb, nn_, kk, H, D)

                logits = torch.einsum("bnhd,bnkhd->bnhk", q, k_) / math.sqrt(D)
                attn = torch.softmax(logits, dim=-1)
                fused = torch.einsum("bnhk,bnkhd->bnhd", attn, v_)
                fused = fused.reshape(bb, nn_, H * D)
            else:
                # Coarse scales: single aligned sample, as in the other samplers.
                grid = torch.clamp(points, -1.0, 1.0).unsqueeze(1)
                s = F.grid_sample(m, grid, mode="bilinear",
                                   padding_mode="border", align_corners=True)
                s = s.squeeze(2).transpose(1, 2)
                fused = proj(s)

            fused = fused * w[:, i][:, None, None]
            feats.append(fused)

        if self.top_k_scales is not None and self.top_k_scales < self.n_scales:
            gathered = _gather_topk_scale_feats(feats, w, self.top_k_scales)
            cat = gathered.flatten(-2)
        else:
            cat = torch.cat(feats, dim=-1)

        t_exp = t_vec.unsqueeze(1).expand(-1, n, -1)
        out = self.mlp(torch.cat([cat, t_exp], dim=-1))
        if return_weights:
            return out, w
        return out


class GlobalMemoryCrossAttention(nn.Module):
    """Global cross-attention branch over a compact, spatially tagged
    memory built from the coarsest backbone scales.

    Motivation: the multi-scale point sampler (above, any variant) only
    ever reads at or near the point's current position. It has no way to
    recover from gross mislocalization -- there is nothing that lets a
    point "look elsewhere" in the image. This branch closes that gap: every
    point's token attends over a pooled representation of the whole image,
    gated to be strongest at high t (localization error is largest, coarse
    global structure matters most) and fade toward 0 at low t (the local
    branch is already precise; global re-attention would just add noise).

    The memory is built once per batch (`build_memory`, called with the
    fused per-scale condition maps already used by the point sampler) and
    reused across every call to `forward`. Zero-initialized output
    projection: the branch starts as an exact no-op and is picked up
    gradually during training.
    """
    def __init__(self, scale_channels, hidden_dim: int = 128, num_heads: int = 4,
                 global_levels: int = 2, global_grid: int = 14,
                 coord_fourier_bands: int = 6, gate_schedule: str = "linear",
                 gate_min: float = 0.20, gate_max: float = 1.00):
        super().__init__()
        self.n_scales = len(scale_channels)
        self.global_levels = max(1, min(int(global_levels), self.n_scales))
        self.global_grid = max(int(global_grid), 2)

        self.global_proj = nn.ModuleList([
            nn.Linear(c, hidden_dim) for c in scale_channels[-self.global_levels:]
        ])
        self.level_emb = nn.Parameter(torch.zeros(self.global_levels, hidden_dim))
        nn.init.normal_(self.level_emb, std=0.02)

        self.spatial_ff = FourierFeatures(in_dim=2, num_bands=coord_fourier_bands)
        self.spatial_proj = nn.Linear(self.spatial_ff.out_dim, hidden_dim, bias=False)

        self.cross_norm_q = nn.LayerNorm(hidden_dim)
        self.cross_norm_m = nn.LayerNorm(hidden_dim)
        self.attn = nn.MultiheadAttention(hidden_dim, num_heads, batch_first=True, dropout=0.0)
        self.ff = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim * 2), nn.GELU(),
            nn.Linear(hidden_dim * 2, hidden_dim),
        )
        nn.init.zeros_(self.ff[-1].weight)
        nn.init.zeros_(self.ff[-1].bias)

        if gate_schedule not in ("linear", "quadratic", "cosine", "constant"):
            raise ValueError(f"unknown gate_schedule: {gate_schedule!r}")
        self.gate_schedule = gate_schedule
        self.gate_min = float(gate_min)
        self.gate_max = float(gate_max)

    def _gate(self, t_norm: torch.Tensor) -> torch.Tensor:
        span = self.gate_max - self.gate_min
        if self.gate_schedule == "linear":
            g = t_norm
        elif self.gate_schedule == "quadratic":
            g = t_norm ** 2
        elif self.gate_schedule == "cosine":
            g = 0.5 * (1.0 - torch.cos(math.pi * t_norm))
        else:  # "constant"
            g = torch.ones_like(t_norm)
        return (self.gate_min + span * g).view(-1, 1, 1)

    def build_memory(self, maps) -> torch.Tensor:
        """maps: full list [B,C_s,H,W] (finest..coarsest). Uses only the
        last `global_levels` (coarsest) scales. -> [B, sum(L), hidden_dim]."""
        start = self.n_scales - self.global_levels
        memories = []
        for local_i, level in enumerate(range(start, self.n_scales)):
            m = maps[level]
            h, w = m.shape[-2:]
            if max(h, w) > self.global_grid:
                m = F.adaptive_avg_pool2d(m, (self.global_grid, self.global_grid))
                h, w = m.shape[-2:]
            raw = m.flatten(2).transpose(1, 2)
            feat_tokens = self.global_proj[local_i](raw)

            ys = torch.linspace(-1, 1, h, device=m.device, dtype=m.dtype)
            xs = torch.linspace(-1, 1, w, device=m.device, dtype=m.dtype)
            grid_y, grid_x = torch.meshgrid(ys, xs, indexing="ij")
            coords = torch.stack([grid_x, grid_y], dim=-1).view(1, h * w, 2)
            pe = self.spatial_proj(self.spatial_ff(coords)).expand(m.shape[0], -1, -1)

            tagged = feat_tokens + pe + self.level_emb[local_i].view(1, 1, -1)
            memories.append(tagged)
        return torch.cat(memories, dim=1)

    def forward(self, tokens, memory, t_norm):
        q = self.cross_norm_q(tokens)
        m = self.cross_norm_m(memory)
        update, _ = self.attn(q, m, m, need_weights=False)
        gate = self._gate(t_norm)
        tokens = tokens + gate * update
        tokens = tokens + gate * self.ff(tokens)
        return tokens

def _local_window_mask(n_points, window):
    idx = torch.arange(n_points)
    dist = (idx[:, None] - idx[None, :]).abs()
    dist = torch.minimum(dist, n_points - dist)
    return dist > window  # True = masked out


class ContourDenoiser(nn.Module):
    def __init__(
        self,
        pos_scale: float = 0.15,
        n_points: int = 200,
        hidden_dim: int = 128,
        num_layers: int = 4,
        num_heads: int = 4,
        attn_window: int = 7,
        scale_channels=(64,),
        proj_dim: int = 64,
        coord_fourier_bands: int = 6,
        gate_offset_scale: float = 0.15,
        timesteps: int = 1000,
        predict_residual: bool = True,
        residual_target: str = "input",
        guidance_time_scale: float = 0.0,
        # Multi-scale sampling strategy (mutually exclusive ablation):
        # "input" -> MultiScalePointSampler (1 aligned sample/scale)
        # "deformable" -> DeformableScalePointSampler (learned offsets)
        # "attention" -> AttentionScalePointSampler (fixed neighborhood + attention)
        sampler_type: str = "input",
        use_deformable_sampling: bool = False,  # deprecated, backward compat only
        deform_n_samples: int = 4,
        deform_min_scale_res: int = 28,
        deform_radius_cells: float = 2.5,
        attn_sampler_window: int = 5,
        attn_sampler_heads: int = 4,
        top_k_scales: int = None,
        # Global cross-attention branch (optional, off by default).
        local_drop_coarsest: int = 0,
        use_global_attn: bool = False,
        global_attn_levels: int = 2,
        global_attn_grid: int = 14,
        global_attn_position: str = "pre",
        global_attn_gate_schedule: str = "linear",
        global_attn_gate_min: float = 0.20,
        global_attn_gate_max: float = 1.00,

    ):
        super().__init__()
        self.n_points = n_points
        self.hidden_dim = hidden_dim
        self.predict_residual = predict_residual
        self.guidance_time_scale = float(guidance_time_scale)

        # NEW: the local sampler is built on a TRUNCATED scale list (finest
        # scales only), while self.global_attn below always uses the FULL
        # scale_channels/maps -- these are deliberately different views of
        # the same pyramid, not two independent configs to keep in sync.
        full_scale_channels = list(scale_channels)
        self.local_drop_coarsest = max(int(local_drop_coarsest), 0)
        if self.local_drop_coarsest >= len(full_scale_channels):
            raise ValueError(
                f"local_drop_coarsest={self.local_drop_coarsest} would remove all "
                f"{len(full_scale_channels)} scales from the local sampler."
            )
        local_scale_channels = (
            full_scale_channels[: len(full_scale_channels) - self.local_drop_coarsest]
            if self.local_drop_coarsest > 0 else full_scale_channels
        )

        # Backward compat: pre-`sampler_type` configs only set the boolean
        # `use_deformable_sampling` flag.
        if sampler_type == "input" and use_deformable_sampling:
            sampler_type = "deformable"
        if sampler_type not in ("input", "deformable", "attention"):
            raise ValueError(f"unknown sampler_type: {sampler_type!r}")
        self.sampler_type = sampler_type
        self.use_deformable_sampling = (sampler_type == "deformable")

        if residual_target not in ("input", "zero", "none"):
            raise ValueError(f"unknown residual_target: {residual_target!r}")
        if not predict_residual and residual_target == "input":
            residual_target = "none"
        self.residual_target = residual_target

        self.time_mlp = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim), nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
        )

        if self.sampler_type == "deformable":
            self.sampler = DeformableScalePointSampler(
                local_scale_channels, proj_dim=proj_dim, out_dim=hidden_dim, time_dim=hidden_dim,
                offset_scale=gate_offset_scale, timesteps=timesteps,
                deform_n_samples=deform_n_samples,
                deform_min_scale_res=deform_min_scale_res,
                radius_cells=deform_radius_cells,
                top_k_scales=top_k_scales,
            )
        elif self.sampler_type == "attention":
            self.sampler = AttentionScalePointSampler(
                local_scale_channels, proj_dim=proj_dim, out_dim=hidden_dim, time_dim=hidden_dim,
                offset_scale=gate_offset_scale, timesteps=timesteps,
                deform_n_samples=deform_n_samples,
                deform_min_scale_res=deform_min_scale_res,
                attn_window=attn_sampler_window,
                n_attn_heads=attn_sampler_heads,
                top_k_scales=top_k_scales,
            )
        else:
            self.sampler = MultiScalePointSampler(
                local_scale_channels, proj_dim=proj_dim, out_dim=hidden_dim, time_dim=hidden_dim,
                offset_scale=gate_offset_scale, timesteps=timesteps,
                top_k_scales=top_k_scales,
            )

        self.use_global_attn = bool(use_global_attn)
        if global_attn_position not in ("pre", "post", "both"):
            raise ValueError(f"unknown global_attn_position: {global_attn_position!r}")
        self.global_attn_position = global_attn_position
        self.global_attn = GlobalMemoryCrossAttention(
            full_scale_channels, hidden_dim=hidden_dim, num_heads=num_heads,
            global_levels=global_attn_levels, global_grid=global_attn_grid,
            coord_fourier_bands=coord_fourier_bands,
            gate_schedule=global_attn_gate_schedule,
            gate_min=global_attn_gate_min, gate_max=global_attn_gate_max,
        ) if self.use_global_attn else None

        # NeRF-style Fourier embedding of point coordinates (additive path only).
        self.coord_ff = FourierFeatures(in_dim=2, num_bands=coord_fourier_bands)
        self.coord_mlp = nn.Sequential(
            nn.Linear(self.coord_ff.out_dim, hidden_dim), nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
        )

        # Fixed sinusoidal positional embedding over the ordered point index.
        pe = torch.zeros(1, n_points, hidden_dim)
        position = torch.arange(0, n_points, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, hidden_dim, 2).float() * (-math.log(10000.0) / hidden_dim))
        pe[0, :, 0::2] = torch.sin(position * div_term)
        pe[0, :, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pos_emb", pe)

        self.local_conv1 = nn.Conv1d(hidden_dim, hidden_dim, 3, padding=1, padding_mode="circular")

        self.attn_window = attn_window
        mask = _local_window_mask(n_points, attn_window)
        self.register_buffer("attn_mask", mask)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim, nhead=num_heads, dim_feedforward=hidden_dim * 2, batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.local_conv2 = nn.Conv1d(hidden_dim, hidden_dim, 3, padding=1, padding_mode="circular")

        self.curve_proj = nn.Sequential(
            nn.Linear(1, hidden_dim // 4), nn.GELU(),
            nn.Linear(hidden_dim // 4, hidden_dim)
        )

        # Prediction head, skip-connected with guidance and raw coordinates.
        self.out_mlp = nn.Sequential(
            nn.Linear(hidden_dim + hidden_dim + 2, hidden_dim // 2), nn.GELU(),
            nn.Linear(hidden_dim // 2, 2),
        )
        # Zero-init in both residual modes: "input" starts at the identity
        # on the noisy input; "zero" starts at exactly the proposal (V5).
        # "none" (full x0 prediction) has no such target to start at.
        if self.residual_target in ("input", "zero"):
            nn.init.zeros_(self.out_mlp[-1].weight)
            nn.init.zeros_(self.out_mlp[-1].bias)

        self.register_buffer("pos_scale", torch.tensor(float(pos_scale)))

    def forward(self, points, t, cond_maps, return_scale_weights: bool = False):
        t_vec = self.time_mlp(timestep_embedding(t, self.hidden_dim))
        t_emb = t_vec.unsqueeze(1)

        # NEW: the local sampler only ever sees the truncated (finer) subset
        # of cond_maps; the global branch (below) always uses the full list.
        if self.local_drop_coarsest > 0:
            local_maps = cond_maps[: len(cond_maps) - self.local_drop_coarsest]
        else:
            local_maps = cond_maps

        if return_scale_weights:
            guidance, scale_w = self.sampler(local_maps, points, t_vec, t_raw=t, return_weights=True)
        else:
            guidance = self.sampler(local_maps, points, t_vec, t_raw=t)

        if self.guidance_time_scale > 0:
            t_frac = (t.float() / max(self.sampler._t_max.item(), 1.0)).clamp(0.0, 1.0)
            g_scale = 1.0 - self.guidance_time_scale * t_frac
            guidance = guidance * g_scale.view(-1, 1, 1)

        coord_pe = self.coord_ff(points)

        prev_p = torch.roll(points, shifts=1, dims=1)
        next_p = torch.roll(points, shifts=-1, dims=1)
        current_curve = torch.norm(next_p + prev_p - 2 * points, dim=-1, keepdim=True)
        curve_bias = self.curve_proj(current_curve)

        x = (self.coord_mlp(coord_pe) + guidance + self.pos_scale * self.pos_emb
             + t_emb + curve_bias)

        if self.use_global_attn:
            memory = self.global_attn.build_memory(cond_maps)  # full pyramid, unchanged
            t_norm = (t.float() / max(self.sampler._t_max.item(), 1.0)).clamp(0.0, 1.0)
        else:
            memory = None
            t_norm = None

        if self.use_global_attn and self.global_attn_position in ("pre", "both"):
            x = self.global_attn(x, memory, t_norm)

        x = x.transpose(1, 2)
        x = F.gelu(self.local_conv1(x))
        x = x.transpose(1, 2)

        x = self.transformer(x, mask=self.attn_mask)
        x = x + guidance

        if self.use_global_attn and self.global_attn_position in ("post", "both"):
            x = self.global_attn(x, memory, t_norm)

        x = x.transpose(1, 2)
        x = F.gelu(self.local_conv2(x))
        x = x.transpose(1, 2)

        head_out = self.out_mlp(torch.cat([x, guidance, points], dim=-1))

        if self.residual_target == "input":
            out = points + head_out
        elif self.residual_target == "zero":
            out = head_out
        else:  # "none"
            out = head_out

        if return_scale_weights:
            return out, scale_w
        return out