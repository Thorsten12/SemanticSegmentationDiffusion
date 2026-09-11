"""Boundary-point denoiser with per-point, time-gated multi-scale conditioning.

Given the current (noisy) point set, the diffusion timestep, and the *raw
multi-scale* feature pyramid from the image backbone, predict the clean points x0.

Conditioning (the key design):
  * For each point, `grid_sample` (bilinear-interpolate) EVERY backbone scale at
    that point's (x, y). Coarse scales (e.g. 7x7) give global context; fine scales
    (e.g. 56x56) give precise local cues. No up-front fusion to a single map.
  * The per-scale weighting is now a FORCED cosine schedule over the timestep:
    coarse scales dominate near pure noise (t close to T), fine scales dominate
    near the end of sampling (t close to 0), with a smooth cosine ramp between.
    A small *learned* offset (bounded to +-0.15 via tanh) is added on top so the
    model can nudge the schedule per-point/per-batch, but cannot override it.
    (Empirically, a fully free `sigmoid(Linear(t_vec))` gate collapsed to a flat,
    time-independent ~0.45-0.65 band across all scales and never specialized -
    see training logs. Forcing the prior fixes that while still letting the
    network learn a bounded correction.)
  * The weighted per-scale features are concatenated and an MLP (also fed the
    timestep) summarizes them to one guidance vector per point.

--- multi-scale sampling strategies (ablation-toggleable, mutually exclusive) --
`MultiScalePointSampler` reads exactly ONE location per scale per point (the
point's own coordinate). This means a point's conditioning signal at a given
scale is a single bilinear sample -- no ability to look around at neighboring
image evidence before deciding what's there.

Two interface-compatible drop-in replacements exist, selected via
`sampler_type`:

  * `DeformableScalePointSampler` (`sampler_type="deformable"`): reads
    `deform_n_samples` locations per point per scale -- one aligned sample at
    the point itself plus `deform_n_samples - 1` *learned offset* samples (a
    small head predicts WHERE else, within a per-scale radius, to look),
    softmax-weighted and summed. Both the offsets AND the fusion weights are
    predicted purely from the point/time token -- the model never sees the
    sampled content before deciding where to look or how much to trust each
    sample. Lightweight, single-scale-per-call analogue of Deformable
    DETR/Deformable-Conv-style sampling.

  * `AttentionScalePointSampler` (`sampler_type="attention"`): reads a FIXED
    local grid neighborhood per point per scale (the `deform_n_samples`
    nearest cells in an `attn_window` x `attn_window` grid around the point's
    own cell -- no learned offsets), then fuses them with real content-based
    scaled-dot-product attention: Query from the point/time token, Key/Value
    from the actually-sampled feature values. Unlike the deformable sampler,
    the fusion weights here DO depend on what is actually seen at each
    candidate location, not just on the token. Same candidate BUDGET
    (`deform_n_samples`) as the deformable sampler for a fair comparison; only
    the aggregation mechanism (learned-offset + token-blind softmax vs.
    fixed-neighborhood + content-based attention) differs.

Motivation (both): on low-contrast or geometrically complex (non-elliptical)
boundaries, a single point-sample is a noisy, easily-ambiguous signal; being
able to gather corroborating (or disambiguating) local evidence around each
point gives the model a chance to disambiguate before committing to a
position. Both are ablations, not replacements: `sampler_type="input"`
(default) gives byte-for-byte the old `MultiScalePointSampler` behavior.

--- top_k_scales (scale-count reduction ablation) ----------------------------
All three samplers accept an optional `top_k_scales: int`. When set (and
smaller than the encoder's actual scale count), the sampler still computes
EVERY scale's features in full (grid_sample / deformable sampling / content
attention -- no compute is skipped), but ONLY the `top_k_scales` scales with
the strongest time-gate weight (per sample in the batch) are gathered into
the final fusion MLP's input -- the rest are simply dropped, not
zero-padded. This is a genuine dimensionality reduction of the fusion MLP
(`Linear(top_k_scales * proj_dim + time_dim, ...)` instead of
`Linear(n_scales * proj_dim + time_dim, ...)`), not just a masked-out
subset of an otherwise unchanged architecture -- so a checkpoint trained
with one `top_k_scales` value is NOT directly compatible with a different
one (or with `None`), see `_gather_topk_scale_feats`'s docstring for why
this is still safe to do (t_vec, fed into the same MLP call, lets the model
recover which absolute scale indices ended up in which slot, since t
determines the top-k selection almost deterministically).

Sequence model (closed ordered contour of N points):
  sinusoidal timestep embedding + fixed positional embedding over point index;
  circular Conv1d (the contour wraps around) -> Transformer encoder -> Conv1d;
  MLP head -> 2D coordinate prediction. Coordinates enter only via an additive
  Fourier path (kept OUT of the content fusion to avoid memorizing tiny datasets).

--- residual_target (V5 fix) -------------------------------------------------
`predict_residual` used to mean exactly one thing: "the head's MLP output is
a delta added to the noisy input points" (`out = points + head_out`), zero-
initialized so training starts at (approximately) the identity on the noisy
input. That is the right behavior when there is NO external initial guess.

Once a deterministic coarse-contour proposal is introduced (V5,
`proposal_target="residual"` in diffusion.py / train.py), the *caller* adds
the proposal on top of whatever this module returns:
    x0 = proposal + denoiser_out
In that regime, `out = points + head_out` is wrong: `points` is the noisy,
~N(0,1)-scale diffusion sample, not a meaningful zero-point to add the
proposal to. The correct zero-init behavior in that regime is
`out = head_out` alone (pure delta, exactly 0 at init), so that
`x0 = proposal + 0 = proposal` at the start of training and the model only
ever has to learn the *local* correction on top of the proposal -- never has
to re-derive global position/size/coarse-shape from noise itself.

`residual_target` makes this explicit instead of overloading
`predict_residual`:
  - "input": legacy behavior, out = points + head_out (zero-init).
  - "zero":  V5 proposal behavior, out = head_out (zero-init); caller adds
             the proposal.
  - "none":  out = head_out, NOT zero-init (full x0 prediction, no residual
             of any kind). Provided for completeness / ablations; not used
             by default anywhere in this codebase.
"""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from .positional import FourierFeatures


def timestep_embedding(t: torch.Tensor, dim: int, max_period: int = 10000) -> torch.Tensor:
    """Standard sinusoidal embedding of integer timesteps -> [B, dim]."""
    half = dim // 2
    freqs = torch.exp(
        -math.log(max_period) * torch.arange(half, device=t.device, dtype=torch.float32) / half
    )
    args = t.float()[:, None] * freqs[None]
    emb = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
    if dim % 2:  # pad odd dims
        emb = F.pad(emb, (0, 1))
    return emb


def sample_deformable(feat_map: torch.Tensor, locations: torch.Tensor) -> torch.Tensor:
    """Bilinear sample a [B,C,H,W] map at [B,N,K,2] locations (range [-1,1]).

    Same pattern as BoundarySnapper's `sample_at_points` (kept as a free
    function here, not imported cross-module, to avoid a snapper<->denoiser
    dependency). Returns [B,N,K,C].
    """
    b, n, k, _ = locations.shape
    grid = locations.view(b, n * k, 1, 2)
    sampled = F.grid_sample(feat_map, grid, mode="bilinear",
                             padding_mode="border", align_corners=True)  # [B,C,N*K,1]
    sampled = sampled.squeeze(-1).transpose(1, 2)                       # [B,N*K,C]
    return sampled.view(b, n, k, -1)


def _local_grid_offsets(window: int) -> torch.Tensor:
    """[(window*window), 2] grid of integer (dy, dx) shifts centered at
    (0, 0), e.g. window=5 -> shifts from -2..+2 along both axes (25
    candidates)."""
    r = window // 2
    ys, xs = torch.meshgrid(
        torch.arange(-r, r + 1), torch.arange(-r, r + 1), indexing="ij"
    )
    return torch.stack([ys.reshape(-1), xs.reshape(-1)], dim=-1).float()  # [W*W, 2]


def _gather_topk_scale_feats(feats: list, w: torch.Tensor, k: int) -> torch.Tensor:
    """feats: Liste von S Tensoren [B,N,proj_dim] (bereits zeitgewichtet).
    w: [B,S] Scale-Gate-Gewichte (bestimmt, welche k Scales pro Sample aktiv sind).
    Gibt [B,N,k,proj_dim] zurueck -- die k aktiven Scale-Features PRO SAMPLE,
    sortiert nach Scale-Index (aufsteigend, NICHT nach Gewichtsstaerke), damit
    Slot-Position <-> relative Scale-Reihenfolge konsistent bleibt. t_vec (das
    ebenfalls in den finalen MLP-Input eingeht) verraet dem Modell indirekt,
    welche absoluten Scale-Indizes gerade in welchem Slot stecken, da t die
    Top-k-Auswahl (fast) deterministisch bestimmt.

    Alle S Scales werden weiterhin voll berechnet (kein Compute-Skip) --
    nur die finale Fusion/das MLP sehen konstant k statt S Bloecke."""
    _, topk_idx = torch.topk(w, k=k, dim=-1)             # [B, k], unsortiert nach Gewicht
    topk_idx, _ = torch.sort(topk_idx, dim=-1)            # [B, k], nach Scale-Index sortiert

    stacked = torch.stack(feats, dim=-2)                  # [B,N,S,proj_dim]
    b, n, s, d = stacked.shape
    idx_exp = topk_idx.view(b, 1, k, 1).expand(-1, n, -1, d)   # [B,N,k,proj_dim]
    gathered = torch.gather(stacked, dim=2, index=idx_exp)     # [B,N,k,proj_dim]
    return gathered


def _contour_frame(points: torch.Tensor, eps: float = 1e-6):
    """Tangente + Normale für eine geschlossene, geordnete Kontur [B,N,2].
    Lokale Kopie der gleichnamigen Funktion in snapper.py (bewusst nicht
    importiert, um keine Modul-übergreifende Abhängigkeit zu erzeugen --
    gleiches Prinzip wie sample_deformable in dieser Datei)."""
    prev_p = torch.roll(points, shifts=1, dims=1)
    next_p = torch.roll(points, shifts=-1, dims=1)
    tangent = next_p - prev_p
    tangent = tangent / torch.norm(tangent, dim=-1, keepdim=True).clamp(min=eps)
    normal = torch.stack([-tangent[..., 1], tangent[..., 0]], dim=-1)
    return tangent, normal


class MultiScalePointSampler(nn.Module):
    """Per-point query of a multi-scale feature pyramid, gated by the timestep.

    The gate over scales is a FORCED cosine schedule (coarse -> fine as t goes
    T -> 0), plus a small learned offset bounded to +-`offset_scale`. This
    guarantees the intended coarse-early/fine-late specialization instead of
    hoping a fully free gate discovers it (it didn't, in practice - see module
    docstring above).
    """

    def __init__(self, scale_channels, proj_dim=64, out_dim=128, time_dim=128,
                 offset_scale: float = 0.15, timesteps: int = 1000,
                 top_k_scales: int = None):
        super().__init__()
        self.n_scales = len(scale_channels)
        self.offset_scale = offset_scale
        self.top_k_scales = top_k_scales
        # Used to normalize the raw integer timestep to [0, 1] for the prior.
        # (timesteps - 1) so t=0 and t=T-1 map exactly to the schedule's ends.
        self.register_buffer("_t_max", torch.tensor(float(max(timesteps - 1, 1))))

        # Project each scale to a common width so concatenation is balanced.
        self.projs = nn.ModuleList(nn.Linear(c, proj_dim) for c in scale_channels)
        # Small learned correction on top of the forced prior. Same shape as
        # before (time_dim -> n_scales) but its output is squashed through
        # tanh and scaled down, so it can only nudge, never dominate.
        self.scale_offset = nn.Linear(time_dim, self.n_scales)
        # Initialize near-zero so training starts at (very close to) the pure
        # forced prior and only drifts away from it as the offset is learned.
        nn.init.zeros_(self.scale_offset.weight)
        nn.init.zeros_(self.scale_offset.bias)

        # Summarize the concatenated, time-weighted scales (+ time) to out_dim.
        # Input dim is top_k_scales * proj_dim when top_k_scales is set
        # (genuine dimensionality reduction -- see module docstring), else
        # the full n_scales * proj_dim (legacy behavior).
        effective_scales = self.top_k_scales if self.top_k_scales is not None else self.n_scales
        self.mlp = nn.Sequential(
            nn.Linear(effective_scales * proj_dim + time_dim, out_dim * 2), nn.GELU(),
            nn.Linear(out_dim * 2, out_dim),
        )

    def _cosine_prior(self, t: torch.Tensor) -> torch.Tensor:
        """Forced local->global cosine schedule -> [B, n_scales], values in [0, 1].

        At t = T (pure noise): scale 0 (local/fine) ~0,
        last scale (global/coarse) ~1.

        At t = 0 (clean): scale 0 (local/fine) ~1,
        last scale (global/coarse) ~0.

        Scales are ordered local/fine (index 0) -> global/coarse
        (index n_scales - 1).
        """
        progress = (t.float() / self._t_max).clamp(0.0, 1.0)          # [B]
        if self.n_scales == 1:
            return torch.ones_like(progress).unsqueeze(-1)

        # Each scale i gets its own cosine "window" centered along the
        # progress axis, evenly spaced from local/fine (i=0, centered near
        # progress=0) to global/coarse (i=n_scales-1, centered near progress=1 ).
        centers = torch.linspace(0.0, 1.0, self.n_scales, device=t.device)
        width = 1.0 / max(self.n_scales - 1, 1)                              # window half-width-ish
        # cosine bump: 1 at the scale's center, decaying smoothly to 0 by
        # +-2*width, clamped so distant scales get exactly 0 (not negative).
        dist = (progress.unsqueeze(-1) - centers.unsqueeze(0)).abs()         # [B, S]
        x = (dist / (2 * width)).clamp(0.0, 1.0)
        prior = 0.5 * (1.0 + torch.cos(math.pi * x))                        # [B, S], in [0,1]
        return prior

    def forward(self, maps, points, t_vec, t_raw=None, return_weights: bool = False):
        """maps: list of [B,C_s,H,W]; points [B,N,2]; t_vec [B,time_dim];
        t_raw: integer timesteps [B] (required - used for the forced prior).
        -> [B,N,out_dim] (and optionally the gate weights [B,S])."""
        if t_raw is None:
            raise ValueError("MultiScalePointSampler.forward requires t_raw (integer timesteps) "
                              "to compute the forced local->global prior.")
        n = points.shape[1]
        grid = torch.clamp(points, -1.0, 1.0).unsqueeze(1)        # [B,1,N,2]

        prior = self._cosine_prior(t_raw)                          # [B, S], forced schedule
        offset = self.offset_scale * torch.tanh(self.scale_offset(t_vec))  # [B, S], bounded +-offset_scale
        w = (prior + offset).clamp(0.0, 1.0)                       # [B, S]

        feats = []
        for i, (proj, m) in enumerate(zip(self.projs, maps)):
            s = F.grid_sample(m, grid, mode="bilinear",
                              padding_mode="border", align_corners=True)
            s = s.squeeze(2).transpose(1, 2)                      # [B,N,C_s]
            s = proj(s)                                           # [B,N,proj_dim]
            s = s * w[:, i][:, None, None]                        # time-gate this scale
            feats.append(s)

        if self.top_k_scales is not None and self.top_k_scales < self.n_scales:
            gathered = _gather_topk_scale_feats(feats, w, self.top_k_scales)  # [B,N,k,proj_dim]
            cat = gathered.flatten(-2)                                        # [B,N, k*proj_dim]
        else:
            cat = torch.cat(feats, dim=-1)                            # [B,N, S*proj_dim]

        t_exp = t_vec.unsqueeze(1).expand(-1, n, -1)
        out = self.mlp(torch.cat([cat, t_exp], dim=-1))          # [B,N,out_dim]
        if return_weights:
            return out, w
        return out


class DeformableScalePointSampler(nn.Module):
    """Drop-in replacement for `MultiScalePointSampler` with per-point,
    per-scale DEFORMABLE (multi-location, learned-offset) reads instead of a
    single aligned `grid_sample` per scale.

    Interface-compatible with `MultiScalePointSampler`:
        forward(maps, points, t_vec, t_raw=t, return_weights=bool)
            -> guidance [B,N,out_dim]  (+ scale_w [B,S] if return_weights)

    The forced coarse<->fine time-gate prior (`_cosine_prior`) is copied
    byte-for-byte from `MultiScalePointSampler` rather than shared/inherited,
    so that toggling deformable sampling on/off never also silently changes
    the time-gating behavior -- the two axes (HOW each scale is sampled vs.
    HOW MUCH each scale is weighted per timestep) stay independently
    ablatable.

    Per-scale search radius is derived automatically at the FIRST forward()
    call from each map's actual H (assumed square maps), as
    `radius_cells / (H - 1)` in the same [-1,1] coordinate space the points
    live in -- e.g. with `radius_cells=2.5`, a 7x7 map gets a much bigger
    (relative-to-canvas) search neighborhood than a 224x224 map, which is
    the correct behavior: "2.5 grid cells" means the same relative thing at
    every resolution. This is cached after the first call (resolutions don't
    change between calls for a fixed encoder/img_size) rather than passed in
    at construction time, so this class needs no encoder-specific config and
    works unchanged with any backbone (ConvNeXt/PVT/ResNet/Swin/VMamba) or
    stem configuration -- see encoder.py: `feature_channels` already varies
    by backbone/stem, and resolutions are only knowable from the actual
    tensors at runtime, not from a static list.

    deform_n_samples: total reads per point per scale, INCLUDING the aligned
        (zero-offset) sample at the point itself. Default 4 (1 aligned + 3
        learned). Set to 1 to make this numerically fall back to plain
        aligned sampling (useful as an internal sanity check).
    deform_min_scale_res: scales with resolution >= this use deformable
        multi-sample reads; scales below it fall back to a single aligned
        `grid_sample` (matches `MultiScalePointSampler` exactly for that
        scale), since very coarse maps (7x7, 14x14) have little sub-grid
        neighborhood to exploit and deforming there mostly adds noisy,
        under-determined parameters. Default 28 -> with a ConvNeXt-style
        5-stage pyramid (7/14/28/56/224 for stem+4 stages at img_size=224)
        this deforms on the 28/56/224 stages and keeps 7/14 aligned. Set to
        1 to deform everywhere, or a very high value (e.g. 1e6) to deform
        nowhere (== `MultiScalePointSampler`, useful as a sanity check that
        the new code path is at least not silently broken/harmful when
        reduced to the old behavior).
    radius_cells: deformable search radius expressed as "this many grid
        cells" at each scale's own resolution.
    top_k_scales: see module docstring -- genuine dimensionality reduction
        of the fusion MLP's input, not just a masked subset.
    """

    def __init__(self, scale_channels, proj_dim=64, out_dim=128, time_dim=128,
                 offset_scale: float = 0.15, timesteps: int = 1000,
                 deform_n_samples: int = 4, deform_min_scale_res: int = 28,
                 radius_cells: float = 2.5, coord_fourier_bands: int = 4,
                 top_k_scales: int = None):
        super().__init__()
        self.n_scales = len(scale_channels)
        self.offset_scale = offset_scale
        self.top_k_scales = top_k_scales
        self.register_buffer("_t_max", torch.tensor(float(max(timesteps - 1, 1))))

        self.deform_n_samples = max(int(deform_n_samples), 1)
        self.deform_min_scale_res = int(deform_min_scale_res)
        self.radius_cells = float(radius_cells)
        # Resolved lazily on the first forward() call from actual map shapes
        # (see class docstring). None until then.
        self._deform_mask = None          # list[bool], length n_scales
        self._scale_radius = None         # list[float], length n_scales

        # Cheap bootstrap token: Fourier(point xy) + t_vec -> a small
        # per-point summary used ONLY to drive the offset/weight heads below
        # (not fed forward into the main guidance output directly). This is
        # a small, self-contained embedding -- it does not reuse/share
        # weights with ContourDenoiser's own coord_mlp/pos_emb, so this
        # sampler stays a fully drop-in, independently-testable module.
        self.token_ff = FourierFeatures(in_dim=2, num_bands=coord_fourier_bands)
        self.token_mlp = nn.Sequential(
            nn.Linear(self.token_ff.out_dim + time_dim, time_dim), nn.GELU(),
            nn.Linear(time_dim, time_dim),
        )

        # Same forced-prior learned nudge as MultiScalePointSampler, kept
        # for parity (see class docstring: time-gating stays identical
        # whether or not deformable sampling is enabled).
        self.scale_offset = nn.Linear(time_dim, self.n_scales)
        nn.init.zeros_(self.scale_offset.weight)
        nn.init.zeros_(self.scale_offset.bias)

        n_learned = max(self.deform_n_samples - 1, 0)
        self.offset_heads = nn.ModuleList(nn.Linear(time_dim, n_learned * 2) for _ in scale_channels)
        self.weight_heads = nn.ModuleList(nn.Linear(time_dim, self.deform_n_samples) for _ in scale_channels)
        for oh, wh in zip(self.offset_heads, self.weight_heads):
            # Start aligned: zero learned offsets, uniform sample weights
            # (post-softmax uniform since all logits are 0). Deformable
            # sampling begins numerically close to "average a few samples
            # right around the point", not a random jump, matching the
            # zero-init philosophy used everywhere else in this file.
            nn.init.zeros_(oh.weight); nn.init.zeros_(oh.bias)
            nn.init.zeros_(wh.weight); nn.init.zeros_(wh.bias)

        self.projs = nn.ModuleList(nn.Linear(c, proj_dim) for c in scale_channels)

        effective_scales = self.top_k_scales if self.top_k_scales is not None else self.n_scales
        self.mlp = nn.Sequential(
            nn.Linear(effective_scales * proj_dim + time_dim, out_dim * 2), nn.GELU(),
            nn.Linear(out_dim * 2, out_dim),
        )

    def _cosine_prior(self, t: torch.Tensor) -> torch.Tensor:
        # Byte-for-byte identical to MultiScalePointSampler._cosine_prior --
        # deliberately duplicated (see class docstring).
        progress = (t.float() / self._t_max).clamp(0.0, 1.0)
        if self.n_scales == 1:
            return torch.ones_like(progress).unsqueeze(-1)
        centers = torch.linspace(0.0, 1.0, self.n_scales, device=t.device)
        width = 1.0 / max(self.n_scales - 1, 1)
        dist = (progress.unsqueeze(-1) - centers.unsqueeze(0)).abs()
        x = (dist / (2 * width)).clamp(0.0, 1.0)
        return 0.5 * (1.0 + torch.cos(math.pi * x))

    def _resolve_scale_geometry(self, maps):
        """Lazily derive per-scale deform-eligibility + radius from the
        actual map resolutions on the first call. Assumes square maps
        (H == W), true for every encoder in this codebase."""
        if self._deform_mask is not None:
            return
        resolutions = [m.shape[-1] for m in maps]
        self._deform_mask = [res >= self.deform_min_scale_res for res in resolutions]
        self._scale_radius = [self.radius_cells / max(res - 1, 1) for res in resolutions]

    def forward(self, maps, points, t_vec, t_raw=None, return_weights: bool = False):
        if t_raw is None:
            raise ValueError("DeformableScalePointSampler.forward requires t_raw (integer timesteps) "
                              "to compute the forced local->global prior.")
        self._resolve_scale_geometry(maps)
        b, n, _ = points.shape

        token_in = torch.cat([self.token_ff(points), t_vec.unsqueeze(1).expand(-1, n, -1)], dim=-1)
        tokens = self.token_mlp(token_in)  # [B,N,time_dim]

        prior = self._cosine_prior(t_raw)
        offset_t = self.offset_scale * torch.tanh(self.scale_offset(t_vec))
        w = (prior + offset_t).clamp(0.0, 1.0)  # [B, S]

        feats = []
        for i, (proj, m) in enumerate(zip(self.projs, maps)):
            if self._deform_mask[i] and self.deform_n_samples > 1:
                radius_i = self._scale_radius[i]
                learned = self.offset_heads[i](tokens).view(b, n, self.deform_n_samples - 1, 2)
                learned = torch.tanh(learned) * radius_i
                zero = learned.new_zeros(b, n, 1, 2)
                offsets = torch.cat([zero, learned], dim=2)               # [B,N,K,2]
            else:
                offsets = points.new_zeros(b, n, 1, 2)                    # aligned only

            locations = (points.unsqueeze(2) + offsets).clamp(-1.0, 1.0)  # [B,N,K,2]
            samples = sample_deformable(m, locations)                     # [B,N,K,C]
            samples = proj(samples)                                       # [B,N,K,proj_dim]

            if offsets.shape[2] > 1:
                weights = torch.softmax(self.weight_heads[i](tokens)[..., :offsets.shape[2]], dim=-1)
                fused = (samples * weights.unsqueeze(-1)).sum(dim=2)       # [B,N,proj_dim]
            else:
                fused = samples[:, :, 0, :]

            fused = fused * w[:, i][:, None, None]                        # time-gate this scale
            feats.append(fused)

        if self.top_k_scales is not None and self.top_k_scales < self.n_scales:
            gathered = _gather_topk_scale_feats(feats, w, self.top_k_scales)  # [B,N,k,proj_dim]
            cat = gathered.flatten(-2)                                        # [B,N, k*proj_dim]
        else:
            cat = torch.cat(feats, dim=-1)

        t_exp = t_vec.unsqueeze(1).expand(-1, n, -1)
        out = self.mlp(torch.cat([cat, t_exp], dim=-1))
        if return_weights:
            return out, w
        return out


class AttentionScalePointSampler(nn.Module):
    """Drop-in replacement for `MultiScalePointSampler` /
    `DeformableScalePointSampler` with per-point, per-scale FIXED-neighborhood,
    content-based ATTENTION reads instead of a single aligned `grid_sample`
    (old) or learned-offset softmax fusion (deformable).

    Interface-compatible:
        forward(maps, points, t_vec, t_raw=t, return_weights=bool)
            -> guidance [B,N,out_dim]  (+ scale_w [B,S] if return_weights)

    Contrast with `DeformableScalePointSampler`:
      * Candidate locations are NOT learned/shifted. They are a FIXED local
        grid neighborhood in feature-cell coordinates around the point's own
        cell (an `attn_window` x `attn_window` grid, e.g. 5x5 = 25
        candidates), from which the `deform_n_samples` cells nearest to the
        point's own cell are kept -- same candidate BUDGET K as the
        deformable sampler, for a fair comparison.
      * Fusion weights are NOT predicted blind from the token. They come from
        real scaled-dot-product attention: Query from the point/time token,
        Key/Value from the actually-sampled (and per-scale-projected) feature
        values at those K fixed cells. The model only decides how much to
        trust a candidate AFTER having seen its content.

    Everything else (forced cosine time-gate prior, which scales get a local
    neighborhood at all via `deform_min_scale_res`, module interface) is
    deliberately byte-for-byte duplicated from the other two samplers (not
    shared/inherited), so a three-way ablation isolates exactly one axis:
    HOW each scale's local evidence is aggregated. Toggling between
    "input" / "deformable" / "attention" never silently changes time-gating
    or which scales get any local context at all.
    """

    def __init__(self, scale_channels, proj_dim=64, out_dim=128, time_dim=128,
                 offset_scale: float = 0.15, timesteps: int = 1000,
                 deform_n_samples: int = 4, deform_min_scale_res: int = 28,
                 attn_window: int = 5, coord_fourier_bands: int = 4,
                 n_attn_heads: int = 4, top_k_scales: int = None):
        super().__init__()
        self.n_scales = len(scale_channels)
        self.offset_scale = offset_scale
        self.top_k_scales = top_k_scales
        self.register_buffer("_t_max", torch.tensor(float(max(timesteps - 1, 1))))

        self.deform_n_samples = max(int(deform_n_samples), 1)
        self.deform_min_scale_res = int(deform_min_scale_res)
        self.attn_window = int(attn_window)
        assert self.attn_window * self.attn_window >= self.deform_n_samples, (
            f"attn_window={attn_window} ({attn_window**2} candidates) must provide at least "
            f"deform_n_samples={deform_n_samples} candidates"
        )

        # Lazy, as in DeformableScalePointSampler: resolved on the first
        # forward() call from the actual map resolutions.
        self._deform_mask = None   # list[bool], length n_scales
        self._cell_size = None     # list[float], length n_scales: 2/(H-1) in [-1,1] coords
        # Fixed (NOT learned) local neighborhood in cell units, shared across
        # scales (rescaled to [-1,1] units per-scale at forward time).
        self.register_buffer("_grid_offsets", _local_grid_offsets(self.attn_window))  # [W*W, 2]

        # Same bootstrap token as DeformableScalePointSampler: Fourier(xy) + t_vec.
        self.token_ff = FourierFeatures(in_dim=2, num_bands=coord_fourier_bands)
        self.token_mlp = nn.Sequential(
            nn.Linear(self.token_ff.out_dim + time_dim, time_dim), nn.GELU(),
            nn.Linear(time_dim, time_dim),
        )

        # Same forced time-gate prior + bounded learned nudge as the other two
        # samplers (byte-identical duplication, see class docstring: time-
        # gating stays identical regardless of the sampling strategy).
        self.scale_offset = nn.Linear(time_dim, self.n_scales)
        nn.init.zeros_(self.scale_offset.weight)
        nn.init.zeros_(self.scale_offset.bias)

        self.projs = nn.ModuleList(nn.Linear(c, proj_dim) for c in scale_channels)

        # Real attention per scale: Query from token, Key/Value from the
        # sampled (and projected) local feature values.
        assert proj_dim % n_attn_heads == 0, "proj_dim must be divisible by n_attn_heads"
        self.n_attn_heads = n_attn_heads
        self.head_dim = proj_dim // n_attn_heads
        self.query_heads = nn.ModuleList(nn.Linear(time_dim, proj_dim) for _ in scale_channels)
        self.key_heads = nn.ModuleList(nn.Linear(proj_dim, proj_dim) for _ in scale_channels)
        self.value_heads = nn.ModuleList(nn.Linear(proj_dim, proj_dim) for _ in scale_channels)
        # Zero-init the query projection: at training start, Query ~0 -> all
        # attention logits ~0 -> softmax ~uniform over the K candidates. Same
        # "starts near the plain average" philosophy as
        # DeformableScalePointSampler (there: uniform softmax weights via
        # zero-init of the weight_heads).
        for qh in self.query_heads:
            nn.init.zeros_(qh.weight)
            nn.init.zeros_(qh.bias)

        effective_scales = self.top_k_scales if self.top_k_scales is not None else self.n_scales
        self.mlp = nn.Sequential(
            nn.Linear(effective_scales * proj_dim + time_dim, out_dim * 2), nn.GELU(),
            nn.Linear(out_dim * 2, out_dim),
        )

    def _cosine_prior(self, t: torch.Tensor) -> torch.Tensor:
        # Byte-identical to the other two samplers.
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
        # Cell size in [-1,1] coordinates (same convention as the deformable sampler).
        self._cell_size = [2.0 / max(res - 1, 1) for res in resolutions]

    def _nearest_k_offsets(self, device):
        """Selects the `deform_n_samples` cells nearest to the center out of
        the fixed (window x window) grid. Constant, computed once, no
        learnable parameters -- a pure candidate set."""
        grid = self._grid_offsets.to(device)                     # [W*W, 2]
        dist = grid.pow(2).sum(-1)                                # [W*W]
        k = min(self.deform_n_samples, grid.shape[0])
        idx = torch.topk(-dist, k=k).indices                      # nearest k (incl. center at dist=0)
        return grid[idx]                                           # [K, 2]

    def forward(self, maps, points, t_vec, t_raw=None, return_weights: bool = False):
        if t_raw is None:
            raise ValueError("AttentionScalePointSampler.forward requires t_raw (integer timesteps) "
                              "to compute the forced local->global prior.")
        self._resolve_scale_geometry(maps)
        b, n, _ = points.shape
        device = points.device

        token_in = torch.cat([self.token_ff(points), t_vec.unsqueeze(1).expand(-1, n, -1)], dim=-1)
        tokens = self.token_mlp(token_in)  # [B,N,time_dim]

        prior = self._cosine_prior(t_raw)
        offset_t = self.offset_scale * torch.tanh(self.scale_offset(t_vec))
        w = (prior + offset_t).clamp(0.0, 1.0)  # [B, S]

        # Fixed K nearest cell offsets (once per forward, device-dependent).
        cell_offsets = self._nearest_k_offsets(device)  # [K, 2], units: cells

        feats = []
        for i, (proj, m) in enumerate(zip(self.projs, maps)):
            if self._deform_mask[i]:
                cell_size = self._cell_size[i]
                # Cell offsets (dy, dx) -> [-1,1] coordinate offsets (x, y).
                # grid_sample expects (x, y); our grid is (dy, dx) -> swap and scale.
                coord_offsets = torch.stack(
                    [cell_offsets[:, 1], cell_offsets[:, 0]], dim=-1
                ) * cell_size  # [K, 2]
                offsets = coord_offsets.view(1, 1, -1, 2).expand(b, n, -1, -1)  # [B,N,K,2]
                locations = (points.unsqueeze(2) + offsets).clamp(-1.0, 1.0)   # [B,N,K,2]

                sampled = sample_deformable(m, locations)  # [B,N,K,C]
                sampled = proj(sampled)                     # [B,N,K,proj_dim]

                # Real scaled-dot-product attention: Query from token, K/V
                # from the actually-sampled (content-dependent) values.
                q = self.query_heads[i](tokens)                          # [B,N,proj_dim]
                k_ = self.key_heads[i](sampled)                          # [B,N,K,proj_dim]
                v_ = self.value_heads[i](sampled)                        # [B,N,K,proj_dim]

                H, D = self.n_attn_heads, self.head_dim
                bb, nn_, kk = sampled.shape[0], sampled.shape[1], sampled.shape[2]
                q = q.view(bb, nn_, H, D)                                 # [B,N,H,D]
                k_ = k_.view(bb, nn_, kk, H, D)                           # [B,N,K,H,D]
                v_ = v_.view(bb, nn_, kk, H, D)                           # [B,N,K,H,D]

                logits = torch.einsum("bnhd,bnkhd->bnhk", q, k_) / math.sqrt(D)  # [B,N,H,K]
                attn = torch.softmax(logits, dim=-1)                             # [B,N,H,K]
                fused = torch.einsum("bnhk,bnkhd->bnhd", attn, v_)               # [B,N,H,D]
                fused = fused.reshape(bb, nn_, H * D)                            # [B,N,proj_dim]
            else:
                # Coarse scales: single aligned sample, exactly as in the
                # other two classes.
                grid = torch.clamp(points, -1.0, 1.0).unsqueeze(1)  # [B,1,N,2]
                s = F.grid_sample(m, grid, mode="bilinear",
                                   padding_mode="border", align_corners=True)
                s = s.squeeze(2).transpose(1, 2)  # [B,N,C_s]
                fused = proj(s)                   # [B,N,proj_dim]

            fused = fused * w[:, i][:, None, None]  # time-gate, as in both other classes
            feats.append(fused)

        if self.top_k_scales is not None and self.top_k_scales < self.n_scales:
            gathered = _gather_topk_scale_feats(feats, w, self.top_k_scales)  # [B,N,k,proj_dim]
            cat = gathered.flatten(-2)                                        # [B,N, k*proj_dim]
        else:
            cat = torch.cat(feats, dim=-1)

        t_exp = t_vec.unsqueeze(1).expand(-1, n, -1)
        out = self.mlp(torch.cat([cat, t_exp], dim=-1))
        if return_weights:
            return out, w
        return out


def _local_window_mask(n_points, window):
    idx = torch.arange(n_points)
    dist = (idx[:, None] - idx[None, :]).abs()
    dist = torch.minimum(dist, n_points - dist)
    return dist > window  # True = wird maskiert (geblockt)


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
        # ----- multi-scale sampling strategy (mutually exclusive ablation) -----
        # "input"       -> MultiScalePointSampler (legacy: 1 aligned sample/scale)
        # "deformable"   -> DeformableScalePointSampler (learned offsets + token-blind softmax)
        # "attention"    -> AttentionScalePointSampler (fixed neighborhood + content-based attention)
        sampler_type: str = "input",
        use_deformable_sampling: bool = False,  # deprecated, kept for backward compat (see below)
        deform_n_samples: int = 4,
        deform_min_scale_res: int = 28,
        deform_radius_cells: float = 2.5,
        attn_sampler_window: int = 5,
        attn_sampler_heads: int = 4,
        top_k_scales: int = None,
    ):
        super().__init__()
        self.n_points = n_points
        self.hidden_dim = hidden_dim
        self.predict_residual = predict_residual
        self.guidance_time_scale = float(guidance_time_scale)

        # Backward compat: older configs/checkpoints only know the boolean
        # `use_deformable_sampling` flag from before `sampler_type` existed.
        # If the caller still only sets that (and leaves sampler_type at its
        # default "input"), honor it so existing configs/scripts don't break.
        if sampler_type == "input" and use_deformable_sampling:
            sampler_type = "deformable"
        if sampler_type not in ("input", "deformable", "attention"):
            raise ValueError(f"unknown sampler_type: {sampler_type!r}")
        self.sampler_type = sampler_type
        # Kept as an attribute (mirrors old configs/logging that read this
        # flag directly) but no longer branches any logic below -- sampler_type is authoritative.
        self.use_deformable_sampling = (sampler_type == "deformable")

        if residual_target not in ("input", "zero", "none"):
            raise ValueError(f"unknown residual_target: {residual_target!r}")
        # Backward-compat: if a caller still only sets predict_residual and
        # never touches residual_target, fall back to the legacy meaning.
        if not predict_residual and residual_target == "input":
            residual_target = "none"
        self.residual_target = residual_target

        # Timestep embedding MLP -> [B, hidden].
        self.time_mlp = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim), nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
        )

        # Per-point, time-gated multi-scale guidance. `sampler_type` picks
        # between the three interface-compatible samplers defined above;
        # nothing else in this class needs to know or care which one is
        # active (see all three classes' forward() signatures).
        if self.sampler_type == "deformable":
            self.sampler = DeformableScalePointSampler(
                list(scale_channels), proj_dim=proj_dim, out_dim=hidden_dim, time_dim=hidden_dim,
                offset_scale=gate_offset_scale, timesteps=timesteps,
                deform_n_samples=deform_n_samples,
                deform_min_scale_res=deform_min_scale_res,
                radius_cells=deform_radius_cells,
                top_k_scales=top_k_scales,
            )
        elif self.sampler_type == "attention":
            self.sampler = AttentionScalePointSampler(
                list(scale_channels), proj_dim=proj_dim, out_dim=hidden_dim, time_dim=hidden_dim,
                offset_scale=gate_offset_scale, timesteps=timesteps,
                deform_n_samples=deform_n_samples,
                deform_min_scale_res=deform_min_scale_res,
                attn_window=attn_sampler_window,
                n_attn_heads=attn_sampler_heads,
                top_k_scales=top_k_scales,
            )
        else:
            self.sampler = MultiScalePointSampler(
                list(scale_channels), proj_dim=proj_dim, out_dim=hidden_dim, time_dim=hidden_dim,
                offset_scale=gate_offset_scale, timesteps=timesteps,
                top_k_scales=top_k_scales,
            )

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
        # Local smoothing along the closed contour (circular padding).

        self.attn_window = attn_window  # neuer Konstruktor-Parameter, z.B. 15
        mask = _local_window_mask(n_points, attn_window)
        self.register_buffer("attn_mask", mask)  # [N, N] bool

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim, nhead=num_heads, dim_feedforward=hidden_dim * 2, batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.local_conv2 = nn.Conv1d(hidden_dim, hidden_dim, 3, padding=1, padding_mode="circular")

        self.curve_proj = nn.Sequential(
            nn.Linear(1, hidden_dim // 4), nn.GELU(),
            nn.Linear(hidden_dim // 4, hidden_dim)
        )

        # Prediction head (skip-connected with guidance and raw coords).
        self.out_mlp = nn.Sequential(
            nn.Linear(hidden_dim + hidden_dim + 2, hidden_dim // 2), nn.GELU(),
            nn.Linear(hidden_dim // 2, 2),
        )
        # Zero-init the head in BOTH residual modes ("input" and "zero"):
        #   - "input": out = points + head_out -> zero-init means out starts
        #     at the identity on the noisy input (legacy V<5 behavior).
        #   - "zero":  out = head_out alone -> zero-init means out starts at
        #     exactly 0, so the caller's x0 = proposal + out starts at
        #     exactly the proposal (V5 behavior -- this is the actual fix).
        # "none" (full x0 prediction, no residual) is the only case that
        # should NOT be zero-init, since there both the identity and the
        # proposal are ambiguous/absent goals to start from.
        if self.residual_target in ("input", "zero"):
            nn.init.zeros_(self.out_mlp[-1].weight)
            nn.init.zeros_(self.out_mlp[-1].bias)

        self.register_buffer("pos_scale", torch.tensor(float(pos_scale)))

    def forward(self, points, t, cond_maps, return_scale_weights: bool = False):
        t_vec = self.time_mlp(timestep_embedding(t, self.hidden_dim))
        t_emb = t_vec.unsqueeze(1)

        if return_scale_weights:
            guidance, scale_w = self.sampler(cond_maps, points, t_vec, t_raw=t, return_weights=True)
        else:
            guidance = self.sampler(cond_maps, points, t_vec, t_raw=t)

        if self.guidance_time_scale > 0:
            t_frac = (t.float() / max(self.sampler._t_max.item(), 1.0)).clamp(0.0, 1.0)  # 0=clean,1=noise
            g_scale = 1.0 - self.guidance_time_scale * t_frac        # [B]
            guidance = guidance * g_scale.view(-1, 1, 1)

        coord_pe = self.coord_ff(points)

        prev_p = torch.roll(points, shifts=1, dims=1)
        next_p = torch.roll(points, shifts=-1, dims=1)
        current_curve = torch.norm(next_p + prev_p - 2 * points, dim=-1, keepdim=True)
        curve_bias = self.curve_proj(current_curve)

        x = (self.coord_mlp(coord_pe) + guidance + self.pos_scale * self.pos_emb
             + t_emb + curve_bias)

        x = x.transpose(1, 2)
        x = F.gelu(self.local_conv1(x))
        x = x.transpose(1, 2)

        x = self.transformer(x, mask=self.attn_mask)
        x = x + guidance

        x = x.transpose(1, 2)
        x = F.gelu(self.local_conv2(x))
        x = x.transpose(1, 2)

        head_out = self.out_mlp(torch.cat([x, guidance, points], dim=-1))  # [B,N,2]

        if self.residual_target == "input":
            # Legacy: residual on the noisy diffusion input itself.
            out = points + head_out
        elif self.residual_target == "zero":
            # V5 proposal mode: pure delta, exactly 0 at init. The caller
            # (train.py / diffusion.ddim_sample, via apply_proposal_target)
            # adds this to the deterministic proposal to form x0. This is
            # what lets the model start from a sensible global initial guess
            # and only ever has to learn LOCAL corrections on top of it,
            # instead of having to reconstruct global position/size from
            # ~N(0,1) noise the way "input" mode implicitly requires.
            out = head_out
        else:  # "none"
            out = head_out

        if return_scale_weights:
            return out, scale_w
        return out