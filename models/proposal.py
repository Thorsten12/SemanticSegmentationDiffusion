"""Deterministic coarse contour proposal, computed once per forward pass.

V6: the proposal can now be EITHER a single closed contour (legacy V5
behavior, `proposal_type="fourier"` with `n_shapes=1`, or the older
`proposal_type="ellipse"`) OR a UNION of several coarse shapes
(`n_shapes > 1`). The motivation for the union mode: a single Fourier/ellipse
query has to compress "where is the whole lesion" into one smooth closed
curve, which struggles with multi-lobed or disconnected-looking shapes. With
several shapes, each query can specialize on one lobe/region and the union
covers the combined footprint, hopefully giving the proposal head more
freedom to capture global structure before the diffusion residual has to fix
the rest up.

--- Two representations, two different jobs ------------------------------

Every shape (whether ellipse or Fourier) is fundamentally an ordered set of
`n_points` boundary points living on a fixed angular reference frame
(`phase`), exactly like the old single-shape proposal. That representation
is what lets `soft_rasterize` (winding-number formula, needs an ordered
closed polygon) turn a *single* shape into a soft mask.

The UNION of several shapes, however, is not itself expressible as one
winding-number polygon -- a winding number is only defined for a single
ordered contour. So the union is built in two, deliberately different, ways
depending on which loss consumes it:

  1. Soft mask (differentiable, used by `proposal_dice_loss`): rasterize
     each shape to a soft occupancy mask independently
     (`soft_rasterize`, borrowed from `rasterize.py`), then combine with a
     soft-OR: `union = 1 - prod_i(1 - mask_i)`. Fully differentiable w.r.t.
     every shape's points, so the proposal head's shape parameters get a
     gradient straight from the mask-level Dice signal -- this is the ONLY
     path that trains the multi-shape geometry directly.

  2. Point contour (non-differentiable, used by the diffusion residual
     path): the soft union mask is thresholded to a hard {0,1} mask,
     `cv2.findContours` extracts its (possibly-merged) outer boundary, and
     that boundary is resampled to exactly `n_points` equidistant points --
     same output contract as the old single-shape proposal
     (`[B, n_points, 2]`), so `apply_proposal_target` / the denoiser / the
     rest of the diffusion pipeline don't need to know anything changed.

     This step goes through `cv2.findContours`, which is a discrete pixel
     operation with no usable gradient. It is wrapped in `torch.no_grad()`
     and IS NOT a source of gradient for the shape parameters -- exactly
     like a `proposal_target="residual"` V5 proposal, where the point
     contour supplies the initial guess and the *diffusion residual* (not
     the proposal itself) is what gets gradient from the point-space losses
     (`loss_x0`, `loss_nearest`, `loss_boundary`, `loss_biou` in
     diffusion.py). The proposal's OWN gradient, in multi-shape mode, comes
     entirely from `proposal_dice_loss` on the soft union mask (path 1
     above). With `n_shapes == 1` this reduces exactly to the old
     `SpatialFourierContourProposal` / `EllipseProposal` output, and
     `find_and_resample_contour` is skipped entirely (no cv2 round-trip, no
     precision loss) -- see `ContourProposalHead.forward`.

Everything else about this module (memory token construction from the
coarsest encoder scale, fixed phase reference frame, when it's called)
matches the V5 module this replaces; see the project README for context.
"""

import math

import cv2
import numpy as np
import torch
import torch.nn as nn

from ..utils.rasterize import soft_dice_loss, soft_rasterize


# ---------------------------------------------------------------------------
# Single-shape predictors: each maps one attention query -> one closed
# contour of `n_points` points, in the same [B, n_points, 2] / [-1, 1]
# convention. Both are drop-in-compatible so ContourProposalHead can pick
# either at construction time.
# ---------------------------------------------------------------------------


class SpatialFourierContourProposal(nn.Module):
    """Predict a low-frequency closed contour from spatial global memory.

    Unlike a plain ellipse prior, the learned shape query attends to tokens
    that already contain 2-D positional encoding. Four Fourier harmonics need
    only 18 outputs (center + 4 coefficients per harmonic) while supporting
    asymmetric, elongated, mildly concave, and near-border proposals.

    One instance handles ONE shape query; `ContourProposalHead` instantiates
    `n_shapes` independent query tokens (each with its own instance of the
    small `attn`/`mlp` stack below) when predicting a multi-shape union --
    see the "one query token per shape" design in the module-level docstring.
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
        """memory: [B, M, hidden_dim] (already normalized-ready tokens).
        phase: [B, N, 2] unit-circle reference frame. -> [B, N, 2]."""
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


class EllipseContourProposal(nn.Module):
    """Predict a single (possibly rotated) ellipse from spatial global memory.

    Simpler / lower-capacity than the Fourier proposal: only 5 degrees of
    freedom (center x/y, semi-axes a/b, rotation), which the earlier ablation
    (`s1_ellipse` vs. `s1_fourier_baseline`) suggests can generalize better
    on small datasets precisely because it can't overfit high-frequency
    contour wiggles. Same query-attends-to-memory structure as the Fourier
    head so the two are interchangeable at the `ContourProposalHead` level.
    """

    def __init__(self, hidden_dim: int, num_heads: int):
        super().__init__()
        self.query = nn.Parameter(torch.zeros(1, 1, hidden_dim))
        nn.init.normal_(self.query, std=0.02)
        self.query_norm = nn.LayerNorm(hidden_dim)
        self.memory_norm = nn.LayerNorm(hidden_dim)
        self.attn = nn.MultiheadAttention(
            hidden_dim, num_heads, batch_first=True, dropout=0.0,
        )
        # center (2) + semi-axes (2) + rotation (1, via sin/cos = 2 outputs
        # to avoid angle-wraparound discontinuities) = 6 raw outputs.
        self.mlp = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim), nn.GELU(),
            nn.Linear(hidden_dim, 6),
        )
        nn.init.zeros_(self.mlp[-1].weight)
        nn.init.zeros_(self.mlp[-1].bias)

    def forward(self, memory, phase):
        """memory: [B, M, hidden_dim]. phase: [B, N, 2] unit-circle frame
        (only its angle is used; ellipse doesn't need higher harmonics).
        -> [B, N, 2]."""
        b = memory.shape[0]
        q = self.query.expand(b, -1, -1)
        shape_token, _ = self.attn(
            self.query_norm(q), self.memory_norm(memory), self.memory_norm(memory),
            need_weights=False,
        )
        raw = self.mlp((q + shape_token)[:, 0])  # [B, 6]
        center = 0.85 * torch.tanh(raw[:, 0:2])
        # Semi-axes in (0, 0.9]: sigmoid keeps them positive and bounded so
        # the ellipse can't blow up past the [-1,1] canvas from one point.
        axes = 0.05 + 0.85 * torch.sigmoid(raw[:, 2:4])
        # Stable init near a radius-0.35 circle: bias axes toward ~0.35 by
        # relying on the zero-init head (sigmoid(0)=0.5 -> axes ~0.475) is
        # close enough; an exact match isn't necessary since gradient will
        # move it, and zero-init already gives a deterministic, sane start.
        rot_raw = raw[:, 4:6]
        rot_raw = rot_raw / (rot_raw.norm(dim=-1, keepdim=True).clamp(min=1e-6) + 1.0)
        cos_r, sin_r = rot_raw[:, 0:1], rot_raw[:, 1:2]

        theta = torch.atan2(phase[..., 1], phase[..., 0])  # [B, N]
        ct, st = theta.cos(), theta.sin()
        ex = axes[:, 0:1] * ct
        ey = axes[:, 1:2] * st
        # Rotate (ex, ey) by the predicted angle, then shift by center.
        x = cos_r * ex - sin_r * ey
        y = sin_r * ex + cos_r * ey
        pts = torch.stack([x, y], dim=-1) + center[:, None, :]
        return pts.clamp(-0.995, 0.995)


_SHAPE_CLASSES = {
    "fourier": SpatialFourierContourProposal,
    "ellipse": EllipseContourProposal,
}


def _build_shape_module(proposal_type: str, hidden_dim: int, num_heads: int, harmonics: int):
    if proposal_type not in _SHAPE_CLASSES:
        raise ValueError(f"unknown proposal_type: {proposal_type!r} (expected 'fourier' or 'ellipse')")
    if proposal_type == "fourier":
        return SpatialFourierContourProposal(hidden_dim, num_heads, harmonics=harmonics)
    return EllipseContourProposal(hidden_dim, num_heads)


# ---------------------------------------------------------------------------
# Non-differentiable union: soft masks -> hard union -> single resampled
# contour. Only used when n_shapes > 1; skipped entirely for the n_shapes==1
# legacy path (see ContourProposalHead.forward).
# ---------------------------------------------------------------------------


def soft_union_mask(shapes_points, size=64):
    """Soft-OR union of several single-shape soft rasterizations.

    shapes_points: [B, S, N, 2] in [-1, 1] -- S shapes, each an ordered
        closed contour of N points (the winding-number formula in
        `soft_rasterize` requires a single ordered contour per call, hence
        the loop over S rather than one combined polygon).
    returns: [B, size, size] soft occupancy in [0, 1].

    Soft-OR: union = 1 - prod_i(1 - mask_i). Differentiable w.r.t. every
    shape's points. Note this can saturate towards 1 with many/large
    overlapping shapes (each factor (1-mask_i) is <1 in the interior, so the
    product shrinks multiplicatively) -- acceptable here since `n_shapes` is
    a small fixed CLI constant (typically 2-4), not something this trains to
    grow unboundedly.
    """
    b, s, n, _ = shapes_points.shape
    complement_prod = None
    for i in range(s):
        mask_i = soft_rasterize(shapes_points[:, i], size=size)  # [B, size, size]
        term = 1.0 - mask_i
        complement_prod = term if complement_prod is None else complement_prod * term
    return 1.0 - complement_prod


@torch.no_grad()
def _find_and_resample_contour(hard_mask_np, n_points):
    """cv2.findContours on a single hard {0,1} HxW mask -> resampled [n_points, 2] in [-1, 1].

    Non-differentiable by construction (pixel-discrete contour extraction).
    If the union is empty or degenerate (no contour found, e.g. all-zero
    mask early in training), falls back to a small centered circle so
    downstream shape handling (fixed n_points, no NaNs) never breaks.

    hard_mask_np: uint8 HxW array in {0, 1}.
    returns: float32 [n_points, 2] array in [-1, 1], (x, y) order.
    """
    h, w = hard_mask_np.shape
    contours, _ = cv2.findContours(hard_mask_np, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    if not contours:
        theta = np.linspace(0, 2 * np.pi, n_points, endpoint=False, dtype=np.float32)
        return np.stack([0.35 * np.cos(theta), 0.35 * np.sin(theta)], axis=1)

    # Multiple disjoint components (union of non-overlapping shapes): pick
    # the largest by area, matching the "single closed contour" output
    # contract the diffusion path expects. (A true multi-component point
    # output would change `n_points`' meaning for the rest of the pipeline,
    # which is out of scope here -- the soft mask, not this point contour,
    # is what actually carries the full multi-shape signal into training via
    # proposal_dice_loss.)
    contour = max(contours, key=cv2.contourArea).squeeze(1)  # [K, 2] in pixel (x, y)
    if contour.ndim != 2 or contour.shape[0] < 3:
        theta = np.linspace(0, 2 * np.pi, n_points, endpoint=False, dtype=np.float32)
        return np.stack([0.35 * np.cos(theta), 0.35 * np.sin(theta)], axis=1)

    # Arclength-uniform resampling to exactly n_points, closed contour.
    pts = contour.astype(np.float64)
    pts = np.concatenate([pts, pts[:1]], axis=0)  # close the loop
    seg = np.diff(pts, axis=0)
    seg_len = np.sqrt((seg ** 2).sum(axis=1))
    cumlen = np.concatenate([[0.0], np.cumsum(seg_len)])
    total = cumlen[-1]
    if total < 1e-6:
        theta = np.linspace(0, 2 * np.pi, n_points, endpoint=False, dtype=np.float32)
        return np.stack([0.35 * np.cos(theta), 0.35 * np.sin(theta)], axis=1)

    targets = np.linspace(0.0, total, n_points, endpoint=False)
    idx = np.searchsorted(cumlen, targets, side="right") - 1
    idx = np.clip(idx, 0, len(seg_len) - 1)
    frac = (targets - cumlen[idx]) / np.clip(seg_len[idx], 1e-6, None)
    resampled = pts[idx] + frac[:, None] * seg[idx]  # [n_points, 2] pixel (x, y)

    # Pixel -> [-1, 1], inverse of points_to_mask's mapping.
    x = resampled[:, 0] / max(w - 1, 1) * 2.0 - 1.0
    y = resampled[:, 1] / max(h - 1, 1) * 2.0 - 1.0
    return np.stack([x, y], axis=1).astype(np.float32)


@torch.no_grad()
def union_mask_to_points(union_mask, n_points, threshold=0.5):
    """Batched wrapper: soft union mask [B, size, size] -> hard threshold ->
    cv2.findContours per-sample -> [B, n_points, 2] tensor in [-1, 1].

    Entirely no_grad (see module docstring: this is the point-contour path,
    which trains the DENOISER's residual, not the proposal's own shape
    parameters -- those get gradient from `proposal_dice_loss` on the soft
    union mask instead).
    """
    device, dtype = union_mask.device, union_mask.dtype
    hard = (union_mask.detach().cpu().numpy() > threshold).astype(np.uint8)  # [B, size, size]
    b = hard.shape[0]
    out = np.zeros((b, n_points, 2), dtype=np.float32)
    for i in range(b):
        out[i] = _find_and_resample_contour(hard[i], n_points)
    return torch.from_numpy(out).to(device=device, dtype=dtype)


# ---------------------------------------------------------------------------
# Public head: wraps N shape predictors (+ union machinery when N > 1).
# ---------------------------------------------------------------------------


class ContourProposalHead(nn.Module):
    """Feeds one or more shape predictors from the raw encoder pyramid.

    Call once per forward pass (not per diffusion step): `proposal(raw)` ->
    [B, n_points, 2] in the same [-1, 1] coordinate space as the diffusion
    points (the point-contour output contract is unchanged from V5,
    regardless of `proposal_type` / `n_shapes`).

    proposal_type: "fourier" (default, matches old V5 behavior when
        n_shapes==1) or "ellipse".
    n_shapes: number of independent shape queries (each its own attention
        query + small MLP head, per the module docstring's "one query token
        per shape" design). n_shapes==1 reduces exactly to the old
        single-shape module (no union / no cv2 round-trip at all).
    union_raster_size: resolution used for BOTH the soft union mask (feeds
        proposal_dice_loss) and the hard union mask (feeds the cv2 contour
        extraction for n_shapes>1). Independent of soft_dice_size used
        elsewhere; kept as its own knob since the two rasterizations have
        different jobs (training signal vs. sampling-time initial guess).
    """

    def __init__(self, coarsest_channels: int, n_points: int, hidden_dim: int = 128,
                 num_heads: int = 4, harmonics: int = 4,
                 proposal_type: str = "fourier", n_shapes: int = 1,
                 union_raster_size: int = 64):
        super().__init__()
        self.n_points = n_points
        self.proposal_type = proposal_type
        self.n_shapes = max(1, int(n_shapes))
        self.union_raster_size = int(union_raster_size)
        self.mem_proj = nn.Linear(coarsest_channels, hidden_dim)

        # One independent shape module per query -- "one query token per
        # shape", each with its own attention + MLP (not shared weights),
        # so different shapes can specialize on different regions rather
        # than all collapsing to the same attention pattern.
        self.shapes = nn.ModuleList([
            _build_shape_module(proposal_type, hidden_dim, num_heads, harmonics)
            for _ in range(self.n_shapes)
        ])

        # Fixed, equidistant angles around the unit circle, one per point --
        # the reference frame shared by every shape query. Precomputed since
        # it never depends on the current sample or timestep.
        idx = torch.arange(n_points, dtype=torch.float32)
        theta = 2 * math.pi * idx / n_points
        phase = torch.stack([theta.cos(), theta.sin()], dim=-1)  # [N, 2]
        self.register_buffer("phase", phase.unsqueeze(0))  # [1, N, 2]

    def _memory_tokens(self, raw_maps):
        coarsest = raw_maps[-1]                      # [B, C, H, W]
        b, c, h, w = coarsest.shape
        tokens = coarsest.flatten(2).transpose(1, 2)  # [B, H*W, C]
        return self.mem_proj(tokens)                  # [B, H*W, hidden_dim]

    def forward_shapes(self, raw_maps):
        """Returns the raw per-shape contours [B, S, n_points, 2], BEFORE any
        union step. Useful for visualization/debugging and is what
        `proposal_dice_loss` consumes for the soft-union training signal."""
        tokens = self._memory_tokens(raw_maps)
        b = tokens.shape[0]
        phase = self.phase.expand(b, -1, -1)          # [B, N, 2]
        per_shape = [shape(tokens, phase) for shape in self.shapes]  # each [B, N, 2]
        return torch.stack(per_shape, dim=1)            # [B, S, N, 2]

    def forward(self, raw_maps):
        """raw_maps: list of [B,C_s,H,W] (the encoder's raw pyramid, same list
        passed to `encoder.fuse`). Uses the coarsest (last) scale.

        Returns [B, n_points, 2]: the single point contour handed to the
        diffusion residual path. With n_shapes==1 this is just that one
        shape's own points (no union machinery touched at all -- identical
        cost/behavior to the old V5 single-shape head). With n_shapes>1 it's
        the union's outer boundary, resampled to n_points (see
        `union_mask_to_points`; this step is non-differentiable, by design
        -- see the module docstring).
        """
        shapes_points = self.forward_shapes(raw_maps)          # [B, S, N, 2]
        if self.n_shapes == 1:
            return shapes_points[:, 0]
        union = soft_union_mask(shapes_points, size=self.union_raster_size)
        return union_mask_to_points(union, self.n_points)


def apply_proposal_target(denoiser_out, proposal, proposal_target: str):
    """Combine the denoiser's output with the deterministic proposal.

    `denoiser_out`: whatever the denoiser predicts as x0 candidate, [B,N,2].
    `proposal`: [B,N,2] or None.
    `proposal_target`:
      - "absolute" (default / pre-V5 behavior): denoiser_out IS x0, proposal
        is ignored (pass proposal=None or this mode to skip the proposal head
        entirely -- e.g. for the ellipse+absolute / plain baseline ablation).
      - "residual": denoiser_out is the residual on top of the proposal;
        x0 = proposal + denoiser_out. The proposal supplies global
        position/size/coarse-shape deterministically; only the remainder
        is modeled generatively by the diffusion process.
    Used identically in training (to reconstruct pred_x0 for the loss) and
    in sampling (inside ddim_sample, applied to the per-step x0 prediction).
    """
    if proposal_target == "absolute":
        return denoiser_out
    if proposal_target == "residual":
        if proposal is None:
            raise ValueError("proposal_target='residual' requires a proposal tensor")
        return proposal + denoiser_out
    raise ValueError(f"unknown proposal_target: {proposal_target!r}")


def proposal_dice_loss(proposal_head, raw_maps, masks, size=64):
    """Standalone, time-independent soft-Dice loss on the raw proposal shape(s).

    Trains the proposal head to already produce a decent coarse *shape* on
    its own, before any diffusion residual is added -- independent of
    `proposal_target` and independent of the denoiser entirely.

    With `proposal_head.n_shapes == 1` this is exactly the old V5 behavior:
    soft-Dice of the single shape's own points against the GT mask.

    With `n_shapes > 1`, this is THE gradient path for the shape parameters
    (see module docstring): it rasterizes each shape independently
    (differentiable `soft_rasterize`), combines them via soft-OR, and scores
    the union against the GT mask -- all differentiable end-to-end, unlike
    `proposal_head.forward`'s point-contour output for n_shapes>1 (which
    goes through non-differentiable cv2 contour extraction and carries no
    gradient back to the shape parameters at all). Without this loss with
    lambda > 0, a multi-shape proposal head would never receive any direct
    training signal on its own geometry.

    Call this directly on the raw encoder pyramid, with its own lambda,
    additive to the total loss:

        loss_proposal_dice = proposal_dice_loss(proposal_head, raw, masks, size=cfg.soft_dice_size)
        ...
        loss = loss + cfg.lambda_proposal_dice * loss_proposal_dice

    No `t`/`T` are passed through to `soft_dice_loss` -- the proposal is not
    part of the diffusion timestep chain, so there is no "early vs. late
    denoising" to weight against; the loss is just plain (unweighted
    per-sample-averaged) soft-Dice.

    proposal_head: a ContourProposalHead instance (needed so this can access
        `forward_shapes` directly and skip the non-differentiable union
        path entirely -- calling `proposal_head(raw_maps)` here instead
        would silently break the gradient at n_shapes>1).
    masks: [B, 1, H, W] in {0, 1}, same GT masks used elsewhere.
    """
    shapes_points = proposal_head.forward_shapes(raw_maps)  # [B, S, N, 2], differentiable
    if proposal_head.n_shapes == 1:
        return soft_dice_loss(shapes_points[:, 0], masks, size=size)

    union = soft_union_mask(shapes_points, size=size)  # [B, size, size], differentiable
    gt = torch.nn.functional.interpolate(masks.float(), size=(size, size), mode="area")
    gt = (gt.squeeze(1) > 0.5).float()
    eps = 1e-6
    inter = (union * gt).sum(dim=(1, 2))
    denom = union.sum(dim=(1, 2)) + gt.sum(dim=(1, 2))
    dice = (2.0 * inter + eps) / (denom + eps)
    return (1.0 - dice).mean()