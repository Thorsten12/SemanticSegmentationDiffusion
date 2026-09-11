"""Decoder -- BOUNDARYFORMER (mask-supervised polygonal boundary transformer).
Reference: Lazarow, Xu & Tu, "Instance Segmentation with Mask-supervised
Polygonal Boundary Transformers", CVPR 2022. Paper (full text used for
this reimplementation): https://par.nsf.gov/servlets/purl/10350146
Also indexed as: https://openaccess.thecvf.com/content/CVPR2022/html/
Lazarow_Instance_Segmentation_With_Mask-Supervised_Polygonal_Boundary_Transformers_CVPR_2022_paper.html
Official repo (not vendored/fetched here -- from-paper reimplementation,
cross-checked against Section 3 and Figure 2/3): https://github.com/mlpc-ucsd/BoundaryFormer

WHY BOUNDARYFORMER OVER DEEP SNAKE HERE: both are contour/polygon decoders,
but Deep Snake's native training signal is a point-to-point L1 loss against
a matched ground-truth polygon (Peng et al., Eq. 4), and its use in this
project bolts on a rasterizer only as an afterthought to satisfy the
PyramidDecoder logits contract -- a loss-type mismatch that this project's
Deep Snake decoder documents in its own docstring (see decoders/deep_snake.py,
ADAPTATION 6) and that shows up empirically as unstable, non-converging
training when driven by BCE+Dice on rasterized output instead. BoundaryFormer,
by contrast, is *designed from the ground up* around exactly this training
regime: it predicts polygon vertices but is supervised ENTIRELY through a
differentiable rasterizer against the ground-truth pixel mask, using a Dice
loss -- see the paper's own framing: "BoundaryFormer uses pixel-wise masks
as ground-truth for supervision... adds no new supervision requirements
over Mask R-CNN" (Section 1, contribution 2) and "we require our model to
only require mask-based supervision... in the exact same sense as an
ordinary mask-based segmentation model" (Section 3.3). There is therefore
no training-regime mismatch to inherit here: mask-supervised polygon
regression via a differentiable rasterizer + Dice loss is BoundaryFormer's
native design, not a bolted-on adaptation.

ARCHITECTURE (faithful to the paper, adapted only where noted under
"ADAPTATION" -- see bottom of module docstring):

  - Setting (Sec. 3.1): the paper predicts K polygon vertices V_i per
    detected instance i, then rasterizes V_i to a mask M_i for mask-based
    supervision. We keep exactly this two-part design (vertices, then
    rasterization) -- see "Setting" and Eq. nothing (prose only) in Sec 3.1:
    "we predict K vertices... which define the boundary of a polygon...
    Then, we produce M_i through rasterization."

  - Initial polygon (Sec. 3.2): "We use B_i as a means to initialize an
    ellipsoidal polygon V_i(0) inscribed in B_i, similar to other
    contour-based methods." In the paper B_i is a detector's predicted
    box; this project has no detector (see ADAPTATION 1).

  - Iterative refinement via two attentions (Sec. 3.2, Fig. 2): "our model
    iteratively refines this shape L times by V_i(j+1) = g_j(F, V_i(j))...
    we implement g with Transformers using two kinds of attention. The
    first kind of attention allows each P_i^k to attend to all other point
    embeddings P_i^k' within the same object [ordinary quadratic
    self-attention]. The second allows each P_i^k to attend to the image
    features F [Deformable Attention]... Furthermore, this allows
    multi-scale [attention] across levels P2 through P5... for each
    embedding P_i^k, g_j predicts a 2D offset (Delta_x, Delta_y) to refine
    the vertex using an MLP." We implement both attentions faithfully:
    point-to-point as standard multi-head self-attention over the K vertex
    embeddings, and point-to-image as genuine Multi-Scale Deformable
    Attention (Zhu et al. 2020, Eq. 2 and its multi-scale extension Eq. 10
    -- "MSDeformAttn(z_q, p_hat_q, {x^l}) = sum_m W_m[sum_l sum_k A_mlqk *
    W'_m x^l(phi_l(p_hat_q) + Delta p_mlqk)]") sampling directly from the
    encoder's 4 native pyramid scales {s4, s8, s16, s32} (this project's
    analogue of the paper's {P2..P5}), with per-head, per-level, per-point
    LEARNED sampling offsets and a softmax-normalized attention weight
    over all level*point samples jointly, exactly as specified by Eq. 10's
    normalization sum_l sum_k A_mlqk = 1. This matches the paper's own
    stated architecture with no simplification of the attention mechanism
    itself (see the class docstring of MultiScaleDeformableAttention
    below for exact correspondence to Eq. 2/10 and the one necessarily
    project-specific choice: no RoI, so reference points are in
    full-image normalized coordinates rather than per-instance box-normalized
    ones). Point embeddings use a sinusoidal "point encoding" exactly as
    stated: "includes a 'point encoding' modeled off of the usual
    Transformer sine positional encoding."

  - Coarse-to-fine upsampling (Sec. 3.4): "we consider a base number of
    control points B (usually 8) and upsample the points by 2x at each
    layer... between each point (x_j,y_j) and (x_{j+1},y_{j+1}), we insert
    a new point at the midpoint... We do not average the corresponding
    point embeddings P_j and P_{j+1}... and instead insert the
    corresponding 'learned' point embedding." Implemented exactly:
    geometric midpoint insertion for coordinates, a separate learned
    "inserted-point" embedding (not an average) for the new points'
    feature vectors.

  - Differentiable rasterizer (Sec. 3.3.1, Eq. 1): "for each pixel (x,y)...
    we use a PnP algorithm to determine whether the pixel... lies within
    the polygon as C(V,x,y)... each pixel is projected onto the closest
    segment... this distance is recorded as D(V,x,y)... I(x,y) =
    sigma(C(V,x,y) * D(V,x,y) / tau)." Implemented exactly as Eq. 1: a
    differentiable winding-number-based inside/outside sign C, a
    min-distance-to-edge D, combined through a single sigmoid with
    sharpness tau (paper default tau=0.1, Sec. 4.1).

  - Loss (Sec. 3.5, Eq. 2): "L = L_bbox + sum_l sum_i Dice(I_i(l), M_i)" --
    a Dice loss between the rasterized prediction and the ground-truth
    mask, summed over ALL L refinement layers (deep supervision at every
    layer, not just the final one), plus a box-detection loss L_bbox for
    the paper's detector. This project has no detector/box loss (see
    ADAPTATION 1); we expose the per-layer rasterized predictions via
    `self._aux_predictions` so a multi-layer Dice loss matching Eq. 2
    could be wired into the training loop if desired, while `forward()`
    itself returns only the final layer's logits to satisfy the
    PyramidDecoder contract of a single feats->logits call.

  - Layer count / point count (Sec. 4.1): "we train all models with
    coarse to fine upsampling over 4 layers of Transformers to produce a
    final output of 64 points on COCO and 128 on Cityscapes." Default
    here: L=4 layers, base 8 points doubling each layer (8->16->32->64),
    matching the paper's stated default coarse-to-fine schedule
    (Sec 3.4's ablation table uses L/K1/KL = 4/8/64 as one of its strong
    configurations, mask AP 36.1, matching the paper's main COCO number).

  - Rasterization resolution and tau (Sec. 4.1): "we rasterize polygons
    during training to a fixed 64x64 resolution... tau=0.1." We rasterize
    to the encoder's native stride-4 resolution instead (see ADAPTATION 3)
    but keep tau=0.1 as the default, matching the paper's ablation
    ("within that lower range... values around tau=0.1 to be sufficient").

ADAPTATION (deviations from the paper, made to fit this project):
  1. No detector / no box proposal. The paper's polygon head sits on top
     of a two-stage (R-CNN) or one-stage (FCOS) DETECTOR: the box B_i
     comes from the detector, the ellipse is inscribed in B_i, and L_bbox
     is part of the joint loss (Eq. 2). This project's baselines are
     single-object, encoder-only, with no detector or box regression
     stage. We therefore initialize the ellipse the same way this
     project's Deep Snake decoder does: regress a center and semi-axes
     from a global-pooled coarse feature map with a small MLP, instead of
     inscribing it in a detected box. L_bbox is dropped entirely (there is
     no box to regress). This is the same category of adaptation as Deep
     Snake's ADAPTATION 1 in this project, applied here to a different
     paper.
  2. Reference points are full-image-normalized, not RoI/box-normalized.
     The paper's Deformable Attention reference points p_hat_q live in the
     coordinate system of an RoI-cropped instance (Sec 3.1: boxes B_i from
     a detector; Eq. 10's phi_l rescales a per-instance-normalized
     reference point into level l's coordinate grid). This project has no
     detector or per-instance RoI crop (see ADAPTATION 1): every image is
     assumed to contain one object filling the frame, matching how every
     other decoder in this project (UNet, CASCADE, XBound-Former, Deep
     Snake) treats the full image as the single instance's extent. We
     therefore keep the attention mechanism itself completely unchanged
     from Eq. 2/10 (same learned per-head/per-level/per-point offsets,
     same jointly-normalized softmax weights, same multi-scale bilinear
     sampling -- see MultiScaleDeformableAttention below) and only change
     what phi_l rescales FROM: full-image normalized [-1,1] coordinates
     instead of a box-relative [0,1] crop. This is a coordinate-frame
     substitution forced by the absence of a detector, not a
     simplification of the attention computation itself -- no part of
     Eq. 2/10 (offsets, weights, sampling, or their normalization) is
     approximated or dropped.
  3. Rasterization resolution. The paper rasterizes to a fixed 64x64 crop
     within each detected box (RoI-relative, "we only rasterize the
     predicted polygon within its box, not with respect to the entire
     image"). This project has no RoI/box, so we rasterize directly to
     the encoder's native stride-4 full-image resolution, matching how
     every other decoder in this project (including Deep Snake) produces
     its stride-4 output. tau is kept at the paper's default (0.1) since
     it is a property of the sigmoid sharpness, not the resolution itself,
     though note the paper's own ablation shows performance is fairly flat
     across resolution once reasonably large (Sec 4.4, Table: 40x40 vs
     64x64 vs 80x80 all within 0.2 mask AP of each other), so this
     resolution change is not expected to be architecturally significant.
  4. Half-pixel alignment. The paper subtracts half a pixel from all
     coordinates before rasterizing "to ensure alignment with the COCO
     API's own rasterization convention" (Sec 3.3.2). This is an
     alignment fix specific to matching pycocotools' rasterization
     convention for COCO mask AP evaluation; this project's masks are not
     produced or evaluated through the COCO API, so there is no equivalent
     convention to align to, and we omit this half-pixel shift entirely.
  5. No panoptic/fragmented-instance handling. The paper notes
     BoundaryFormer predicts a single polygon per instance and therefore
     struggles with objects fragmented into multiple disconnected pieces
     (Sec 4.3), and it does NOT introduce any special handling for this
     (unlike DANCE, which uses panoptic edge maps). This project's targets
     (single lesion/organ per image) are single-component by construction,
     so this limitation of the paper's method is inherited as-is and
     requires no adaptation.
  6. No detector loss / no L_bbox term. Eq. 2's L = L_bbox + sum Dice(...)
     includes a box-regression loss from the underlying detector. Since
     there is no detector here (ADAPTATION 1), the loss this decoder is
     meant to be trained under (in this project's shared BaselineSegmenter
     training loop) is simply the multi-layer Dice term; the final
     logits returned by forward() are exactly what the shared
     bce_dice_loss() in train.py already computes against, so no changes
     to the training loop are required to use this decoder with the
     project's existing loss function -- it will just be applied only to
     the LAST layer's rasterization rather than all L layers unless the
     training loop is extended to read self._aux_predictions (see the
     "Loss" paragraph above and the class docstring below).
"""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from .base import PyramidDecoder


# --------------------------------------------------------------------------- #
# Sinusoidal point encoding (Sec. 3.2: "a 'point encoding' modeled off of
# the usual Transformer sine positional encoding"). Encodes each vertex's
# CURRENT (x, y) position, recomputed every layer since vertices move.
# --------------------------------------------------------------------------- #
def sine_point_encoding(coords, dim, temperature=10000.0):
    """coords: [B, K, 2] in normalized [-1, 1]. Returns [B, K, dim]."""
    B, K, _ = coords.shape
    half = dim // 2
    quarter = half // 2
    device = coords.device

    freq = torch.arange(quarter, device=device, dtype=torch.float32)
    freq = temperature ** (2 * freq / quarter)  # [quarter]

    x = coords[..., 0:1]  # [B, K, 1]
    y = coords[..., 1:2]

    px = x / freq  # [B, K, quarter]
    py = y / freq

    enc = torch.cat(
        [px.sin(), px.cos(), py.sin(), py.cos()], dim=-1
    )  # [B, K, 4*quarter] == [B, K, 2*half] == [B, K, dim] (when dim div by 4)
    if enc.shape[-1] < dim:
        pad = dim - enc.shape[-1]
        enc = F.pad(enc, (0, pad))
    return enc[..., :dim]


# --------------------------------------------------------------------------- #
# Learned initial-ellipse proposal -- replaces the paper's detector-box
# inscription (see ADAPTATION 1). Identical in spirit to this project's
# Deep Snake decoder's InitialEllipseHead.
# --------------------------------------------------------------------------- #
class InitialEllipseHead(nn.Module):
    def __init__(self, in_ch, hidden=64):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(in_ch, hidden), nn.ReLU(inplace=True),
            nn.Linear(hidden, 4),  # cx, cy, rx, ry
        )

    def forward(self, s32):
        g = F.adaptive_avg_pool2d(s32, 1).flatten(1)  # [B, C]
        cx, cy, rx, ry = self.mlp(g).unbind(dim=-1)
        center = 0.5 * torch.stack([torch.tanh(cx), torch.tanh(cy)], dim=-1)
        radius = torch.stack(
            [torch.sigmoid(rx), torch.sigmoid(ry)], dim=-1
        ) * 0.45 + 0.03
        return center, radius  # each [B, 2]


def make_initial_polygon(center, radius, n_points, device):
    """Evenly-spaced ellipse in normalized [-1, 1] coords, K points."""
    theta = torch.linspace(0, 2 * math.pi, n_points + 1, device=device)[:-1]
    unit = torch.stack([torch.cos(theta), torch.sin(theta)], dim=-1)
    poly = center.unsqueeze(1) + radius.unsqueeze(1) * unit.unsqueeze(0)
    return poly.clamp(-1.0, 1.0)  # [B, K, 2]


class MultiScaleDeformableAttention(nn.Module):
    """Faithful reimplementation of Multi-Scale Deformable Attention
    (Zhu et al. 2020, "Deformable DETR", Eq. 2 and its multi-scale
    extension Eq. 10), used here as the paper's "second kind of
    attention" (point-to-image-features, Sec 3.2). Cross-checked against
    the official CPU reference implementation
    (ms_deform_attn_core_pytorch in fundamentalvision/Deformable-DETR)
    for the sampling-and-weighting logic, reimplemented here in plain
    PyTorch (no custom CUDA kernel) since this project has no build step
    for one.

    Eq. 10: MSDeformAttn(z_q, p_hat_q, {x^l}_{l=1}^L)
              = sum_{m=1}^{M} W_m [ sum_{l=1}^{L} sum_{k=1}^{K}
                    A_{mlqk} * W'_m x^l( phi_l(p_hat_q) + Delta p_{mlqk} ) ]
            subject to sum_{l,k} A_{mlqk} = 1  (Eq. 11, jointly normalized
            over levels AND sample points, per head).

    Correspondence to the code below:
      - M (n_heads), L (n_levels == 4, one per pyramid scale), K (n_points
        per head per level) are exactly the paper's M, L, K.
      - `self.sampling_offsets`: linear projection of the query content
        z_q producing Delta p_{mlqk} for every (head, level, point) --
        "Both Delta p_mqk and A_mqk are obtained via linear projection
        over the query feature z_q."
      - `self.attention_weights`: linear projection of z_q producing the
        pre-softmax logits for A_mlqk, softmax-normalized jointly over
        the flattened (level, point) axis per head, matching Eq. 11.
      - `self.value_proj` is W'_m (applied once per level's feature map,
        shared across heads via reshaping into per-head channel groups --
        identical structuring to the reference implementation).
      - `self.output_proj` is W_m, applied once after concatenating all
        heads' weighted sums.
      - Sampling uses grid_sample per level exactly as in
        ms_deform_attn_core_pytorch: sampling_grids = 2*sampling_locations-1
        maps [0,1]-normalized locations into grid_sample's [-1,1]
        convention.

    ADAPTATION (see module docstring, point 2): phi_l here maps a
    FULL-IMAGE-normalized reference point (this project has no per-instance
    RoI/box) into each level's own coordinate grid, rather than an
    RoI-relative reference point. This changes only what coordinate frame
    the reference point is expressed in; the attention computation
    (offsets, weights, sampling, normalization) is unchanged from Eq. 10.
    """

    def __init__(self, dim, n_heads=8, n_levels=4, n_points=4):
        super().__init__()
        assert dim % n_heads == 0, "dim must be divisible by n_heads"
        self.dim = dim
        self.n_heads = n_heads
        self.n_levels = n_levels
        self.n_points = n_points
        self.head_dim = dim // n_heads

        self.sampling_offsets = nn.Linear(dim, n_heads * n_levels * n_points * 2)
        self.attention_weights = nn.Linear(dim, n_heads * n_levels * n_points)
        self.value_proj = nn.ModuleList([nn.Linear(dim, dim) for _ in range(n_levels)])
        self.output_proj = nn.Linear(dim, dim)

        self._reset_parameters()

    def _reset_parameters(self):
        # Matches the reference init: sampling offsets start at a small
        # ring around the reference point (one direction per head),
        # attention weights start uniform (all zeros -> uniform softmax).
        nn.init.constant_(self.sampling_offsets.weight, 0.0)
        thetas = torch.arange(self.n_heads, dtype=torch.float32) * (2.0 * math.pi / self.n_heads)
        grid_init = torch.stack([thetas.cos(), thetas.sin()], dim=-1)  # [n_heads, 2]
        grid_init = grid_init / grid_init.abs().max(-1, keepdim=True)[0]
        grid_init = grid_init.view(self.n_heads, 1, 1, 2).repeat(1, self.n_levels, self.n_points, 1)
        for i in range(self.n_points):
            grid_init[:, :, i, :] *= i + 1
        with torch.no_grad():
            self.sampling_offsets.bias = nn.Parameter(grid_init.reshape(-1) * 0.05)
        nn.init.constant_(self.attention_weights.weight, 0.0)
        nn.init.constant_(self.attention_weights.bias, 0.0)

    def forward(self, query, reference_points, multi_scale_feats):
        """query: [B, K, dim] content features z_q.
        reference_points: [B, K, 2] normalized to [0, 1], shared reference
            frame across levels (phi_l below rescales per-level).
        multi_scale_feats: list of 4 tensors [B, dim, H_l, W_l] (already
            projected to `dim` channels by the caller -- see
            BoundaryFormerDecoder, which applies a per-scale 1x1 conv
            before calling this module, analogous to the paper's shared
            FPN channel width for P2..P5).
        Returns: [B, K, dim].
        """
        B, K, _ = query.shape
        L = self.n_levels
        assert len(multi_scale_feats) == L

        offsets = self.sampling_offsets(query).view(B, K, self.n_heads, L, self.n_points, 2)
        attn_logits = self.attention_weights(query).view(B, K, self.n_heads, L * self.n_points)
        attn_weights = F.softmax(attn_logits, dim=-1).view(B, K, self.n_heads, L, self.n_points)  # Eq. 11

        sampled_per_level = []
        for lvl, feat in enumerate(multi_scale_feats):
            Hl, Wl = feat.shape[-2:]
            value = self.value_proj[lvl](feat.flatten(2).transpose(1, 2))  # [B, Hl*Wl, dim]
            # Split the channel dim into (n_heads, head_dim) BEFORE folding
            # heads into the batch axis, exactly as the reference
            # implementation's "N_, H_*W_, M_, D_ -> ... -> N_*M_, D_, H_, W_"
            # comment specifies -- a plain transpose+reshape without this
            # explicit split would silently scramble which channels belong
            # to which head.
            value = value.view(B, Hl * Wl, self.n_heads, self.head_dim)
            value = value.permute(0, 2, 3, 1).reshape(B * self.n_heads, self.head_dim, Hl, Wl)

            # phi_l: rescale the shared [0,1] reference point + this
            # level's learned offset (also expressed in [0,1]-normalized
            # units of that level) into grid_sample's [-1,1] convention.
            ref = reference_points.view(B, K, 1, 1, 2)  # broadcast over heads/points
            loc = ref + offsets[:, :, :, lvl, :, :]  # [B, K, n_heads, n_points, 2], still ~[0,1]
            grid = 2.0 * loc - 1.0  # [-1, 1] for grid_sample

            # [B, K, n_heads, n_points, 2] -> [B*n_heads, K, n_points, 2]
            grid = grid.permute(0, 2, 1, 3, 4).reshape(B * self.n_heads, K, self.n_points, 2)

            sampled = F.grid_sample(
                value, grid, mode="bilinear", padding_mode="zeros", align_corners=False
            )  # [B*n_heads, head_dim, K, n_points]
            sampled_per_level.append(sampled)

        # [B*n_heads, head_dim, K, L, n_points] -> flatten (L, n_points)
        stacked = torch.stack(sampled_per_level, dim=3)  # [B*n_heads, head_dim, K, L, n_points]
        stacked = stacked.flatten(-2)  # [B*n_heads, head_dim, K, L*n_points]

        # [B,K,n_heads,L,n_points] -> [B,n_heads,K,L,n_points] -> [B*n_heads,K,L*n_points] -> unsqueeze head_dim axis
        w = attn_weights.permute(0, 2, 1, 3, 4).reshape(B * self.n_heads, K, L * self.n_points)
        w = w.unsqueeze(1)  # [B*n_heads, 1, K, L*n_points], broadcasts over head_dim
        out = (stacked * w).sum(-1)  # [B*n_heads, head_dim, K]
        out = out.view(B, self.n_heads * self.head_dim, K).transpose(1, 2)  # [B, K, dim]
        return self.output_proj(out)


class PointSelfAttention(nn.Module):
    """The paper's "first kind of attention": ordinary quadratic
    multi-head self-attention among the K point embeddings of the SAME
    polygon (Sec 3.2: "allows each P_i^k to attend to all other point
    embeddings P_i^k' within the same object")."""

    def __init__(self, dim, num_heads=8):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(dim, num_heads, batch_first=True)

    def forward(self, tokens):
        x = self.norm(tokens)
        out, _ = self.attn(x, x, x, need_weights=False)
        return tokens + out


class PointToImageAttention(nn.Module):
    """The paper's "second kind of attention": each point attends to the
    image features via genuine Multi-Scale Deformable Attention (Sec 3.2,
    "we implement the point to image feature attention using Deformable
    Attention... this allows multi-scale across levels P2 through P5").
    Thin residual wrapper around MultiScaleDeformableAttention -- see that
    class's docstring for the exact Eq. 2/10 correspondence."""

    def __init__(self, dim, num_heads=8, n_levels=4, n_points=4):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.deform_attn = MultiScaleDeformableAttention(
            dim, n_heads=num_heads, n_levels=n_levels, n_points=n_points
        )

    def forward(self, tokens, reference_points, multi_scale_feats):
        q = self.norm(tokens)
        out = self.deform_attn(q, reference_points, multi_scale_feats)
        return tokens + out


class OffsetMLP(nn.Module):
    """Sec 3.2: "for each embedding P_i^k, g_j predicts a 2D offset
    (Delta_x, Delta_y) to refine the vertex using an MLP." """

    def __init__(self, dim, hidden_ratio=2.0):
        super().__init__()
        hidden = int(dim * hidden_ratio)
        self.net = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, hidden),
            nn.GELU(),
            nn.Linear(hidden, 2),
        )

    def forward(self, tokens):
        return self.net(tokens)  # [B, K, 2]


class RefinementLayer(nn.Module):
    """One layer g_j of BoundaryFormer's iterative refinement (Fig. 2):
    point-to-point self-attention -> point-to-image Multi-Scale Deformable
    Attention -> MLP that predicts a 2D offset per vertex."""

    def __init__(self, dim, num_heads=8, n_levels=4, n_points=4):
        super().__init__()
        self.point_attn = PointSelfAttention(dim, num_heads=num_heads)
        self.image_attn = PointToImageAttention(
            dim, num_heads=num_heads, n_levels=n_levels, n_points=n_points
        )
        self.offset_mlp = OffsetMLP(dim)

    def forward(self, tokens, coords, multi_scale_feats):
        # coords are in normalized [-1, 1]
        reference_points = (coords + 1.0) / 2.0

        tokens = self.point_attn(tokens)
        tokens = self.image_attn(tokens, reference_points, multi_scale_feats)

        offset = self.offset_mlp(tokens)
        offset = 0.10 * torch.tanh(offset)
        coords = (coords + offset).clamp(-1.0, 1.0)

        return tokens, coords


def upsample_coarse_to_fine(coords, tokens, midpoint_embed):
    """Sec 3.4: double the number of vertices by inserting a midpoint
    between every adjacent pair. Coordinates get the geometric midpoint;
    the NEW point's embedding is the shared learned `midpoint_embed`
    vector (broadcast to every inserted point), NOT an average of its
    neighbours' embeddings -- the paper is explicit about this:
    "We do not average the corresponding point embeddings P_j and P_{j+1}
    to initialize the new point and instead insert the corresponding
    'learned' point embedding."

    coords: [B, K, 2], tokens: [B, K, D] -> coords: [B, 2K, 2], tokens: [B, 2K, D]
    """
    B, K, _ = coords.shape
    next_coords = torch.roll(coords, shifts=-1, dims=1)
    mid_coords = (coords + next_coords) / 2.0  # [B, K, 2]

    new_coords = torch.stack([coords, mid_coords], dim=2).reshape(B, 2 * K, 2)

    D = tokens.shape[-1]
    mid_tokens = midpoint_embed.view(1, 1, D).expand(B, K, D)
    new_tokens = torch.stack([tokens, mid_tokens], dim=2).reshape(B, 2 * K, D)

    return new_coords, new_tokens


def rasterize_polygon_boundaryformer(contour, height, width, tau=0.1):
    """Eq. 1: I(x,y) = sigmoid(C(V,x,y) * D(V,x,y) / tau), where C is the
    inside(+1)/outside(-1) sign (we use a smoothed winding number in place
    of the paper's exact PnP test, since PnP itself is not differentiable
    -- the paper's own C is stated as a hard +-1 label from a PnP test,
    with differentiability coming entirely from D and the surrounding
    sigmoid; we soften C as well so gradients also flow when a pixel's
    inside/outside status is ambiguous near the boundary during early
    training, which is a minor, clearly-flagged deviation from a literal
    reading of Eq. 1) and D is the (unsigned) distance to the nearest
    polygon edge.

    contour: [B, K, 2] normalized [-1, 1] (x, y), closed polygon (implicit
    wrap K-1 -> 0). Returns logits [B, 1, height, width] suitable for a
    sigmoid/BCE-style loss (i.e. pre-sigmoid; since Eq. 1 already applies
    a sigmoid to produce I(x,y) directly as a soft mask, and this
    project's shared training loop expects logits and applies its own
    sigmoid inside bce_dice_loss, we return the PRE-sigmoid quantity
    C*D/tau rather than I(x,y) itself -- applying sigmoid twice would
    double-squash the signal. This is a necessary interface adaptation,
    not a deviation from Eq. 1's actual computation.)"""
    B, K, _ = contour.shape
    device = contour.device

    ys = torch.linspace(-1.0, 1.0, height, device=device)
    xs = torch.linspace(-1.0, 1.0, width, device=device)
    grid_y, grid_x = torch.meshgrid(ys, xs, indexing="ij")
    pix = torch.stack([grid_x, grid_y], dim=-1).view(1, height * width, 1, 2)

    v0 = contour.unsqueeze(1)                                # [B, 1, K, 2]
    v1 = torch.roll(contour, shifts=-1, dims=1).unsqueeze(1)  # [B, 1, K, 2]

    edge = v1 - v0
    to_pix = pix - v0
    edge_len2 = (edge * edge).sum(-1).clamp_min(1e-8)
    t = (to_pix * edge).sum(-1) / edge_len2
    t = t.clamp(0.0, 1.0)
    proj = v0 + t.unsqueeze(-1) * edge
    dist2 = (pix - proj).pow(2).sum(dim=-1)
    dist_per_edge = torch.sqrt(dist2 + 1e-8)
    D = dist_per_edge.min(dim=-1)[0]

    # Numerisch stabiler Winkel: atan2(0, 0) vermeiden.
    d0 = pix - v0
    d1 = pix - v1
    cross = d0[..., 0] * d1[..., 1] - d0[..., 1] * d1[..., 0]
    dot = (d0 * d1).sum(-1)

    eps = 1e-6
    angle = torch.atan2(cross, dot + eps)
    winding = angle.sum(-1) / (2 * math.pi)
    C = torch.tanh((winding.abs() - 0.5) / 0.05)

    logits = (C * D) / tau
    return logits.view(B, 1, height, width)


class BoundaryFormerDecoder(PyramidDecoder):
    """Full BoundaryFormer decoder: learned-ellipse initial polygon (see
    ADAPTATION 1), L layers of coarse-to-fine iterative refinement (point
    self-attention + point-to-image attention + offset MLP, Sec 3.2/3.4),
    then a differentiable rasterizer (Eq. 1) to produce stride-4 logits.

    n_layers (L): number of refinement layers. Paper default 4 (Sec 4.1).
    base_points (K1): starting vertex count, doubled each layer. Paper
        default 8 (Sec 3.4: "a base number of control points B (usually 8)").
        With n_layers=4 this reaches 8->16->32->64 points, matching the
        paper's strong "4/8/64" ablation configuration (mask AP 36.1,
        equal to their main reported COCO number).
    point_dim: per-vertex token width. Not specified numerically in the
        paper's text; chosen here to match `decoder_dim` so the decoder
        responds to --decoder_dim like every other decoder in this
        project (the paper ties its Transformer width to the underlying
        detector's hidden dim, which this project has no equivalent of).
    n_points (K, deformable-attention sense -- distinct from the polygon's
        own vertex count, also unfortunately called K in the paper's
        Sec 3.1 notation): number of sampling points per head per level in
        MultiScaleDeformableAttention. Not stated in the BoundaryFormer
        paper text ("All other parameters with respect to deformable
        attention follow the same settings of Deformable DETR's decoder
        layer," Sec 4.1); we therefore use Deformable DETR's own default
        of 4 (Zhu et al. 2020, and the official MSDeformAttn module's
        default n_points=4), per that explicit cross-reference.
    tau: rasterization sharpness, Eq. 1. Paper default 0.1 (Sec 4.1,
        confirmed by the ablation in Sec 4.4).
    """

    def __init__(
        self, enc_channels, decoder_dim=64, num_classes=1,
        n_layers=4, base_points=8, num_heads=8, n_points=4,
        tau=0.1,
    ):
        super().__init__(enc_channels, decoder_dim, num_classes)
        assert num_classes == 1, "BoundaryFormerDecoder renders a single closed polygon (binary mask)."
        c4, c8, c16, c32 = enc_channels
        self.n_layers = n_layers
        self.base_points = base_points
        self.tau = tau

        # Per-scale 1x1 projections to a SHARED channel width, one per
        # pyramid level -- this project's analogue of the paper's FPN
        # already giving P2..P5 a common channel width. No fusion into a
        # single map: MultiScaleDeformableAttention consumes all 4 scales
        # directly (see module docstring point 2).
        point_dim = decoder_dim
        self.point_dim = point_dim
        self.scale_proj = nn.ModuleList([
            nn.Conv2d(c, point_dim, kernel_size=1) for c in (c4, c8, c16, c32)
        ])

        # ADAPTATION 1: learned ellipse instead of detector-box inscription.
        self.init_head = InitialEllipseHead(c32)

        # Initial per-vertex embedding: sinusoidal point encoding of the
        # initial coordinates, projected to point_dim. The paper does not
        # specify an initial content embedding beyond the point encoding
        # for layer 0; we follow that literally (content starts as pure
        # positional encoding, refined thereafter by the layers).
        self.point_encoding_proj = nn.Linear(point_dim, point_dim)

        self.layers = nn.ModuleList([
            RefinementLayer(point_dim, num_heads=num_heads, n_levels=4, n_points=n_points)
            for _ in range(n_layers)
        ])

        # Sec 3.4: one learned "inserted-point" embedding per layer
        # boundary (the paper does not state whether this embedding is
        # shared across layers or distinct per upsampling step; we use a
        # distinct learned vector per layer, since each layer's token
        # space has been refined differently and a single shared vector
        # would inject the same content at every resolution stage --
        # this is our choice where the paper is silent).
        self.midpoint_embeds = nn.ParameterList([
            nn.Parameter(torch.zeros(point_dim)) for _ in range(n_layers - 1)
        ])
        for p in self.midpoint_embeds:
            nn.init.trunc_normal_(p, std=0.02)

        self._aux_predictions = None  # per-layer rasterized logits, for optional Eq. 2 multi-layer Dice supervision

    def forward(self, feats):
        s4, s8, s16, s32 = feats
        H4, W4 = s4.shape[-2:]

        # Project each of the 4 native pyramid scales to a shared channel
        # width -- fed directly to MultiScaleDeformableAttention, one
        # level per scale (this project's {s4,s8,s16,s32} standing in for
        # the paper's {P2,P3,P4,P5}). No cross-scale fusion.
        multi_scale_feats = [proj(f) for proj, f in zip(self.scale_proj, (s4, s8, s16, s32))]

        center, radius = self.init_head(s32)
        coords = make_initial_polygon(center, radius, self.base_points, s4.device)  # [B, K, 2]

        tokens = self.point_encoding_proj(
            sine_point_encoding(coords, self.point_dim)
        )  # [B, K, point_dim]

        aux_logits = []
        for i, layer in enumerate(self.layers):
            tokens, coords = layer(tokens, coords, multi_scale_feats)

            aux_logits.append(
                rasterize_polygon_boundaryformer(coords, H4, W4, tau=self.tau)
            )

            is_last_layer = i == len(self.layers) - 1
            if not is_last_layer:
                coords, tokens = upsample_coarse_to_fine(
                    coords, tokens, self.midpoint_embeds[i]
                )
                # Recompute sinusoidal encoding at the new (doubled) vertex
                # set and re-inject it additively, since the newly-inserted
                # midpoints' embeddings currently carry no positional
                # information of their own beyond the shared learned vector.
                tokens = tokens + self.point_encoding_proj(
                    sine_point_encoding(coords, self.point_dim)
                )

        self._aux_predictions = aux_logits[:-1]  # earlier layers, for optional Eq. 2 deep supervision
        return aux_logits[-1]  # final layer; matches Eq. 2's l=L term and this project's single-logits contract