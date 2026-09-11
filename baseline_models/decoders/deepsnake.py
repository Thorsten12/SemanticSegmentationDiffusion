"""Decoder -- DEEP SNAKE (contour-based, direct polygon regression).
Reference: Peng et al., "Deep Snake for Real-Time Instance Segmentation",
CVPR 2020 (oral). Paper: https://arxiv.org/abs/2001.01629
Official repo (not vendored/fetched here -- GitHub blocks automated raw-file
access -- this is a from-equations reimplementation, cross-checked against
the paper's Figure 2/3 and Section 3, plus secondary sources that describe
the released snake.py structure): https://github.com/zju3dv/snake

WHY DEEP SNAKE OVER BOUNDARYFORMER HERE: both do direct contour/polygon
regression instead of per-pixel diffusion, but BoundaryFormer needs a
transformer decoder + a bespoke differentiable rasterizer loss and a fixed
target vertex count matched by Hungarian-style ordering. Deep Snake's
"iteratively deform an initial contour" design drops straight onto this
project's existing n_points-contour representation (already used by
P2SDiff / build_contour_dataset) with a plain 1-D CNN, so it is the
simpler decoder to wire into the shared BaselineSegmenter / PyramidDecoder
training loop.

ARCHITECTURE (faithful to the paper, adapted only where noted under
"ADAPTATION" -- see bottom of module docstring):

  - Circular Convolution (Fig. 2): the contour is a closed cycle graph, so
    every 1-D conv over the vertex sequence wraps around (circular / "same"
    padding with no seam). Kernel size 9 (n_adj=4 neighbours each side),
    matching the paper's stated CirConv kernel size.

  - Backbone (Fig. 3a): "8 CirConv-BN-ReLU layers with residual skip
    connections for all layers" -- 1 head layer (dilation 1) + 7 residual
    BasicBlocks with dilation schedule [1, 1, 1, 2, 2, 4, 4], each block
    output added back to its input (residual). This matches the widely
    reported zju3dv/snake structure (state_dim=128 channels throughout).

  - Fusion block (Fig. 3a): concatenate the 8 backbone states channel-wise,
    1x1-conv to a 256-d "fusion" space, then max-pool over the vertex
    dimension to get one global contour descriptor, broadcast back to
    every vertex and concatenated with the per-vertex backbone states.

  - Prediction head (Fig. 3a): 3 pointwise (1x1) convs
    (in -> 256 -> 64 -> 2), ReLU between, producing a per-vertex 2-D
    offset that is added to the current contour to deform it.

  - Two-stage pipeline (Fig. 3b): the paper proposes box -> diamond ->
    (deep snake) -> extreme points -> octagon -> (deep snake, iterated)
    -> final contour. We keep the *iterative contour deformation* idea
    (several independent Snake instances applied one after another, each
    seeing the previous stage's deformed contour) but replace the
    detector-driven "diamond from box" initial proposal with a learned
    initial-ellipse proposal, since this codebase has no detector/box
    stage (see ADAPTATION below).

ADAPTATION (deviations from the paper, made to fit this project):
  1. Initial contour proposal. The paper gets its initial contour from a
     detector's bounding box (box -> diamond -> extreme points -> octagon).
     This project's baselines are single-object, encoder-only (no
     detector), so the initial contour here is instead a learned ellipse:
     global-average-pool the coarsest pyramid scale (s32), regress a
     center (cx, cy) and semi-axes (rx, ry) with a tiny MLP, and sample
     N_INIT points evenly spaced by angle on that ellipse. This is the
     same "coarse box-like proposal -> refine" spirit as the paper's
     diamond/octagon stages, just regressed from features instead of a
     detector.
  2. Feature map sampled by the snake. The paper samples a single CNN
     feature map (its detector backbone's output). Here the encoder gives
     a 4-scale pyramid, so all 4 scales are bilinearly upsampled to the
     stride-4 grid, concatenated, and 1x1-conv'd down to a 64-channel map
     -- this is the map every Snake stage samples via grid_sample at the
     current vertex positions (bilinear, matching the paper's feature
     sampling).
  3. Rasterization to logits. Deep Snake's native output IS the polygon
     (used directly for the mAP metric via its vertices). This project's
     PyramidDecoder contract requires per-pixel logits, so the final
     deformed contour is rendered to a soft mask with a differentiable
     signed-distance rasterizer (winding number for inside/outside,
     min-distance-to-edge for softness) -- this rasterization is this
     file's own addition, not part of the Deep Snake paper, and exists
     purely to satisfy forward(feats) -> logits.
  4. Iterative weight sharing. The original paper uses two Snake
     networks: one for diamond-to-extreme-point regression and one for
     iterative contour deformation. The deformation Snake is reused with
     shared weights for all three inference iterations. Because this
     codebase has no detector or bounding-box stage, the first network is
     omitted together with the diamond/octagon pipeline. The learned ellipse
     replaces that proposal stage, while the remaining deformation stage
     uses one shared Snake module repeatedly. This is intentionally
     different from stacking independently parameterized Snake modules.

  5. Vertex coordinates. Deep Snake concatenates sampled image features
     with translations-invariant contour coordinates. We therefore
     subtract the per-contour minimum from every coordinate before passing
     them to the Snake.

  6. Rasterization. The paper optimizes contour vertices directly with a
     smooth-L1 loss. The differentiable polygon rasterizer below is an
     adaptation required by this project's pixel-mask decoder interface;
     it is not part of the original Deep Snake architecture.
"""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from .base import PyramidDecoder


# --------------------------------------------------------------------------- #
# Circular 1-D convolution building block
# --------------------------------------------------------------------------- #
class BasicBlock(nn.Module):
    """CirConv -> BN -> ReLU. Conv1d with padding_mode='circular' gives the
    wrap-around ("cycle graph") padding Deep Snake uses so the first and
    last contour vertices are neighbours, exactly like every other
    adjacent pair. kernel_size = 2*n_adj + 1 (n_adj=4 -> kernel 9, matching
    the paper's stated CirConv kernel size)."""

    def __init__(self, in_dim, out_dim, n_adj=4, dilation=1):
        super().__init__()
        kernel_size = 2 * n_adj + 1
        # padding_mode='circular' needs padding = dilation * (kernel_size // 2)
        # on each side to keep the sequence length unchanged.
        pad = dilation * (kernel_size // 2)
        self.conv = nn.Conv1d(in_dim, out_dim, kernel_size=kernel_size,
                               dilation=dilation, padding=pad, padding_mode="circular")
        self.bn = nn.BatchNorm1d(out_dim)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        return self.relu(self.bn(self.conv(x)))


class Snake(nn.Module):
    """One full Deep Snake module: backbone (8 CirConv-BN-ReLU layers, 7 of
    them residual) -> fusion (concat + 1x1 conv + global max-pool) ->
    prediction head (3 pointwise convs) -> per-vertex 2-D offset.

    Input:  x of shape [B, feature_dim, N]  (N = number of contour points)
    Output: offsets of shape [B, 2, N]
    """

    DILATIONS = [1, 1, 1, 2, 2, 4, 4]  # 7 residual layers

    def __init__(self, feature_dim, state_dim=128, fusion_state_dim=256):
        super().__init__()
        self.head = BasicBlock(feature_dim, state_dim, dilation=1)
        self.res_layers = nn.ModuleList([
            BasicBlock(state_dim, state_dim, dilation=d) for d in self.DILATIONS
        ])
        n_states = 1 + len(self.res_layers)  # head + 7 residual = 8 states
        self.fusion = nn.Conv1d(state_dim * n_states, fusion_state_dim, kernel_size=1)
        self.prediction = nn.Sequential(
            nn.Conv1d(state_dim * n_states + fusion_state_dim, 256, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Conv1d(256, 64, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Conv1d(64, 2, kernel_size=1),
        )

    def forward(self, x):
        states = []
        x = self.head(x)
        states.append(x)
        for layer in self.res_layers:
            x = layer(x) + x  # residual skip connection, every layer
            states.append(x)

        state = torch.cat(states, dim=1)                      # [B, state_dim*8, N]
        global_state = torch.max(self.fusion(state), dim=2, keepdim=True)[0]  # [B, 256, 1]
        global_state = global_state.expand(-1, -1, state.size(2))
        state = torch.cat([global_state, state], dim=1)
        return self.prediction(state)                         # [B, 2, N]


# --------------------------------------------------------------------------- #
# Learned initial-ellipse proposal (ADAPTATION #1 -- replaces the paper's
# detector-box -> diamond stage, see module docstring)
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
        center = 0.5 * torch.stack(
            [torch.tanh(cx), torch.tanh(cy)], dim=-1
        )
        radius = torch.stack(
            [torch.sigmoid(rx), torch.sigmoid(ry)], dim=-1
        ) * 0.45 + 0.03
        return center, radius  # each [B, 2]


def make_initial_contour(center, radius, n_points, device):
    """Evenly-spaced ellipse in normalized [-1, 1] coords. center/radius: [B, 2]."""
    theta = torch.linspace(0, 2 * math.pi, n_points + 1, device=device)[:-1]  # [N]
    unit = torch.stack([torch.cos(theta), torch.sin(theta)], dim=-1)          # [N, 2]
    contour = center.unsqueeze(1) + radius.unsqueeze(1) * unit.unsqueeze(0)   # [B, N, 2]
    return contour.clamp(-1.0, 1.0)


# --------------------------------------------------------------------------- #
# Feature sampling at contour vertices (bilinear, matches the paper's
# per-vertex feature lookup on the shared CNN feature map)
# --------------------------------------------------------------------------- #
def sample_features(feat_map, contour):
    """feat_map: [B, C, H, W] in stride-4 space. contour: [B, N, 2] in
    normalized [-1, 1] (x, y). Returns [B, C, N]."""
    grid = contour.unsqueeze(2)  # [B, N, 1, 2], grid_sample expects (x, y)
    sampled = F.grid_sample(feat_map, grid, mode="bilinear",
                             padding_mode="border", align_corners=False)
    return sampled.squeeze(-1)  # [B, C, N]


# --------------------------------------------------------------------------- #
# Differentiable polygon -> mask rasterization (this file's own addition --
# NOT part of the Deep Snake paper, needed only to satisfy the shared
# PyramidDecoder contract of returning per-pixel logits; see ADAPTATION #3)
# --------------------------------------------------------------------------- #
def rasterize_polygon(contour, height, width):
    """contour: [B, N, 2] normalized [-1, 1] (x, y), closed polygon (implicit
    wrap N-1 -> 0). Returns logits [B, 1, height, width]: positive inside,
    negative outside, magnitude ~ distance to the boundary / temperature."""
    B, N, _ = contour.shape
    device = contour.device

    ys = torch.linspace(-1.0, 1.0, height, device=device)
    xs = torch.linspace(-1.0, 1.0, width, device=device)
    grid_y, grid_x = torch.meshgrid(ys, xs, indexing="ij")
    pix = torch.stack([grid_x, grid_y], dim=-1).view(1, height * width, 1, 2)  # [1, HW, 1, 2]

    v0 = contour.unsqueeze(1)                                   # [B, 1, N, 2]
    v1 = torch.roll(contour, shifts=-1, dims=1).unsqueeze(1)    # [B, 1, N, 2]

    edge = v1 - v0                                              # [B, 1, N, 2]
    to_pix = pix - v0                                           # [B, HW, N, 2]
    edge_len2 = (edge * edge).sum(-1).clamp_min(1e-8)           # [B, 1, N]
    t = (to_pix * edge).sum(-1) / edge_len2                     # [B, HW, N]
    t = t.clamp(0.0, 1.0)
    proj = v0 + t.unsqueeze(-1) * edge                          # [B, HW, N, 2]
    dist_per_edge = (pix - proj).norm(dim=-1)                   # [B, HW, N]
    min_dist = dist_per_edge.min(dim=-1)[0]                     # [B, HW]

    # Winding number (differentiable, but only its SIGN is used -- the
    # magnitude signal for the mask edge comes from min_dist above).
    d0 = pix - v0                                               # [B, HW, N, 2]
    d1 = pix - v1                                               # [B, HW, N, 2]
    cross = d0[..., 0] * d1[..., 1] - d0[..., 1] * d1[..., 0]
    dot = (d0 * d1).sum(-1)
    angle = torch.atan2(cross, dot)                             # [B, HW, N]
    winding = angle.sum(-1) / (2 * math.pi)                     # [B, HW]

    # Replace the hard, non-differentiable inside/outside sign.
    winding_temperature = 0.12
    soft_inside = torch.sigmoid(
        (winding.abs() - 0.5) / winding_temperature
    )
    soft_sign = soft_inside.mul(2.0).sub(1.0)

    # Keep logits in a useful BCE range during early training.
    tau = 0.80
    logits = (soft_sign * min_dist / tau).clamp(-12.0, 12.0)
    return logits.view(B, 1, height, width)


# --------------------------------------------------------------------------- #
# Full decoder
# --------------------------------------------------------------------------- #
class DeepSnakeDecoder(PyramidDecoder):
    """Direct polygon regression via iterative contour deformation
    (Deep Snake). Ignores `decoder_dim` for the snake's internal channel
    width (fixed at the paper's state_dim=128 for faithfulness) but uses it
    to size the pyramid-fusion feature map so it still responds to
    --decoder_dim like every other decoder in this project.

    n_points:  vertices per contour (paper uses ~128 for octagon-refined
               contours; kept configurable since this project's contour
               datasets already parameterize vertex count as n_points).
    n_iter:    number of chained Snake stages after the initial proposal
               (paper's ablation table sweeps "Iter. 1" .. "Iter. 5";
               3 is a reasonable default matching the released config).
    """

    def __init__(self, enc_channels, decoder_dim=64, num_classes=1,
                 n_points=128, n_iter=3, snake_state_dim=128):
        super().__init__(enc_channels, decoder_dim, num_classes)
        assert num_classes == 1, "DeepSnakeDecoder renders a single closed contour (binary mask)."
        c4, c8, c16, c32 = enc_channels
        self.n_points = n_points
        self.n_iter = n_iter

        # ADAPTATION #2: fuse the 4-scale pyramid into one stride-4 feature
        # map the snakes sample from (paper samples a single backbone map).
        fused_dim = max(decoder_dim, 32)
        self.pyramid_fuse = nn.Sequential(
            nn.Conv2d(c4 + c8 + c16 + c32, fused_dim, kernel_size=1),
            nn.BatchNorm2d(fused_dim), nn.ReLU(inplace=True),
        )

        self.init_head = InitialEllipseHead(c32)
        feature_dim = fused_dim + 2  # sampled image feature + (x, y) coords, per vertex
        self.snake = Snake(
            feature_dim,
            state_dim=snake_state_dim,
        )

    def _fuse_pyramid(self, feats):
        s4, s8, s16, s32 = feats
        target = s4.shape[-2:]
        up = [s4] + [F.interpolate(s, size=target, mode="bilinear", align_corners=False)
                     for s in (s8, s16, s32)]
        return self.pyramid_fuse(torch.cat(up, dim=1))  # [B, fused_dim, H4, W4]

    def forward(self, feats):
        s4, s8, s16, s32 = feats
        B = s4.shape[0]
        H4, W4 = s4.shape[-2:]

        feat_map = self._fuse_pyramid(feats)              # [B, fused_dim, H4, W4]
        center, radius = self.init_head(s32)
        contour = make_initial_contour(center, radius, self.n_points, s4.device)  # [B, N, 2]

        for _ in range(self.n_iter):
            sampled = sample_features(feat_map, contour)

            # Deep Snake verwendet translationsinvariante Vertex-Koordinaten:
            # von jeder Koordinate wird das Minimum der Kontur abgezogen.
            relative_coords = (
                contour - contour.amin(dim=1, keepdim=True)
            ).transpose(1, 2)

            snake_in = torch.cat(
                [sampled, relative_coords],
                dim=1,
            )

            offset = 0.05 * torch.tanh(
                self.snake(snake_in)
            ).transpose(1, 2)

            contour = torch.tanh(
                torch.atanh(contour.clamp(-0.98, 0.98)) + offset
            )

        logits = rasterize_polygon(contour, H4, W4)        # [B, 1, H4, W4], stride 4
        return logits