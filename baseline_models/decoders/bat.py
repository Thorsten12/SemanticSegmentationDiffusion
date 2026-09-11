"""Decoder -- BAT (Boundary-Aware Transformer), architecture-faithful rewrite.
Reference: Wang, Wei, Wang, Zhou, Zhu, Qin, "Boundary-Aware Transformers for
Skin Lesion Segmentation", MICCAI 2021. Paper: https://arxiv.org/abs/2110.03864
Official code: https://github.com/jcwang123/BA-Transformer

This supersedes the previous version of this file. That version invented a
multi-resolution U-Net-style decoder pyramid with a "Local-Global Transformer"
(a top-k boundary-token sub-branch, scatter-added back into a global branch)
and derived "boundary-ness" from how close the segmentation probability was
to 0.5. None of that exists in the paper. This rewrite follows Sec. 2 of the
paper directly:

  - Image sequentialization (Sec. 2.1): the paper takes ONE CNN feature map
    (ResNet50 at stride 16) and flattens it to patch tokens, adding a
    learnable positional embedding.
  - Sequence transformation (Eq. 1) + BAG (Eq. 3): n stacked transformer
    encoder layers (MSA + MLP), each immediately followed by a Boundary-wise
    Attention Gate. BAG is a genuine 1x1-conv boundary classifier
    M_hat = sigmoid(d_1^1(Z)) -- NOT a derived uncertainty score -- fed back
    as a residual gate: Z^i = V^{i-1} + V^{i-1} * M_hat^{i-1}.
  - Query-embedding BAG (Sec. 2.2, second paragraph): one more BAG variant
    applied once after the encoder stack, using a learnable prototype
    embedding Q_b compared against every token instead of a conv classifier,
    same residual-gate formula.
  - Atrous prediction (Eq. 2): three parallel dilated 3x3 convs (rates
    1, 3, 6), concatenated channel-wise, projected by a 1x1 conv.

There is deliberately no multi-scale decoder pyramid, no skip-fusion, and no
extra local-attention sub-branch -- the paper doesn't have any of these.

ADAPTATION REQUIRED FOR THIS CODEBASE'S ENCODERS (the only real deviation
left, and it concerns *which feature map goes in*, not what happens to it):

  - This project's conditioners (ConvNeXt/PVT/ResNet/Swin/VMamba, see
    models/conditioners.py) always return a list of feature maps in fixed
    stride order [ (optional stride-1 stem), stride4, stride8, stride16,
    stride32 ]. The paper needs exactly one map -- the stride-16 one,
    matching ResNet50's stage-3 output the original authors used -- so
    `forward()` picks `feats[-2]` (second-to-last is always stride16,
    regardless of whether a stem is prepended) and ignores every other
    pyramid level. This mirrors the paper's choice of ResNet50-stride16 as
    the sequentialization source; we just let that source be whichever
    backbone the experiment is configured with, instead of hard-coding
    ResNet50.
  - Channel count C at stride 16 differs per backbone (1024 for ResNet50,
    320 for PVTv2-b2, 384 for ConvNeXt-tiny/Swin-tiny/VMamba-tiny, ...).
    The paper doesn't add a projection layer -- it just uses whatever width
    ResNet50-stride16 has as C -- so we do the same: d_model = C dynamically,
    rather than forcing a fixed decoder_dim like the previous file did.
    n_heads must divide C for every backbone in this codebase; 8 does
    (320/8=40, 384/8=48, 1024/8=128, 768/8=96), so that is the default.
  - Positional embedding: stored at a reference grid size and bilinearly
    resized to the actual (H, W) at forward time (standard ViT-style
    resize), since crop size can differ between train/eval in this project.
  - Output resolution: the paper predicts directly at the sequentialization
    resolution and relies on the surrounding pipeline to upsample. To stay
    drop-in with this codebase's decoder contract, the atrous-prediction
    output is bilinearly upsampled from stride 16 to the spatial size of
    `feats[0]` (the finest available pyramid entry) before being returned.
    This is a resize only, not an architectural change.

NOT reproduced (training-loop / data-pipeline concerns, out of scope for a
decoder module):

  - The auxiliary boundary-map supervision (Eq. 4's sum of L_Map^i terms)
    needs ground-truth key-patch maps produced by the paper's edge-detection
    + NMS algorithm (Sec. 2.2, "Boundary-Supervised Generator") -- a
    data-preprocessing step, not something a decoder module can synthesize.
    So that adding it later doesn't require another architecture rewrite,
    `forward(..., return_boundary_maps=True)` optionally returns the list of
    predicted boundary maps, one per BAG (n encoder-layer BAGs + 1
    query-embedding BAG) -- exactly the {M_hat^1 ... M_hat^{n+1}} the paper
    supervises in Eq. 4. Wiring up the actual loss is a training-loop
    decision.
  - The paper's atrous module ends in an internal sigmoid producing S_pred
    directly; this decoder returns raw logits instead, like every other
    decoder in this codebase (BCE-with-logits + Dice is applied externally).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from .base import PyramidDecoder


# --------------------------------------------------------------------------- #
# Atrous Prediction (Eq. 2)
# --------------------------------------------------------------------------- #
class DilatedConvBranch(nn.Module):
    """d_r^s(.) -- a single dilated conv branch used inside atrous prediction."""

    def __init__(self, channels, rate, kernel_size=3):
        super().__init__()
        padding = rate * (kernel_size - 1) // 2
        self.conv = nn.Conv2d(channels, channels, kernel_size, padding=padding, dilation=rate)

    def forward(self, x):
        return self.conv(x)


class AtrousPrediction(nn.Module):
    """Eq. 2: S_pred = sigmoid(d_1^1([d_1^3(Z), d_3^3(Z), d_6^3(Z)])).
    Sigmoid dropped in favor of raw logits (see module docstring); the
    dilated-branch-concat-then-project structure itself is unchanged."""

    def __init__(self, channels, num_classes):
        super().__init__()
        self.branch_r1 = DilatedConvBranch(channels, rate=1)
        self.branch_r3 = DilatedConvBranch(channels, rate=3)
        self.branch_r6 = DilatedConvBranch(channels, rate=6)
        self.project = nn.Conv2d(channels * 3, num_classes, kernel_size=1)

    def forward(self, z):
        feats = torch.cat([self.branch_r1(z), self.branch_r3(z), self.branch_r6(z)], dim=1)
        return self.project(feats)


# --------------------------------------------------------------------------- #
# Boundary-wise Attention Gate variants (Eq. 3 + Sec. 2.2 query-embedding BAG)
# --------------------------------------------------------------------------- #
class BoundaryWiseAttentionGate(nn.Module):
    """Key-patch map generator + residual attention scheme (Eq. 3):
        M_hat = sigmoid(d_1^1(Z))                      (1x1 conv classifier)
        Z^i   = V^{i-1} + V^{i-1} * M_hat^{i-1}         (residual gate)
    A genuine per-patch boundary classifier -- not derived from segmentation
    probability. Returns the gated tokens and the raw boundary map (for
    optional auxiliary supervision, see module docstring)."""

    def __init__(self, channels):
        super().__init__()
        self.key_patch_head = nn.Conv2d(channels, 1, kernel_size=1)

    def forward(self, v, h, w):
        # v: [B, N, C], the already MSA+MLP'd tokens (V^{i-1} in Eq. 3)
        b, n, c = v.shape
        v_2d = v.transpose(1, 2).reshape(b, c, h, w)
        m_hat = torch.sigmoid(self.key_patch_head(v_2d))   # [B, 1, H, W]
        m_flat = m_hat.flatten(2).transpose(1, 2)            # [B, N, 1]
        z = v + v * m_flat
        return z, m_hat


class QueryEmbeddingBAG(nn.Module):
    """Second BAG variant (Sec. 2.2, applied once after the encoder stack):
    a learnable prototype embedding Q_b compared against every patch to
    produce a similarity-based boundary map, then the same residual gate."""

    def __init__(self, channels):
        super().__init__()
        self.query = nn.Parameter(torch.randn(1, 1, channels) * 0.02)

    def forward(self, z, h, w):
        b, n, c = z.shape
        q = self.query.expand(b, -1, -1)                     # [B, 1, C]
        sim = F.cosine_similarity(z, q, dim=-1).unsqueeze(-1)  # [B, N, 1] in [-1, 1]
        m_hat = torch.sigmoid(sim)
        out = z + z * m_hat
        m_2d = m_hat.transpose(1, 2).reshape(b, 1, h, w)
        return out, m_2d


# --------------------------------------------------------------------------- #
# Sequence transformation (Eq. 1) + BAG, one per encoder layer
# --------------------------------------------------------------------------- #
class BATEncoderLayer(nn.Module):
    """Eq. 1's transformer encoder layer, immediately followed by Eq. 3's
    BAG -- exactly the paper's per-layer unit, no extra sub-branches."""

    def __init__(self, channels, n_heads, mlp_ratio=4, dropout=0.0):
        super().__init__()
        self.block = nn.TransformerEncoderLayer(
            d_model=channels, nhead=n_heads, dim_feedforward=channels * mlp_ratio,
            dropout=dropout, batch_first=True, norm_first=True,
        )
        self.bag = BoundaryWiseAttentionGate(channels)

    def forward(self, z, h, w):
        v = self.block(z)                  # Eq. 1
        z_next, m_hat = self.bag(v, h, w)   # Eq. 3
        return z_next, m_hat


# --------------------------------------------------------------------------- #
# Full decoder
# --------------------------------------------------------------------------- #
class BATDecoder(PyramidDecoder):
    """Single-scale Boundary-Aware Transformer decoder, matching the paper's
    architecture (Sec. 2). Operates only on the stride-16 pyramid entry; see
    module docstring for how that entry is selected from this codebase's
    multi-scale encoders and why no other architectural change was needed.

    decoder_dim is accepted for PyramidDecoder interface compatibility but
    unused: the paper has no projection layer, so d_model is fixed to
    whatever channel width the stride-16 feature map actually has.
    """

    def __init__(self, enc_channels, decoder_dim=64, num_classes=1,
                 n_heads=8, n_layers=4, mlp_ratio=4, pos_embed_grid=16,
                 dropout=0.0):
        super().__init__(enc_channels, decoder_dim, num_classes)

        # enc_channels is the full pyramid's channel list (with or without a
        # prepended stem channel count) -- stride16 is always second-to-last.
        channels = enc_channels[-2]
        if channels % n_heads != 0:
            raise ValueError(
                f"stride-16 channel width {channels} is not divisible by "
                f"n_heads={n_heads}; pick an n_heads that divides it -- the "
                f"paper doesn't project channels, so this decoder doesn't "
                f"either."
            )

        self.channels = channels
        self.n_layers = n_layers

        # Learnable positional embedding at a reference grid size, resized
        # to the actual (H, W) at forward time (paper: "a learnable
        # positional embedding which is randomly initialized").
        self.pos_embed = nn.Parameter(torch.randn(1, channels, pos_embed_grid, pos_embed_grid) * 0.02)

        self.layers = nn.ModuleList([
            BATEncoderLayer(channels, n_heads=n_heads, mlp_ratio=mlp_ratio, dropout=dropout)
            for _ in range(n_layers)
        ])
        self.query_bag = QueryEmbeddingBAG(channels)
        self.atrous_head = AtrousPrediction(channels, num_classes)

    def _add_pos_embed(self, x):
        h, w = x.shape[-2:]
        pos = F.interpolate(self.pos_embed, size=(h, w), mode="bilinear", align_corners=False)
        return x + pos

    def forward(self, feats, return_boundary_maps=False):
        # Full pyramid comes in (PyramidDecoder contract, and to know the
        # output resolution), but only the stride-16 entry is ever touched --
        # matching the paper's single-scale design.
        s16 = feats[-2]
        target_size = feats[0].shape[-2:]  # finest available scale

        x = self._add_pos_embed(s16)                # sequentialization + pos embed
        b, c, h, w = x.shape
        z = x.flatten(2).transpose(1, 2)             # [B, N, C], N = H*W

        boundary_maps = []
        for layer in self.layers:
            z, m_hat = layer(z, h, w)
            boundary_maps.append(m_hat)

        z, m_final = self.query_bag(z, h, w)          # post-encoder query-embedding BAG
        boundary_maps.append(m_final)

        z_2d = z.transpose(1, 2).reshape(b, c, h, w)
        logits = self.atrous_head(z_2d)
        logits = F.interpolate(logits, size=target_size, mode="bilinear", align_corners=False)

        if return_boundary_maps:
            return logits, boundary_maps
        return logits