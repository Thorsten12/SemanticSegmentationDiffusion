"""Decoder #6 (Group B) -- G-CASCADE.
Reference: Rahman & Marculescu, "G-CASCADE: Efficient Cascaded Graph
Convolutional Decoding for 2D Medical Image Segmentation", WACV 2024.
Paper: https://openaccess.thecvf.com/content/WACV2024/papers/Rahman_G-CASCADE_Efficient_Cascaded_Graph_Convolutional_Decoding_for_2D_Medical_Image_WACV_2024_paper.pdf

Faithful reimplementation of Section 3.1 (Eq. 1-6) and Section 3.1.2 (UCB,
Eq. 5), NOT a copy of the official repo (github.com/SLDGroup/G-CASCADE,
whose raw files are blocked from automated fetching). Traceable to:

  - GCAM, Eq. 1:  GCAM(x) = SPA(GCB(x))  -- graph conv block THEN spatial
    attention (Table 4 in the paper: GCB->SPA beats SPA->GCB).

  - GCB, Eq. 2-3 (the "Grapher" design from Vision GNN / ViG, Han et al.
    2022): GCB(x) = R(BN(C(GConv(R(BN(C(x)))))))
    i.e. 1x1 conv+BN+ReLU -> graph conv -> 1x1 conv+BN+ReLU, with a
    residual add around the whole block (standard Grapher design, the
    paper's Eq. 2 omits the residual symbol but the Grapher module it
    cites always includes it -- we keep it for training stability).
    GConv, Eq. 3: GConv(x) = GELU(BN(DynConv(x))), where DynConv is
    max-relative graph convolution over K=11 neighbors (Section 4.3,
    "Implementation details": "We construct dense dilated graph using
    K = 11 neighbors for KNN and use the Max-Relative (MR) graph
    convolution"). NOTE on "dilated": the paper builds its KNN graph the
    way ViG's *dilated* KNN does -- sampling every d-th neighbor from a
    larger candidate pool (dilation rate d) rather than always taking the
    true K nearest, specifically to increase receptive-field diversity
    and reduce over-smoothing across stacked graph-conv layers. Our
    `MaxRelativeGraphConv` below does NOT implement that dilation step;
    it computes a plain dense K-NN (true K nearest neighbors by Euclidean
    distance, K=11, no dilation sampling). This is a real simplification
    vs. the paper, not just "no custom CUDA kernel" -- flagged explicitly
    so it isn't mistaken for a faithful reproduction of Eq. 3's graph
    construction.

  - SPA, Eq. 4: SPA(x) = Sigmoid(Conv7x7([max_c(x), avg_c(x)])) * x
    (channel-max and channel-avg CONCATENATED then conv -- same as
    standard CBAM spatial attention, unlike CASCADE's SA which sums).

  - UCB (efficient up-conv), Eq. 5: UCB(x) = Conv1x1(ReLU(BN(DWConv3x3(
    Upsample_x2(x))))) -- upsample, then a depthwise 3x3 (not full 3x3,
    this is the "efficient" part vs. CASCADE's UpConv), BN, ReLU, 1x1.

  - SegHead, Eq. 6: a single 1x1 conv per stage -> num_classes channels.

  - Multi-stage output aggregation, Eq. 7: 4 prediction heads, upsampled
    and SUMMED with equal weights (same pattern as CASCADE's Eq. 8). As
    with cascade.py, our shared trainer only supervises the single final
    sum (no MUTATION multi-stage loss, Section 3.3) -- architectural
    aggregation is faithful, per-stage loss weighting is not reproduced.

  - Skip aggregation: the paper compares additive vs. concat aggregation
    of the upsampled features with skip connections (Table 6) and uses
    ADDITIVE in all main experiments (lower FLOPs/params, only marginally
    worse DICE) -- we follow that choice here.

Graph convolution implementation note: a literal K-NN graph conv (as in
ViG) operates on a token sequence and is relatively expensive to build a
fresh k-NN graph for at every forward pass over full-resolution feature
maps. We implement the standard "max-relative" graph convolution exactly
as ViG defines it (DynConv = 1x1 conv over [x, max_j(x_j - x_i)] for each
node's k nearest neighbors in feature space), computed via native PyTorch
k-NN (dense pairwise distances) over the flattened spatial tokens --
functionally equivalent to ViG's Grapher for the "which neighbors, what
message" part, but WITHOUT dilated sampling (see note above) and without
a custom CUDA kernel. At high spatial resolutions (e.g. the finest, stride-4
stage on large inputs) this dense NxN distance computation can become a
real memory/compute bottleneck since it's O(N^2); worth watching if this
decoder is applied to bigger input resolutions than the paper used
(224x224 / 384x384).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from .base import PyramidDecoder


class MaxRelativeGraphConv(nn.Module):
    """DynConv, Eq. 3's inner op: max-relative graph convolution over a
    dense K-NN graph in feature space (Han et al., Vision GNN, 2022).
    For each spatial location (node) i, find its K nearest neighbors j in
    feature space, compute the max over j of (x_j - x_i), concat with x_i,
    and project with a 1x1 conv. This is graph convolution's "message
    passing" step condensed into one efficient operation.

    K defaults to 11 to match the paper's Section 4.3 implementation
    detail ("K = 11 neighbors for KNN"). Note this is a plain dense K-NN,
    not the paper's dilated K-NN -- see the module-level docstring."""

    def __init__(self, channels, k=11):
        super().__init__()
        self.k = k
        self.proj = nn.Conv2d(channels * 2, channels, kernel_size=1)

    def forward(self, x):
        B, C, H, W = x.shape
        N = H * W
        tokens = x.flatten(2).transpose(1, 2)  # [B, N, C]
        k = min(self.k, N - 1) if N > 1 else 1

        with torch.no_grad():
            dist = torch.cdist(tokens, tokens)
            dist.diagonal(dim1=1, dim2=2).fill_(float("inf"))
            knn_idx = dist.topk(k, dim=-1, largest=False).indices  # [B, N, k]

        # Batch-Offset addieren, dann auf geflatteter (B*N) Token-Liste indizieren
        batch_offset = torch.arange(B, device=x.device).view(B, 1, 1) * N
        flat_idx = (knn_idx + batch_offset).reshape(-1)          # [B*N*k]
        flat_tokens = tokens.reshape(B * N, C)                    # [B*N, C]
        neighbors = flat_tokens[flat_idx].view(B, N, k, C)        # [B, N, k, C] -- viel kleiner als [B,N,N,C]

        relative = neighbors - tokens.unsqueeze(2)
        max_relative = relative.max(dim=2).values
        fused = torch.cat([tokens, max_relative], dim=-1)
        fused = fused.transpose(1, 2).reshape(B, 2 * C, H, W)
        return self.proj(fused)

class GraphConvBlock(nn.Module):
    """GConv, Eq. 3: GConv(x) = GELU(BN(DynConv(x)))."""

    def __init__(self, channels, k=11):
        super().__init__()
        self.dynconv = MaxRelativeGraphConv(channels, k=k)
        self.bn = nn.BatchNorm2d(channels)

    def forward(self, x):
        return F.gelu(self.bn(self.dynconv(x)))


class GCB(nn.Module):
    """Full Grapher-style block, Eq. 2:
        GCB(x) = R(BN(C(GConv(R(BN(C(x)))))))
    1x1conv+BN+ReLU -> GConv (graph conv) -> 1x1conv+BN+ReLU, with a
    residual connection around the whole block (standard in the Grapher
    design this equation is based on)."""

    def __init__(self, channels, k=11):
        super().__init__()
        self.pre = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=1),
            nn.BatchNorm2d(channels),
        )
        self.gconv = GraphConvBlock(channels, k=k)
        self.post = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=1),
            nn.BatchNorm2d(channels),
        )

    def forward(self, x):
        residual = x
        x = F.relu(self.pre(x), inplace=True)
        x = self.gconv(x)
        x = F.relu(self.post(x), inplace=True)
        return x + residual


class SpatialAttention(nn.Module):
    """SPA, Eq. 4: Sigmoid(Conv7x7([channel_max(x), channel_avg(x)])) * x.
    Concat (not sum, unlike CASCADE's SA) of channel-max and channel-avg."""

    def __init__(self, kernel_size=7):
        super().__init__()
        self.conv = nn.Conv2d(2, 1, kernel_size=kernel_size, padding=kernel_size // 2)

    def forward(self, x):
        avg = x.mean(dim=1, keepdim=True)
        mx, _ = x.max(dim=1, keepdim=True)
        gate = torch.sigmoid(self.conv(torch.cat([mx, avg], dim=1)))
        return x * gate


class GCAM(nn.Module):
    """Eq. 1: GCAM(x) = SPA(GCB(x))."""

    def __init__(self, channels, k=11):
        super().__init__()
        self.gcb = GCB(channels, k=k)
        self.spa = SpatialAttention()

    def forward(self, x):
        return self.spa(self.gcb(x))


class UCB(nn.Module):
    """Eq. 5: UCB(x) = Conv1x1(ReLU(BN(DWConv3x3(Upsample_x2(x))))).
    The 'efficient' up-conv: a depthwise (not full) 3x3 conv after
    upsampling, then a 1x1 to mix channels -- far fewer FLOPs than
    CASCADE's full 3x3 UpConv."""

    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.dwconv = nn.Conv2d(in_ch, in_ch, kernel_size=3, padding=1, groups=in_ch)
        self.bn = nn.BatchNorm2d(in_ch)
        self.pwconv = nn.Conv2d(in_ch, out_ch, kernel_size=1)

    def forward(self, x, size=None):
        if size is not None:
            x = F.interpolate(x, size=size, mode="bilinear", align_corners=False)
        else:
            x = F.interpolate(x, scale_factor=2, mode="bilinear", align_corners=False)
        x = F.relu(self.bn(self.dwconv(x)), inplace=True)
        return self.pwconv(x)


class GCascadeStage(nn.Module):
    """One decoder stage: UCB the previous stage's output to the skip's
    resolution, ADDITIVELY aggregate with the skip connection (paper's
    Table 6 choice: additive over concat, far cheaper for similar DICE),
    then refine with a GCAM."""

    def __init__(self, in_channels, skip_channels, stage_channels, k=11):
        super().__init__()
        self.ucb = UCB(in_channels, stage_channels)
        self.skip_proj = nn.Conv2d(skip_channels, stage_channels, kernel_size=1)
        self.gcam = GCAM(stage_channels, k=k)

    def forward(self, x, skip):
        g = self.ucb(x, size=skip.shape[-2:])
        fused = g + self.skip_proj(skip)  # additive aggregation (Table 6)
        return self.gcam(fused)


class GCascadeDecoder(PyramidDecoder):
    """Full G-CASCADE decoder: GCAM on the bottleneck (X4/s32, no skip),
    then 3 GCascadeStage blocks (UCB + additive skip + GCAM) over
    s16/s8/s4, with a 1x1 SegHead at every stage. All 4 stage predictions
    are upsampled to stride 4 and summed (Eq. 7, equal weights)."""

    def __init__(self, enc_channels, decoder_dim=64, num_classes=1, knn_k=11):
        super().__init__(enc_channels, decoder_dim, num_classes)
        c4, c8, c16, c32 = enc_channels

        # Paper uses reduction ratios [1, 1, 4, 2] across stages for the
        # graph conv's channel width -- we approximate this by scaling
        # decoder_dim per stage similarly to our other cascaded decoders,
        # narrower toward the finest scale.
        d32, d16, d8, d4 = decoder_dim * 8, decoder_dim * 4, decoder_dim * 2, decoder_dim

        self.bottleneck_gcam = nn.Sequential(
            nn.Conv2d(c32, d32, kernel_size=1),
            GCAM(d32, k=knn_k),
        )

        self.stage16 = GCascadeStage(d32, c16, d16, k=knn_k)
        self.stage8 = GCascadeStage(d16, c8, d8, k=knn_k)
        self.stage4 = GCascadeStage(d8, c4, d4, k=knn_k)

        self.head32 = nn.Conv2d(d32, num_classes, kernel_size=1)
        self.head16 = nn.Conv2d(d16, num_classes, kernel_size=1)
        self.head8 = nn.Conv2d(d8, num_classes, kernel_size=1)
        self.head4 = nn.Conv2d(d4, num_classes, kernel_size=1)

    def forward(self, feats):
        s4, s8, s16, s32 = feats
        target_size = s4.shape[-2:]

        d32 = self.bottleneck_gcam(s32)
        d16 = self.stage16(d32, s16)
        d8 = self.stage8(d16, s8)
        d4 = self.stage4(d8, s4)

        p32 = F.interpolate(self.head32(d32), size=target_size, mode="bilinear", align_corners=False)
        p16 = F.interpolate(self.head16(d16), size=target_size, mode="bilinear", align_corners=False)
        p8 = F.interpolate(self.head8(d8), size=target_size, mode="bilinear", align_corners=False)
        p4 = self.head4(d4)

        return p32 + p16 + p8 + p4  # logits at stride 4; central resize handles the rest
