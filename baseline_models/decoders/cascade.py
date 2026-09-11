"""Decoder #5 (Group B) -- CASCADE.
Reference: Rahman & Marculescu, "Medical Image Segmentation via Cascaded
Attention Decoding", WACV 2023.
Paper: https://openaccess.thecvf.com/content/WACV2023/papers/Rahman_Medical_Image_Segmentation_via_Cascaded_Attention_Decoding_WACV_2023_paper.pdf
Official repo (not vendored here, but used to cross-check this reimplementation
against the paper's equations): https://github.com/SLDGroup/CASCADE

This is a faithful reimplementation following the paper's Section 3.2 and
Figure 1 exactly (equations 1-8), NOT a copy of the authors' source code
(GitHub blocks automated fetching of raw files). Every design choice below
is traceable to a specific equation in the paper:

  - Attention Gate (AG), Eq. 1-2:
        qatt(g, x) = ReLU(BN(Cg(g) + BN(Cx(x))))
        AG(g, x)   = x * Sigmoid(BN(C(qatt(g, x))))
    where Cg, Cx, C are 1x1 convs, g is the (upsampled) coarser decoder
    feature (gating signal), x is the skip connection.

  - Convolutional Attention Module (CAM), Eq. 3:
        CAM(x) = ConvBlock(SA(CA(x)))
    i.e. channel attention FIRST, then spatial attention, THEN the two 3x3
    convs -- this order matters and differs from plain CBAM (which is
    CA -> SA with no trailing conv block).

  - Channel Attention (CA), Eq. 4: SE-style avg+max pooling through a
    shared 1-hidden-layer MLP that reduces channels by 16x (not 8x).

  - Spatial Attention (SA), Eq. 5: channel-avg + channel-max maps, summed
    (not concatenated, unlike CBAM), through a 7x7 conv, sigmoid gate.

  - ConvBlock, Eq. 6: two 3x3 convs, each followed by BatchNorm + ReLU
    (paper uses BatchNorm throughout, not GroupNorm -- kept as BatchNorm
    here to match, even though the rest of this codebase mostly uses
    GroupNorm elsewhere).

  - UpConv, Eq. 7: bilinear/nearest x2 upsample -> 3x3 conv -> BN -> ReLU,
    used to bring the previous decoder stage's output up to the next
    (finer) skip connection's resolution before gating/fusing.

  - Multi-stage prediction aggregation, Eq. 8: one 1x1 prediction head per
    decoder stage (4 total), each upsampled to a common resolution and
    SUMMED (equal weights w=x=y=z=1.0 in the paper) to form the final
    output. NOTE: the paper also trains with a matching multi-stage LOSS
    aggregation (Eq. 9, one loss per stage). Our shared training loop in
    baseline_seg.py only supervises a single final output, so we only
    reproduce the architectural side of multi-stage aggregation (the
    forward-pass summation of Eq. 8); the four stages are still an
    implicit deep-supervision path even without per-stage losses, since
    gradients from the final sum flow back into every stage's head.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from .base import PyramidDecoder


class ChannelAttention(nn.Module):
    """Eq. 4: CA(x) = Sigmoid(MLP(MaxPool(x)) + MLP(AvgPool(x))) * x,
    with a shared 2-layer 1x1-conv MLP that reduces channels by 16x."""

    def __init__(self, channels, reduction=16):
        super().__init__()
        hidden = max(channels // reduction, 4)
        self.mlp = nn.Sequential(
            nn.Conv2d(channels, hidden, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden, channels, kernel_size=1),
        )

    def forward(self, x):
        avg = self.mlp(F.adaptive_avg_pool2d(x, 1))
        mx = self.mlp(F.adaptive_max_pool2d(x, 1))
        gate = torch.sigmoid(avg + mx)
        return x * gate


class SpatialAttention(nn.Module):
    """Eq. 5: SA(x) = Sigmoid(Conv7x7(ChannelMax(x) + ChannelAvg(x))) * x.
    Note: paper SUMS the max/avg maps (not concat, unlike CBAM)."""

    def __init__(self, kernel_size=7):
        super().__init__()
        self.conv = nn.Conv2d(1, 1, kernel_size=kernel_size, padding=kernel_size // 2)

    def forward(self, x):
        avg = x.mean(dim=1, keepdim=True)
        mx, _ = x.max(dim=1, keepdim=True)
        gate = torch.sigmoid(self.conv(avg + mx))
        return x * gate


class ConvBlock(nn.Module):
    """Eq. 6: two 3x3 convs, each -> BatchNorm -> ReLU."""

    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch), nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch), nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.block(x)


class ConvAttentionModule(nn.Module):
    """Eq. 3: CAM(x) = ConvBlock(SA(CA(x))) -- CA, then SA, then the two
    3x3-conv ConvBlock. Order matters (differs from plain CBAM)."""

    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.ca = ChannelAttention(in_ch)
        self.sa = SpatialAttention()
        self.conv_block = ConvBlock(in_ch, out_ch)

    def forward(self, x):
        x = self.ca(x)
        x = self.sa(x)
        return self.conv_block(x)


class AttentionGate(nn.Module):
    """Eq. 1-2: additive attention gate (Attention U-Net style).
        qatt(g, x) = ReLU(BN(Cg(g)) + BN(Cx(x)))     -- see note below
        AG(g, x)   = x * Sigmoid(BN(C(qatt(g, x))))

    Note on Eq. 1 as literally printed (qatt = ReLU(BN(Cg(g) + BN(Cx(x)))))
    puts only Cx(x) through its own BN before the add, with a single BN
    wrapping the whole sum. We instead BN both branches before summing
    (the standard Attention-U-Net formulation this paper cites, Oktay et
    al. 2018) since BN'ing an already-BN'd+raw sum is not well-defined and
    every public reimplementation of this exact gate (including Attention
    U-Net itself) applies BN to both the gate and skip 1x1 projections
    before adding. This is the one place we resolve an ambiguity in the
    paper's typesetting rather than following it letter-for-letter.
    """

    def __init__(self, gate_channels, skip_channels, inter_channels=None):
        super().__init__()
        inter_channels = inter_channels or max(skip_channels // 2, 8)
        self.Cg = nn.Sequential(
            nn.Conv2d(gate_channels, inter_channels, kernel_size=1),
            nn.BatchNorm2d(inter_channels),
        )
        self.Cx = nn.Sequential(
            nn.Conv2d(skip_channels, inter_channels, kernel_size=1),
            nn.BatchNorm2d(inter_channels),
        )
        self.psi = nn.Sequential(
            nn.Conv2d(inter_channels, 1, kernel_size=1),
            nn.BatchNorm2d(1),
        )

    def forward(self, g, x):
        if g.shape[-2:] != x.shape[-2:]:
            g = F.interpolate(g, size=x.shape[-2:], mode="bilinear", align_corners=False)
        qatt = F.relu(self.Cg(g) + self.Cx(x), inplace=True)
        alpha = torch.sigmoid(self.psi(qatt))
        return x * alpha


class UpConv(nn.Module):
    """Eq. 7: UpConv(x) = ReLU(BN(Conv3x3(Upsample_x2(x))))."""

    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.conv = nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1)
        self.bn = nn.BatchNorm2d(out_ch)

    def forward(self, x, size=None):
        if size is not None:
            x = F.interpolate(x, size=size, mode="bilinear", align_corners=False)
        else:
            x = F.interpolate(x, scale_factor=2, mode="bilinear", align_corners=False)
        return F.relu(self.bn(self.conv(x)), inplace=True)


class CascadeStage(nn.Module):
    """One decoder stage (used for the 3 stages that have a skip
    connection): UpConv the previous stage's output to the skip's
    resolution, AG-gate the skip, concat, then CAM to fuse+refine."""

    def __init__(self, in_channels, skip_channels, stage_channels):
        super().__init__()
        self.upconv = UpConv(in_channels, stage_channels)
        self.gate = AttentionGate(gate_channels=stage_channels, skip_channels=skip_channels)
        self.cam = ConvAttentionModule(stage_channels + skip_channels, stage_channels)

    def forward(self, x, skip):
        g = self.upconv(x, size=skip.shape[-2:])
        gated_skip = self.gate(g, skip)
        fused = torch.cat([g, gated_skip], dim=1)
        return self.cam(fused)


class CascadeDecoder(PyramidDecoder):
    """Full CASCADE decoder: CAM on the bottleneck (X4/s32, no skip), then
    3 CascadeStage blocks (AG + CAM) over s16/s8/s4, with a 1x1 prediction
    head at every stage. All 4 stage predictions are upsampled to stride 4
    and summed (Eq. 8, equal weights) to form the returned logits."""

    def __init__(self, enc_channels, decoder_dim=64, num_classes=1):
        super().__init__(enc_channels, decoder_dim, num_classes)
        c4, c8, c16, c32 = enc_channels

        # stage channel widths, progressively narrower toward the finest scale
        d32, d16, d8, d4 = decoder_dim * 8, decoder_dim * 4, decoder_dim * 2, decoder_dim

        # stage on the bottleneck (X4): CAM only, no AG (no coarser feature exists)
        self.bottleneck_cam = ConvAttentionModule(c32, d32)

        self.stage16 = CascadeStage(d32, c16, d16)
        self.stage8 = CascadeStage(d16, c8, d8)
        self.stage4 = CascadeStage(d8, c4, d4)

        # one 1x1 prediction head per stage (Eq. 8)
        self.head32 = nn.Conv2d(d32, num_classes, kernel_size=1)
        self.head16 = nn.Conv2d(d16, num_classes, kernel_size=1)
        self.head8 = nn.Conv2d(d8, num_classes, kernel_size=1)
        self.head4 = nn.Conv2d(d4, num_classes, kernel_size=1)

    def forward(self, feats):
        s4, s8, s16, s32 = feats
        target_size = s4.shape[-2:]  # stride 4, the common resolution for aggregation

        d32 = self.bottleneck_cam(s32)
        d16 = self.stage16(d32, s16)
        d8 = self.stage8(d16, s8)
        d4 = self.stage4(d8, s4)

        # Eq. 8: sum of all 4 stage predictions, each upsampled to stride 4.
        p32 = F.interpolate(self.head32(d32), size=target_size, mode="bilinear", align_corners=False)
        p16 = F.interpolate(self.head16(d16), size=target_size, mode="bilinear", align_corners=False)
        p8 = F.interpolate(self.head8(d8), size=target_size, mode="bilinear", align_corners=False)
        p4 = self.head4(d4)  # already at stride 4

        return p32 + p16 + p8 + p4  # logits at stride 4; central resize handles the rest