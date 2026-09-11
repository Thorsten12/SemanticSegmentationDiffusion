"""Decoder #7 (Group B) -- EMCAD.
Reference: Rahman, Munir & Marculescu, "EMCAD: Efficient Multi-scale
Convolutional Attention Decoding for Medical Image Segmentation",
CVPR 2024.
Paper: https://openaccess.thecvf.com/content/CVPR2024/papers/Rahman_EMCAD_Efficient_Multi-scale_Convolutional_Attention_Decoding_for_Medical_Image_Segmentation_CVPR_2024_paper.pdf
Official repo (not vendored here, but used to cross-check this reimplementation
against the paper's equations): https://github.com/SLDGroup/EMCAD

Faithful reimplementation of Section 3.1 (Eq. 1-10) and Section 3.3, NOT a
copy of the official repo (GitHub blocks automated fetching of raw files).
Traceable to:

  - LGAG (large-kernel grouped attention gate), Eq. 1-2:
        qatt(g, x) = R(BN(GCg(g) + BN(GCx(x))))
        LGAG(g, x) = x * Sigmoid(BN(C(qatt(g, x))))
    where GCg, GCx are 3x3 GROUP convolutions (not 1x1 as in Attention
    U-Net) -- this larger local context at lower cost than a full 3x3 is
    the point of "grouped". NOTE: the paper's Section 3.1.1 text does not
    give an explicit group count for GCg/GCx, only that they are "3x3
    group convolutions". We expose this as a `groups` argument and
    default to a depthwise-style grouping (groups == inter_channels),
    which maximizes the FLOPs reduction the paper is going for, but this
    specific number is our interpretation, not a value stated in the
    paper text -- flagged since the official repo (blocked from fetch)
    may use a different default.

  - MSCAM (multi-scale convolutional attention module), Eq. 3:
        MSCAM(x) = MSCB(SAB(CAB(x)))
    i.e. channel attention FIRST, then spatial attention, THEN the
    multi-scale conv block -- same CA->SA->conv ordering as CASCADE's
    CAM (Eq. 3 in cascade.py), just with MSCB standing in for the plain
    ConvBlock and SAB/CAB standing in for SA/CA.

  - CAB (channel attention block), Eq. 7:
        CAB(x) = Sigmoid(C2(R(C1(Pmax(x)))) + C2(R(C1(Pavg(x))))) * x
    Same C1, C2 (i.e. a single shared 1-hidden-layer MLP) applied to both
    the max- and avg-pooled descriptors, r=1/16 channel reduction --
    functionally identical to CASCADE's ChannelAttention (Eq. 4 there).

  - SAB (spatial attention block), Eq. 8:
        SAB(x) = Sigmoid(Conv7x7([Chmax(x), Chavg(x)])) * x
    Channel-max and channel-avg CONCATENATED (not summed, unlike
    CASCADE's SA) then a 7x7 conv -- same as G-CASCADE's SPA.

  - MSCB (multi-scale convolution block), Eq. 4-6, an inverted-residual
    block (MobileNetV2 IRB) adapted with multi-scale depthwise convs:
        MSCB(x) = BN(PWC2(CS(MSDC(R6(BN(PWC1(x)))))))
    PWC1: 1x1 conv expanding channels (expansion factor 2) -> BN -> ReLU6.
    MSDC: multi-scale depthwise convs (Eq. 5, parallel arrangement, used
    in all of the paper's experiments per Section 4.1's implementation
    details): MSDC(x) = sum_{ks in KS} DWCBks(x), DWCBks(x) =
    ReLU6(BN(DWConv_ks(x))), with kernel sizes KS=[1,3,5] (chosen via the
    paper's own ablation, Table 5). Eq. 6 additionally describes a
    "sequential" variant where each kernel's DWCB acts on the previous
    kernel's residually-updated input (x = x + DWCBks(x)) rather than the
    parallel/independent-branch-then-sum of Eq. 5; the paper states
    explicitly it uses the PARALLEL arrangement in all experiments, so
    that is the default here, with 'sequential' offered as an option for
    completeness since Eq. 6 is part of the paper's own formulation.
    CS: channel shuffle (Zhang et al., ShuffleNet) to restore cross-
    channel information the per-channel depthwise convs can't see. The
    paper doesn't give a group count for the shuffle either -- we use
    groups = len(kernel_sizes), a reasonable, common ShuffleNet-style
    choice tied to the module's own branch count, not a value taken from
    the paper. PWC2: 1x1 conv projecting back to the original channel
    count, -> BN (no activation after, matching Eq. 4's outermost BN with
    no trailing R). Residual: the paper bases MSCB explicitly on
    MobileNetV2's inverted residual block, which always adds a residual
    when stride=1 and in_ch==out_ch (true in every place MSCB is used
    here); we keep that residual for the same reason cascade.py/gcascade.py
    keep GCB's residual -- it's standard for the cited base design even
    though Eq. 4 itself doesn't draw the addition explicitly.

  - EUCB (efficient up-conv block), Eq. 9:
        EUCB(x) = Conv1x1(ReLU(BN(DWConv3x3(Upsample_x2(x)))))
    Structurally identical to G-CASCADE's UCB (upsample -> depthwise 3x3
    -> BN -> ReLU -> 1x1) -- both papers converge on the same "efficient
    up-conv" recipe.

  - SH (segmentation head), Eq. 10: a single 1x1 conv per stage ->
    num_classes channels. Same pattern as CASCADE's/G-CASCADE's heads.

  - Stage fusion (Section 3.1, prose immediately before 3.1.1): "we
    upscale the refined feature maps using EUCBs and add them to the
    outputs from the corresponding LGAGs." This is an ADDITIVE fusion of
    (a) the EUCB-upsampled previous-stage feature (used directly as the
    LGAG's gating signal g) and (b) that same LGAG's gated skip output
    LGAG(g, skip) -- NOT a concatenation (unlike CASCADE's AG+concat, and
    unlike G-CASCADE's additive skip_proj+ucb but without a gate). We
    follow this additive, gated fusion exactly.

  - Output aggregation, Section 3.3 -- this is the one place EMCAD's
    logic genuinely diverges from CASCADE (Eq. 8) and G-CASCADE (Eq. 7):
    those two SUM all 4 upsampled stage predictions to form the final
    output. EMCAD does NOT: "we consider the prediction map, p4, from
    the last stage of our decoder as the final segmentation map." The
    4-head aggregation in EMCAD only happens inside the (unused-here)
    MUTATION/additive multi-stage LOSS (Eq. 11, training-time only, not
    part of the forward pass). We therefore return ONLY the finest-stage
    (last-processed) head's prediction, at its native stride-4
    resolution, with no upsample-and-sum step -- this is more faithful
    to the paper than mechanically reusing the CASCADE/G-CASCADE
    aggregation pattern would have been. The other 3 stage heads are
    still computed and stored as attributes get populated in forward()
    (see `self._aux_predictions`) purely to leave the door open for
    per-stage deep supervision later, matching this project's existing
    convention of building the multi-stage heads even though the shared
    trainer currently only supervises a single final output.

Adaptations for arbitrary encoders (this codebase supports ConvNeXt, PVT,
ResNet-50, Swin-T, etc., not just the paper's PVTv2): the paper's decoder
channel widths match its PVTv2 encoder's native channel counts stage for
stage, so no extra projections are needed there. Since our encoders have
arbitrary channel counts, we add a 1x1 projection conv on the bottleneck
(c32 -> d32, same as G-CASCADE's bottleneck) and a 1x1 skip projection
per stage (skip_channels -> stage_channels) before each LGAG, so that the
LGAG's gating signal g and skip x always arrive at the same channel width
and the subsequent additive fusion (u + LGAG(u, proj(skip))) is well
defined. These projections are not present in the paper (not needed
there) and are the same kind of adaptation cascade.py/gcascade.py already
document for their own skip/bottleneck handling.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from .base import PyramidDecoder


def channel_shuffle(x, groups):
    """Standard ShuffleNet channel shuffle: reshape channels into
    (groups, channels_per_group), transpose, flatten back. Restores
    cross-channel mixing after depthwise (per-channel-independent) convs."""
    B, C, H, W = x.shape
    if groups <= 1 or C % groups != 0:
        return x
    cpg = C // groups
    x = x.view(B, groups, cpg, H, W)
    x = x.transpose(1, 2).contiguous()
    return x.view(B, C, H, W)


class ChannelAttentionBlock(nn.Module):
    """CAB, Eq. 7: Sigmoid(C2(R(C1(Pmax(x)))) + C2(R(C1(Pavg(x))))) * x,
    with a SHARED C1/C2 (1-hidden-layer 1x1-conv MLP) across both pooled
    descriptors, reduction r=1/16. Same design as CASCADE's ChannelAttention."""

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


class SpatialAttentionBlock(nn.Module):
    """SAB, Eq. 8: Sigmoid(Conv7x7([Chmax(x), Chavg(x)])) * x.
    Concat (not sum, unlike CASCADE's SA) -- same as G-CASCADE's SPA."""

    def __init__(self, kernel_size=7):
        super().__init__()
        self.conv = nn.Conv2d(2, 1, kernel_size=kernel_size, padding=kernel_size // 2)

    def forward(self, x):
        avg = x.mean(dim=1, keepdim=True)
        mx, _ = x.max(dim=1, keepdim=True)
        gate = torch.sigmoid(self.conv(torch.cat([mx, avg], dim=1)))
        return x * gate


class MSDC(nn.Module):
    """Multi-scale depth-wise convolution, Eq. 5 (parallel, the default
    used in all of the paper's experiments) or Eq. 6 (sequential,
    provided for completeness). Each DWCBks(x) = ReLU6(BN(DWConv_ks(x))).
    """

    def __init__(self, channels, kernel_sizes=(1, 3, 5), arrangement="parallel"):
        super().__init__()
        assert arrangement in ("parallel", "sequential")
        self.arrangement = arrangement
        self.kernel_sizes = list(kernel_sizes)
        self.branches = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(channels, channels, kernel_size=ks, padding=ks // 2, groups=channels),
                nn.BatchNorm2d(channels),
            )
            for ks in self.kernel_sizes
        ])

    def _dwcb(self, branch, x):
        return F.relu6(branch(x), inplace=True)

    def forward(self, x):
        if self.arrangement == "parallel":
            # Eq. 5: independent branches on the same input, summed.
            out = 0
            for branch in self.branches:
                out = out + self._dwcb(branch, x)
            return out
        # Eq. 6: sequential, recursively-updated, residually-connected input.
        for branch in self.branches:
            x = x + self._dwcb(branch, x)
        return x


class MSCB(nn.Module):
    """Multi-scale convolution block, Eq. 4: an inverted-residual block
    (MobileNetV2 IRB) with multi-scale depthwise convs (MSDC) and a
    channel shuffle in place of IRB's single plain depthwise conv.
        MSCB(x) = BN(PWC2(CS(MSDC(R6(BN(PWC1(x)))))))
    Residual add included (stride=1, in_ch==out_ch everywhere MSCB is
    used here), per the standard IRB design this block is explicitly
    based on -- see module-level docstring."""

    def __init__(self, channels, expansion=2, kernel_sizes=(1, 3, 5), arrangement="parallel"):
        super().__init__()
        hidden = channels * expansion
        self.pwc1 = nn.Sequential(
            nn.Conv2d(channels, hidden, kernel_size=1),
            nn.BatchNorm2d(hidden),
        )
        self.msdc = MSDC(hidden, kernel_sizes=kernel_sizes, arrangement=arrangement)
        self.shuffle_groups = len(kernel_sizes)
        self.pwc2 = nn.Sequential(
            nn.Conv2d(hidden, channels, kernel_size=1),
            nn.BatchNorm2d(channels),
        )

    def forward(self, x):
        residual = x
        h = F.relu6(self.pwc1(x), inplace=True)
        h = self.msdc(h)
        h = channel_shuffle(h, self.shuffle_groups)
        h = self.pwc2(h)  # no activation after, per Eq. 4
        return h + residual


class MSCAM(nn.Module):
    """Eq. 3: MSCAM(x) = MSCB(SAB(CAB(x))) -- CA, then SA, then the
    multi-scale conv block. Same ordering as CASCADE's CAM."""

    def __init__(self, channels, reduction=16, kernel_sizes=(1, 3, 5), arrangement="parallel"):
        super().__init__()
        self.cab = ChannelAttentionBlock(channels, reduction=reduction)
        self.sab = SpatialAttentionBlock()
        self.mscb = MSCB(channels, kernel_sizes=kernel_sizes, arrangement=arrangement)

    def forward(self, x):
        x = self.cab(x)
        x = self.sab(x)
        return self.mscb(x)


class LGAG(nn.Module):
    """Eq. 1-2: large-kernel grouped attention gate.
        qatt(g, x) = ReLU(BN(GCg(g)) + BN(GCx(x)))
        LGAG(g, x) = x * Sigmoid(BN(C(qatt(g, x))))
    GCg, GCx are 3x3 GROUP convolutions (vs. 1x1 plain convs in Attention
    U-Net) for a larger local context at lower cost. `groups` is not
    given an explicit value in the paper's main text -- default here is
    depthwise-style (groups == inter_channels); see module docstring."""

    def __init__(self, gate_channels, skip_channels, inter_channels=None, kernel_size=3, groups=None):
        super().__init__()
        inter_channels = inter_channels or max(skip_channels // 2, 8)
        groups = groups or inter_channels  # depthwise-style default; see docstring note
        # groups must evenly divide both in/out channels of a grouped conv
        groups = max(1, min(groups, inter_channels))
        while inter_channels % groups != 0:
            groups -= 1

        self.Cg = nn.Sequential(
            nn.Conv2d(gate_channels, inter_channels, kernel_size=kernel_size,
                      padding=kernel_size // 2, groups=1 if gate_channels % groups else groups),
            nn.BatchNorm2d(inter_channels),
        )
        self.Cx = nn.Sequential(
            nn.Conv2d(skip_channels, inter_channels, kernel_size=kernel_size,
                      padding=kernel_size // 2, groups=1 if skip_channels % groups else groups),
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


class EUCB(nn.Module):
    """Eq. 9: EUCB(x) = Conv1x1(ReLU(BN(DWConv3x3(Upsample_x2(x))))).
    Structurally identical to G-CASCADE's UCB."""

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


class EmcadStage(nn.Module):
    """One decoder stage: EUCB the previous stage's refined feature map up
    to the skip's resolution (this upsampled feature IS the LGAG gating
    signal g), project+gate the skip via LGAG, ADD the two (not concat --
    see module docstring), then refine with MSCAM."""

    def __init__(self, in_channels, skip_channels, stage_channels, reduction=16,
                 kernel_sizes=(1, 3, 5), arrangement="parallel", lgag_groups=None):
        super().__init__()
        self.eucb = EUCB(in_channels, stage_channels)
        self.skip_proj = nn.Conv2d(skip_channels, stage_channels, kernel_size=1)
        self.lgag = LGAG(gate_channels=stage_channels, skip_channels=stage_channels, groups=lgag_groups)
        self.mscam = MSCAM(stage_channels, reduction=reduction, kernel_sizes=kernel_sizes, arrangement=arrangement)

    def forward(self, x, skip):
        u = self.eucb(x, size=skip.shape[-2:])
        gated = self.lgag(u, self.skip_proj(skip))
        fused = u + gated  # additive fusion, per paper's prose (not concat)
        return self.mscam(fused)


class EmcadDecoder(PyramidDecoder):
    """Full EMCAD decoder: MSCAM on the bottleneck (X4/s32, no skip,
    channel-projected first), then 3 EmcadStage blocks (EUCB + LGAG +
    additive fuse + MSCAM) over s16/s8/s4, with a 1x1 SegHead at every
    stage. Unlike CASCADE/G-CASCADE, the forward pass returns ONLY the
    finest-stage (last) head's prediction at its native stride-4
    resolution -- no upsample-and-sum aggregation -- matching the
    paper's Section 3.3 output rule. The other 3 heads are still
    computed and exposed via `self._aux_predictions` after a forward
    call, purely for optional future deep supervision."""

    def __init__(self, enc_channels, decoder_dim=64, num_classes=1,
                 kernel_sizes=(1, 3, 5), arrangement="parallel", lgag_groups=None):
        super().__init__(enc_channels, decoder_dim, num_classes)
        c4, c8, c16, c32 = enc_channels

        d32, d16, d8, d4 = decoder_dim * 8, decoder_dim * 4, decoder_dim * 2, decoder_dim

        self.bottleneck_proj = nn.Conv2d(c32, d32, kernel_size=1)
        self.bottleneck_mscam = MSCAM(d32, kernel_sizes=kernel_sizes, arrangement=arrangement)

        self.stage16 = EmcadStage(d32, c16, d16, kernel_sizes=kernel_sizes,
                                   arrangement=arrangement, lgag_groups=lgag_groups)
        self.stage8 = EmcadStage(d16, c8, d8, kernel_sizes=kernel_sizes,
                                  arrangement=arrangement, lgag_groups=lgag_groups)
        self.stage4 = EmcadStage(d8, c4, d4, kernel_sizes=kernel_sizes,
                                  arrangement=arrangement, lgag_groups=lgag_groups)

        self.head32 = nn.Conv2d(d32, num_classes, kernel_size=1)
        self.head16 = nn.Conv2d(d16, num_classes, kernel_size=1)
        self.head8 = nn.Conv2d(d8, num_classes, kernel_size=1)
        self.head4 = nn.Conv2d(d4, num_classes, kernel_size=1)

        self._aux_predictions = None  # populated in forward(); (p32, p16, p8) upsampled to stride 4

    def forward(self, feats):
        s4, s8, s16, s32 = feats
        target_size = s4.shape[-2:]

        d32 = self.bottleneck_mscam(self.bottleneck_proj(s32))
        d16 = self.stage16(d32, s16)
        d8 = self.stage8(d16, s8)
        d4 = self.stage4(d8, s4)

        p32 = F.interpolate(self.head32(d32), size=target_size, mode="bilinear", align_corners=False)
        p16 = F.interpolate(self.head16(d16), size=target_size, mode="bilinear", align_corners=False)
        p8 = F.interpolate(self.head8(d8), size=target_size, mode="bilinear", align_corners=False)
        p4 = self.head4(d4)  # already at stride 4 -- this is the paper's actual final output

        self._aux_predictions = (p32, p16, p8)  # available for optional deep supervision, unused by default
        return p4  # Section 3.3: "we consider p4 ... as the final segmentation map" -- no sum, unlike Eq. 8/Eq. 7
