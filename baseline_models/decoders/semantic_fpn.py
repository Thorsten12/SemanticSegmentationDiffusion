"""Decoder #2 (Group A) -- Semantic FPN.
Reference: Kirillov et al., "Panoptic Feature Pyramid Networks", CVPR 2019.
Paper (Sec. "Semantic segmentation branch", Fig. 3):
https://openaccess.thecvf.com/content_CVPR_2019/papers/Kirillov_Panoptic_Feature_Pyramid_Networks_CVPR_2019_paper.pdf

Standard lightweight pyramid decoder, following Fig. 3 / the "Semantic
segmentation branch" section exactly: build a top-down FPN (lateral 1x1 +
top-down upsample-add, then a 3x3 smoothing conv per level -- the classic
FPN recipe from Lin et al. 2017), then for each pyramid level run a stack
of upsampling stages until it reaches stride 4, where EVERY upsampling
stage is "3x3 convolution, group norm, ReLU, and 2x bilinear upsampling"
-- conv/norm/act happen at the CURRENT (coarser) resolution, THEN the 2x
upsample, not the other way around. P5 (stride 32) needs 3 such stages,
P4 (stride 16) needs 2, P3 (stride 8) needs 1, P2 (stride 4) needs 0. All
four stride-4 outputs are then summed elementwise (not concatenated) and
passed through a final 1x1 classifier + 4x upsample. Applied here to a
4-scale (stride 4/8/16/32) pyramid instead of the original 5-scale
(stride 4/8/16/32/64) FPN, since our encoders only go down to stride 32.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from .base import PyramidDecoder


class _ConvGNReLU(nn.Module):
    """3x3 conv + GroupNorm + ReLU, the basic FPN building block."""

    def __init__(self, in_ch, out_ch, groups=8):
        super().__init__()
        g = min(groups, out_ch)
        self.block = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1, bias=False),
            nn.GroupNorm(g, out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.block(x)


class SemanticFPNDecoder(PyramidDecoder):
    """Top-down FPN fusion (lateral 1x1 + upsample-add + 3x3 smooth), then
    a per-level upsample stack to stride 4, then elementwise sum, then a
    1x1 classifier. Standard Panoptic-FPN semantic head."""

    def __init__(self, enc_channels, decoder_dim=64, num_classes=1):
        super().__init__(enc_channels, decoder_dim, num_classes)
        c4, c8, c16, c32 = enc_channels

        # ----- lateral 1x1 projections: every scale -> decoder_dim channels -----
        self.lateral4 = nn.Conv2d(c4, decoder_dim, kernel_size=1)
        self.lateral8 = nn.Conv2d(c8, decoder_dim, kernel_size=1)
        self.lateral16 = nn.Conv2d(c16, decoder_dim, kernel_size=1)
        self.lateral32 = nn.Conv2d(c32, decoder_dim, kernel_size=1)

        # ----- top-down smoothing convs (after upsample-add) -----
        self.smooth4 = _ConvGNReLU(decoder_dim, decoder_dim)
        self.smooth8 = _ConvGNReLU(decoder_dim, decoder_dim)
        self.smooth16 = _ConvGNReLU(decoder_dim, decoder_dim)
        # p32 needs no smoothing conv -- it's the coarsest level, nothing to add into it.

        # ----- per-level upsample stacks to bring every FPN level to stride 4 -----
        # p4 is already at stride 4: identity (no conv needed).
        # p8 -> stride 4: one upsample-conv step (x2).
        self.up8 = _ConvGNReLU(decoder_dim, decoder_dim)
        # p16 -> stride 4: two upsample-conv steps (x2, x2).
        self.up16_a = _ConvGNReLU(decoder_dim, decoder_dim)
        self.up16_b = _ConvGNReLU(decoder_dim, decoder_dim)
        # p32 -> stride 4: three upsample-conv steps (x2, x2, x2).
        self.up32_a = _ConvGNReLU(decoder_dim, decoder_dim)
        self.up32_b = _ConvGNReLU(decoder_dim, decoder_dim)
        self.up32_c = _ConvGNReLU(decoder_dim, decoder_dim)

        self.classifier = nn.Conv2d(decoder_dim, num_classes, kernel_size=1)

    @staticmethod
    def _upsample_to(x, size):
        # Explicit target size instead of scale_factor=2: encoder feature
        # maps aren't guaranteed to be exact powers of 2 apart (e.g. a
        # 63x63 stride-4 map from a 253px input paired with a 32x32 stride-8
        # map isn't a clean x2 relationship), so blind scale_factor=2 upsampling
        # can produce a shape 1px off from the tensor it needs to be added to.
        # Always resize to the exact size of the target level instead.
        if x.shape[-2:] == tuple(size):
            return x
        return F.interpolate(x, size=size, mode="bilinear", align_corners=False)

    def forward(self, feats):
        s4, s8, s16, s32 = feats
        target_size = s4.shape[-2:]

        # ----- top-down pathway: lateral + upsample-add + smooth -----
        p32 = self.lateral32(s32)
        p16 = self.smooth16(self.lateral16(s16) + self._upsample_to(p32, s16.shape[-2:]))
        p8 = self.smooth8(self.lateral8(s8) + self._upsample_to(p16, s8.shape[-2:]))
        p4 = self.smooth4(self.lateral4(s4) + self._upsample_to(p8, s4.shape[-2:]))

        # ----- bring every level to stride 4: conv+GN+ReLU at current res,
        # THEN upsample (Fig. 3's per-stage order), repeated until stride 4 -----
        out4 = p4  # already stride 4, no upsampling stages needed

        out8 = self.up8(p8)  # conv at stride 8...
        out8 = self._upsample_to(out8, target_size)  # ...then upsample to stride 4

        out16 = self.up16_a(p16)  # conv at stride 16...
        out16 = self._upsample_to(out16, s8.shape[-2:])  # ...then upsample to stride 8
        out16 = self.up16_b(out16)  # conv at stride 8...
        out16 = self._upsample_to(out16, target_size)  # ...then upsample to stride 4

        out32 = self.up32_a(p32)  # conv at stride 32...
        out32 = self._upsample_to(out32, s16.shape[-2:])  # ...then upsample to stride 16
        out32 = self.up32_b(out32)  # conv at stride 16...
        out32 = self._upsample_to(out32, s8.shape[-2:])  # ...then upsample to stride 8
        out32 = self.up32_c(out32)  # conv at stride 8...
        out32 = self._upsample_to(out32, target_size)  # ...then upsample to stride 4

        fused = out4 + out8 + out16 + out32  # elementwise SUM, not concat (per Fig. 3)
        return self.classifier(fused)  # logits at stride 4; central resize handles the rest