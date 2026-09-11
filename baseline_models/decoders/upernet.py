"""Decoder #4 (Group A) -- UPerNet.
Reference: Xiao et al., "Unified Perceptual Parsing for Scene Understanding",
ECCV 2018.

Strong conventional context-aggregating decoder: a Pyramid Pooling Module
(PPM, from PSPNet) is applied to the coarsest scale (s32) to inject
multi-receptive-field global context, then the PPM output is fed into a
standard top-down FPN (lateral 1x1 + upsample-add + 3x3 smooth) over the
remaining scales, and finally all FPN levels are upsampled to stride 4,
concatenated, and fused by a 3x3 conv + 1x1 classifier -- exactly the
"fusion" step from the original UPerNet head (their eq. right before the
final classifier), except we concat+fuse instead of sum (also a common
variant, e.g. used in mmsegmentation's UPerHead) since it gives the
classifier explicit per-level information instead of an implicit sum.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from .base import PyramidDecoder


class _ConvGNReLU(nn.Module):
    def __init__(self, in_ch, out_ch, kernel_size=3, groups=8):
        super().__init__()
        g = min(groups, out_ch)
        padding = kernel_size // 2
        self.block = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size, padding=padding, bias=False),
            nn.GroupNorm(g, out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.block(x)


class PyramidPoolingModule(nn.Module):
    """PSPNet-style PPM: adaptive-avg-pool the input to several fixed grid
    sizes, 1x1-project each to a small channel count, upsample back to the
    input resolution, and concat with the input itself. Injects global /
    multi-scale context into the coarsest feature map."""

    def __init__(self, in_ch, out_ch, pool_sizes=(1, 2, 3, 6)):
        super().__init__()
        branch_ch = out_ch // len(pool_sizes)
        self.branches = nn.ModuleList([
            nn.Sequential(
                nn.AdaptiveAvgPool2d(pool_size),
                nn.Conv2d(in_ch, branch_ch, kernel_size=1, bias=False),
                nn.GroupNorm(min(8, branch_ch), branch_ch),
                nn.ReLU(inplace=True),
            )
            for pool_size in pool_sizes
        ])
        self.fuse = _ConvGNReLU(in_ch + branch_ch * len(pool_sizes), out_ch, kernel_size=3)

    def forward(self, x):
        size = x.shape[-2:]
        pooled = [x] + [
            F.interpolate(branch(x), size=size, mode="bilinear", align_corners=False)
            for branch in self.branches
        ]
        return self.fuse(torch.cat(pooled, dim=1))


class UPerNetDecoder(PyramidDecoder):
    """PPM on the coarsest scale (global context) -> top-down FPN over all
    4 scales -> upsample every level to stride 4 -> concat -> 3x3 fuse ->
    1x1 classifier."""

    def __init__(self, enc_channels, decoder_dim=64, num_classes=1):
        super().__init__(enc_channels, decoder_dim, num_classes)
        c4, c8, c16, c32 = enc_channels

        # ----- PPM on the coarsest scale -> decoder_dim channels -----
        self.ppm = PyramidPoolingModule(c32, decoder_dim)

        # ----- lateral 1x1 projections for the finer scales -----
        self.lateral4 = nn.Conv2d(c4, decoder_dim, kernel_size=1)
        self.lateral8 = nn.Conv2d(c8, decoder_dim, kernel_size=1)
        self.lateral16 = nn.Conv2d(c16, decoder_dim, kernel_size=1)

        # ----- top-down smoothing convs (after upsample-add) -----
        self.smooth4 = _ConvGNReLU(decoder_dim, decoder_dim)
        self.smooth8 = _ConvGNReLU(decoder_dim, decoder_dim)
        self.smooth16 = _ConvGNReLU(decoder_dim, decoder_dim)

        # ----- final fusion: concat all 4 levels (upsampled to stride 4) -----
        self.fuse = _ConvGNReLU(decoder_dim * 4, decoder_dim, kernel_size=3)
        self.classifier = nn.Conv2d(decoder_dim, num_classes, kernel_size=1)

    @staticmethod
    def _upsample_to(x, size):
        if x.shape[-2:] == tuple(size):
            return x
        return F.interpolate(x, size=size, mode="bilinear", align_corners=False)

    def forward(self, feats):
        s4, s8, s16, s32 = feats
        target_size = s4.shape[-2:]  # stride 4

        # ----- PPM injects global context into the coarsest scale -----
        p32 = self.ppm(s32)

        # ----- top-down pathway: lateral + upsample-add + smooth -----
        # Explicit target size (not scale_factor=2): encoder feature maps
        # aren't guaranteed to be exact powers of 2 apart, so resize to the
        # exact size of the tensor being added into, not a blind x2.
        p16 = self.smooth16(self.lateral16(s16) + self._upsample_to(p32, s16.shape[-2:]))
        p8 = self.smooth8(self.lateral8(s8) + self._upsample_to(p16, s8.shape[-2:]))
        p4 = self.smooth4(self.lateral4(s4) + self._upsample_to(p8, s4.shape[-2:]))

        # ----- bring every level to stride 4, concat, fuse -----
        levels = [
            p4,
            self._upsample_to(p8, target_size),
            self._upsample_to(p16, target_size),
            self._upsample_to(p32, target_size),
        ]
        fused = self.fuse(torch.cat(levels, dim=1))
        return self.classifier(fused)  # logits at stride 4; central resize handles the rest