"""Decoder #1 (Group A) -- SegFormer-style Linear / all-MLP head.
Reference: Xie et al., "SegFormer: Simple and Efficient Design for
Semantic Segmentation with Transformers", NeurIPS 2021.

This is the TRUE trivial floor: no 3x3 convs anywhere, no progressive
upsampling structure. Per scale: a single 1x1 conv projects to decoder_dim
channels, then each scale is upsampled to stride 4 and concatenated, then
one more 1x1 conv fuses the concatenation into logits. That's it.

Contrast with shallow_conv_head.py, which has two 3x3 convs after the
concat (real, if shallow, spatial mixing) -- this file is the head that
actually earns the name "linear/all-MLP" and answers the reviewer's
question "why is a simple head not enough?".
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from .base import PyramidDecoder


class LinearMLPDecoder(PyramidDecoder):
    """SegFormer-style floor: per-scale 1x1 projection -> upsample to
    stride 4 -> concat -> 1x1 fusion to logits. No 3x3 convs, no
    progressive/skip-connection decoding."""

    def __init__(self, enc_channels, decoder_dim=64, num_classes=1):
        super().__init__(enc_channels, decoder_dim, num_classes)

        # One 1x1 "MLP" projection per scale (Linear applied per-pixel == 1x1 conv).
        self.proj = nn.ModuleList([
            nn.Conv2d(c, decoder_dim, kernel_size=1) for c in enc_channels
        ])
        # Fuse the 4 concatenated projections (4 * decoder_dim channels) -> logits.
        self.fuse = nn.Conv2d(decoder_dim * len(enc_channels), num_classes, kernel_size=1)

    def forward(self, feats):
        target_size = feats[0].shape[-2:]  # stride 4, the finest scale
        projected = []
        for f, p in zip(feats, self.proj):
            x = p(f)  # 1x1 projection, no spatial mixing
            if x.shape[-2:] != target_size:
                x = F.interpolate(x, size=target_size, mode="bilinear", align_corners=False)
            projected.append(x)
        fused = torch.cat(projected, dim=1)
        return self.fuse(fused)  # logits at stride 4; central resize handles the rest