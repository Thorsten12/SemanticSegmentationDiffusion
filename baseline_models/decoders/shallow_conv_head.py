"""Shallow conv head -- this is your ORIGINAL 'simple_head' (SimpleHeadSegmenter),
renamed to shallow_conv_head to avoid confusion with the true SegFormer-style
linear/all-MLP floor (see linear_mlp.py).

Why the rename: this head does concat -> 3x3 conv -> 3x3 conv -> 1x1 conv,
i.e. it has two spatial 3x3 convs that mix neighbouring pixels. That's real
(if shallow) decoder capacity -- it is NOT the "linear" floor a reviewer
means by "why is a simple head not enough". It stays in the comparison as
its own row (a shallow-but-nonlinear dense head), just correctly labeled.

Moved here UNCHANGED so already-trained checkpoints
(table/<encoder>/baseline_simple_head/...) stay valid.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from .base import PyramidDecoder


class ShallowConvHeadDecoder(PyramidDecoder):
    """Backbone pyramid -> upsample all scales to stride-4 -> concat ->
    small conv head (two 3x3 convs) -> 1x1 to logits. No progressive
    decoder, no skip-connection refinement beyond the single concat."""

    def __init__(self, enc_channels, decoder_dim=64, num_classes=1):
        super().__init__(enc_channels, decoder_dim, num_classes)
        total_ch = sum(enc_channels)
        g = min(8, decoder_dim)
        self.head = nn.Sequential(
            nn.Conv2d(total_ch, decoder_dim, kernel_size=3, padding=1),
            nn.GroupNorm(g, decoder_dim), nn.SiLU(),
            nn.Conv2d(decoder_dim, decoder_dim, kernel_size=3, padding=1),
            nn.GroupNorm(g, decoder_dim), nn.SiLU(),
            nn.Conv2d(decoder_dim, num_classes, kernel_size=1),
        )

    def forward(self, feats):
        target_size = feats[0].shape[-2:]  # resize everything to stride 4 (finest)
        resized = [feats[0]] + [
            F.interpolate(f, size=target_size, mode="bilinear", align_corners=False)
            for f in feats[1:]
        ]
        fused = torch.cat(resized, dim=1)
        return self.head(fused)  # logits at stride 4; central resize handles the rest