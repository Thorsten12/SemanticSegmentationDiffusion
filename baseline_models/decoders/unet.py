"""UNet decoder -- Decoder #3 in the group-A comparison (U-Net, MICCAI 2015).

This is your ORIGINAL BaselineUNetSegmenter decoder, moved here UNCHANGED so
that already-trained checkpoints (table/<encoder>/baseline_unet/...) stay
valid and comparable. Only the packaging changed: it now exposes the shared
PyramidDecoder.forward(feats) -> logits contract instead of owning its own
build_encoder() call.

Do NOT modify UNetDecoderBlock's topology (upsample -> concat skip -> two
3x3 convs, GroupNorm+SiLU) -- any change here invalidates the existing
sweep results.
"""

import torch
import torch.nn as nn

from .base import PyramidDecoder


class UNetDecoderBlock(nn.Module):
    """Upsample x2 -> concat skip -> two 3x3 convs (GroupNorm + SiLU).
    Unchanged from the original baseline_seg.py."""

    def __init__(self, in_channels, skip_channels, out_channels, groups=8):
        super().__init__()
        self.upsample = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True)
        combined = in_channels + skip_channels if skip_channels > 0 else in_channels
        g = min(groups, out_channels)
        self.conv = nn.Sequential(
            nn.Conv2d(combined, out_channels, 3, padding=1),
            nn.GroupNorm(g, out_channels), nn.SiLU(),
            nn.Conv2d(out_channels, out_channels, 3, padding=1),
            nn.GroupNorm(g, out_channels), nn.SiLU(),
        )

    def forward(self, x, skip=None):
        x = self.upsample(x)
        if skip is not None:
            x = torch.cat([x, skip], dim=1)
        return self.conv(x)


class UNetDecoder(PyramidDecoder):
    """Same topology as the original BaselineUNetSegmenter:
    s32 -> s16 -> s8 -> s4 -> s2 -> s1 (full image resolution), with skip
    connections from s16/s8/s4 and two extra skip-less upsample blocks to
    reach full resolution directly (no final F.interpolate needed, though
    BaselineSegmenter's central resize is a harmless no-op here)."""

    def __init__(self, enc_channels, decoder_dim=64, num_classes=1):
        super().__init__(enc_channels, decoder_dim, num_classes)
        c4, c8, c16, c32 = enc_channels

        self.dec1 = UNetDecoderBlock(c32, c16, decoder_dim * 4)   # s32 -> s16
        self.dec2 = UNetDecoderBlock(decoder_dim * 4, c8, decoder_dim * 2)  # s16 -> s8
        self.dec3 = UNetDecoderBlock(decoder_dim * 2, c4, decoder_dim)     # s8  -> s4
        self.dec4 = UNetDecoderBlock(decoder_dim, 0, decoder_dim)  # s4 -> s2
        self.dec5 = UNetDecoderBlock(decoder_dim, 0, decoder_dim)  # s2 -> s1 (full res)

        self.head = nn.Conv2d(decoder_dim, num_classes, kernel_size=1)

    def forward(self, feats):
        s4, s8, s16, s32 = feats
        x = self.dec1(s32, s16)
        x = self.dec2(x, s8)
        x = self.dec3(x, s4)
        x = self.dec4(x, skip=None)
        x = self.dec5(x, skip=None)
        return self.head(x)