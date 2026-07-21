"""Conditioning U-Net: RGB image -> dense feature map ("condition map").

The boundary denoiser samples this feature map at the (noisy) point locations,
so the U-Net's only job is to turn the image into spatially-aligned features that
encode "where the lesion boundary is". It is *not* time-conditioned: the same
condition map is reused across all diffusion steps for a given image.

`forward` returns a *list* of feature maps so the denoiser can be extended to
multi-scale conditioning later; the baseline returns a single full-resolution map.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from .attention import ImageTransformerBlock


class ResidualBlock(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, groups: int = 16):
        super().__init__()
        g_in = min(groups, in_ch)
        g_out = min(groups, out_ch)
        self.norm1 = nn.GroupNorm(g_in, in_ch)
        self.conv1 = nn.Conv2d(in_ch, out_ch, 3, padding=1)
        self.norm2 = nn.GroupNorm(g_out, out_ch)
        self.conv2 = nn.Conv2d(out_ch, out_ch, 3, padding=1)
        self.skip = nn.Conv2d(in_ch, out_ch, 1) if in_ch != out_ch else nn.Identity()

    def forward(self, x):
        h = self.conv1(F.silu(self.norm1(x)))
        h = self.conv2(F.silu(self.norm2(h)))
        return h + self.skip(x)


class FeatureUNet(nn.Module):
    def __init__(
        self,
        in_channels: int = 3,
        start_dim: int = 64,
        dim_mults=(1, 2, 4),
        out_features: int = 64,
        groups: int = 16,
    ):
        super().__init__()
        dims = [start_dim * m for m in dim_mults]            # e.g. [64, 128, 256]
        self.in_proj = nn.Conv2d(in_channels, dims[0], 3, padding=1)

        # ----- encoder -----
        self.down_blocks = nn.ModuleList()
        self.downsamples = nn.ModuleList()
        for i in range(len(dims)):
            in_ch = dims[i - 1] if i > 0 else dims[0]
            self.down_blocks.append(ResidualBlock(in_ch, dims[i], groups))
            # Downsample after every stage except the last.
            if i < len(dims) - 1:
                self.downsamples.append(nn.Conv2d(dims[i], dims[i], 3, stride=2, padding=1))
            else:
                self.downsamples.append(nn.Identity())

        # ----- bottleneck (attention only at the coarsest, cheapest resolution) -----
        self.mid_block1 = ResidualBlock(dims[-1], dims[-1], groups)
        self.mid_attn = ImageTransformerBlock(dims[-1])
        self.mid_block2 = ResidualBlock(dims[-1], dims[-1], groups)

        # ----- decoder (mirrors encoder, with skip connections) -----
        self.up_blocks = nn.ModuleList()
        self.upsamples = nn.ModuleList()
        for i in reversed(range(len(dims))):
            # Skip channels come from the matching encoder stage (dims[i]).
            self.up_blocks.append(ResidualBlock(dims[i] * 2, dims[i], groups))
            if i > 0:
                self.upsamples.append(
                    nn.Sequential(nn.Upsample(scale_factor=2, mode="nearest"),
                                  nn.Conv2d(dims[i], dims[i - 1], 3, padding=1))
                )
            else:
                self.upsamples.append(nn.Identity())

        self.out_proj = nn.Conv2d(dims[0], out_features, 3, padding=1)

    def forward(self, x):
        x = self.in_proj(x)

        skips = []
        for block, down in zip(self.down_blocks, self.downsamples):
            x = block(x)
            skips.append(x)
            x = down(x)

        x = self.mid_block1(x)
        x = self.mid_attn(x)
        x = self.mid_block2(x)

        for block, up in zip(self.up_blocks, self.upsamples):
            skip = skips.pop()
            if x.shape[-2:] != skip.shape[-2:]:
                x = F.interpolate(x, size=skip.shape[-2:], mode="bilinear", align_corners=False)
            x = block(torch.cat([x, skip], dim=1))
            x = up(x)

        return [self.out_proj(x)]
