"""Light from-scratch encoder for contour conditioning.

This is intentionally only the encoder half of a U-Net: full-resolution and
multi-scale skip features are exposed to the point decoder, but there is no
pixel decoder.  The bottleneck uses a ConvNeXt-style depthwise context block
rather than spatial self-attention, keeping compute roughly linear in image
size.
"""

import torch.nn as nn
import torch.nn.functional as F


class ResidualBlock(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, groups: int = 16):
        super().__init__()
        g_in = min(groups, in_ch)
        while in_ch % g_in != 0 and g_in > 1:
            g_in -= 1
        g_out = min(groups, out_ch)
        while out_ch % g_out != 0 and g_out > 1:
            g_out -= 1
        self.norm1 = nn.GroupNorm(g_in, in_ch)
        self.conv1 = nn.Conv2d(in_ch, out_ch, 3, padding=1)
        self.norm2 = nn.GroupNorm(g_out, out_ch)
        self.conv2 = nn.Conv2d(out_ch, out_ch, 3, padding=1)
        self.skip = nn.Conv2d(in_ch, out_ch, 1) if in_ch != out_ch else nn.Identity()

    def forward(self, x):
        h = self.conv1(F.silu(self.norm1(x)))
        h = self.conv2(F.silu(self.norm2(h)))
        return h + self.skip(x)


class LightContextBlock(nn.Module):
    """Large-kernel depthwise context with no quadratic spatial attention."""

    def __init__(self, channels: int, groups: int = 16, expansion: int = 2):
        super().__init__()
        g = min(groups, channels)
        while channels % g != 0 and g > 1:
            g -= 1
        hidden = channels * expansion
        self.norm = nn.GroupNorm(g, channels)
        self.dw = nn.Conv2d(channels, channels, 7, padding=3, groups=channels)
        self.pw1 = nn.Conv2d(channels, hidden, 1)
        self.pw2 = nn.Conv2d(hidden, channels, 1)

    def forward(self, x):
        h = self.dw(F.silu(self.norm(x)))
        h = self.pw2(F.gelu(self.pw1(h)))
        return x + h


class FeatureUNet(nn.Module):
    """Downsampling stages + lightweight bottleneck. Returns finest-first maps."""

    def __init__(self, in_channels=3, start_dim=64, dim_mults=(1, 2, 4), groups=16):
        super().__init__()
        dims = [start_dim * m for m in dim_mults]
        self.feature_channels = list(dims)
        self.in_proj = nn.Conv2d(in_channels, dims[0], 3, padding=1)

        self.down_blocks = nn.ModuleList()
        self.downsamples = nn.ModuleList()
        for i, dim in enumerate(dims):
            in_ch = dims[i - 1] if i > 0 else dims[0]
            self.down_blocks.append(ResidualBlock(in_ch, dim, groups))
            if i < len(dims) - 1:
                self.downsamples.append(nn.Conv2d(dim, dim, 3, stride=2, padding=1))
            else:
                self.downsamples.append(nn.Identity())

        self.mid_block1 = ResidualBlock(dims[-1], dims[-1], groups)
        self.mid_context = LightContextBlock(dims[-1], groups=groups)
        self.mid_block2 = ResidualBlock(dims[-1], dims[-1], groups)

    def forward(self, x):
        x = self.in_proj(x)
        skips = []
        for block, down in zip(self.down_blocks, self.downsamples):
            x = block(x)
            skips.append(x)
            x = down(x)
        bottleneck = self.mid_block2(self.mid_context(self.mid_block1(x)))
        return list(skips[:-1]) + [bottleneck]
