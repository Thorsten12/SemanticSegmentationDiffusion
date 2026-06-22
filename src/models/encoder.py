"""Conditioning encoders: image -> raw multi-scale feature pyramid.

Each encoder exposes a uniform interface:

    extract(image)      -> list of feature maps (the raw pyramid)  (run ONCE per image)
    fuse(raw, t)        -> raw (identity; kept for interface symmetry)
    forward(image, t)   -> extract(image)
    .feature_channels   -> list of per-scale channel counts

The actual multi-scale combination is done PER POINT and PER TIMESTEP inside the
denoiser's `MultiScalePointSampler` (sample every scale at each point, gate scales
by time, MLP-summarize). So the encoder's only job is to produce a good pyramid;
the heavy backbone runs once per image while the cheap per-point query/gate runs
each diffusion step.

  * ConvNeXtConditioner : frozen/fine-tuned timm ConvNeXt (ImageNet).
  * PVTConditioner      : pvt_v2 (local models/pvtv2.py) — a transformer pyramid.
  * UNetConditioner     : from-scratch FeatureUNet (single-scale, for ablation).
"""

import os

import torch
import torch.nn as nn

from .feature_unet import FeatureUNet


class _PretrainedPyramid(nn.Module):
    """Shared logic for pretrained pyramid backbones (ImageNet norm + freeze)."""

    def _setup_norm_freeze(self, mean, std, freeze):
        self.register_buffer("norm_mean", torch.tensor(mean).view(1, 3, 1, 1))
        self.register_buffer("norm_std", torch.tensor(std).view(1, 3, 1, 1))
        self.freeze = freeze
        if freeze:
            for p in self.backbone.parameters():
                p.requires_grad_(False)
            self.backbone.eval()

    def _setup_stem(self, stem_dim):
        """Optional full-resolution learnable feature map prepended to the pyramid.

        Pretrained backbones only emit features down to stride 4 (finest 56x56 at
        224), capping how precisely a boundary point can read image cues. This
        small conv stem produces a stride-1 (full-res) feature map so points get
        pixel-precise local cues *in addition* to the backbone's semantics. Always
        trainable, even when the backbone is frozen.
        """
        if stem_dim and stem_dim > 0:
            g = min(8, stem_dim)
            self.stem = nn.Sequential(
                nn.Conv2d(3, stem_dim, 3, padding=1),
                nn.GroupNorm(g, stem_dim), nn.SiLU(),
                nn.Conv2d(stem_dim, stem_dim, 3, padding=1),
            )
            self.feature_channels = [stem_dim] + list(self.feature_channels)
        else:
            self.stem = None

    def _preprocess(self, image):
        """Dataset images are in [-1, 1]; map to [0, 1] then ImageNet-normalize."""
        x = image * 0.5 + 0.5
        return (x - self.norm_mean) / self.norm_std

    def extract(self, image):
        x = self._preprocess(image)
        if self.freeze:
            with torch.no_grad():
                feats = list(self.backbone(x))
        else:
            feats = list(self.backbone(x))
        if getattr(self, "stem", None) is not None:
            feats = [self.stem(x)] + feats          # full-res scale, finest first
        return feats

    def fuse(self, feats, t):
        return feats                     # combination happens per-point in the denoiser

    def forward(self, image, t):
        return self.extract(image)

    def train(self, mode=True):
        super().train(mode)
        if self.freeze:
            self.backbone.eval()         # keep frozen backbone (and norm stats) in eval
        return self


class ConvNeXtConditioner(_PretrainedPyramid):
    def __init__(self, backbone="convnext_tiny", pretrained=True, freeze=False, stem_dim=32):
        super().__init__()
        import timm  # local import: only needed for this encoder
        self.backbone = timm.create_model(
            backbone, pretrained=pretrained, features_only=True, out_indices=(0, 1, 2, 3)
        )
        self.feature_channels = list(self.backbone.feature_info.channels())
        cfg = getattr(self.backbone, "pretrained_cfg", None) or {}
        self._setup_norm_freeze(cfg.get("mean", (0.485, 0.456, 0.406)),
                                cfg.get("std", (0.229, 0.224, 0.225)), freeze)
        self._setup_stem(stem_dim)


class PVTConditioner(_PretrainedPyramid):
    """Pyramid Vision Transformer v2 (local pvtv2.py). pvt_v2_b2: channels
    [64,128,320,512] at strides [4,8,16,32] (finest = image/4)."""

    def __init__(self, variant="pvt_v2_b2", pretrained_path=None, freeze=False, stem_dim=32):
        super().__init__()
        from . import pvtv2
        self.backbone = getattr(pvtv2, variant)()
        if pretrained_path and os.path.isfile(pretrained_path):
            sd = torch.load(pretrained_path, map_location="cpu")
            sd = {k: v for k, v in sd.items() if not k.startswith("head.")}
            self.backbone.load_state_dict(sd, strict=False)
        self.backbone.eval()
        with torch.no_grad():
            self.feature_channels = [f.shape[1] for f in self.backbone(torch.zeros(1, 3, 64, 64))]
        # PVT uses standard ImageNet normalization.
        self._setup_norm_freeze((0.485, 0.456, 0.406), (0.229, 0.224, 0.225), freeze)
        self._setup_stem(stem_dim)


class UNetConditioner(nn.Module):
    """Single-scale conditioner wrapping the from-scratch FeatureUNet (ablation)."""

    def __init__(self, in_channels=3, cond_dim=64, start_dim=64,
                 dim_mults=(1, 2, 4), groups=16):
        super().__init__()
        self.unet = FeatureUNet(in_channels, start_dim, dim_mults, cond_dim, groups)
        self.feature_channels = [cond_dim]
        self.freeze = False

    def extract(self, image):
        return self.unet(image)          # list of one map

    def fuse(self, feats, t):
        return feats

    def forward(self, image, t):
        return self.unet(image)

class UNetDecoderBlock(nn.Module):
    """Standard U-Net decoder block: Upsample -> Concat skip -> Conv."""
    def __init__(self, in_channels, skip_channels, out_channels):
        super().__init__()
        self.upsample = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True)
        # Account for concatenated channels from the skip connection
        combined_channels = in_channels + skip_channels if skip_channels > 0 else in_channels
        self.conv = nn.Sequential(
            nn.Conv2d(combined_channels, out_channels, kernel_size=3, padding=1),
            nn.GroupNorm(min(8, out_channels), out_channels),
            nn.SiLU(),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.GroupNorm(min(8, out_channels), out_channels),
            nn.SiLU()
        )

    def forward(self, x, skip=None):
        x = self.upsample(x)
        if skip is not None:
            x = torch.cat([x, skip], dim=1)
        return self.conv(x)


class BackboneUNetConditioner(_PretrainedPyramid):
    """Full U-Net conditioner using a pretrained timm backbone as the encoder."""

    def __init__(self, backbone_name="convnext_tiny", pretrained=True, freeze=False, decoder_dim=64):
        super().__init__()
        import timm
        
        # 1. Setup the pretrained Encoder (e.g., ConvNeXt)
        self.backbone = timm.create_model(
            backbone_name, pretrained=pretrained, features_only=True, out_indices=(0, 1, 2, 3)
        )
        encoder_channels = list(self.backbone.feature_info.channels()) # e.g., [96, 192, 384, 768]
        
        # Setup normalization and freeze logic from base class
        cfg = getattr(self.backbone, "pretrained_cfg", None) or {}
        self._setup_norm_freeze(cfg.get("mean", (0.485, 0.456, 0.406)),
                                cfg.get("std", (0.229, 0.224, 0.225)), freeze)
        
        # 2. Setup the U-Net Decoder Blocks
        # Top block handles Stride 32 -> Stride 16
        self.dec1 = UNetDecoderBlock(encoder_channels[3], encoder_channels[2], decoder_dim * 4)
        # Conv block handles Stride 16 -> Stride 8
        self.dec2 = UNetDecoderBlock(decoder_dim * 4, encoder_channels[1], decoder_dim * 2)
        # Conv block handles Stride 8 -> Stride 4
        self.dec3 = UNetDecoderBlock(decoder_dim * 2, encoder_channels[0], decoder_dim)
        
        # Final upsampling stages to reach full resolution (Stride 2 and Stride 1)
        # Since the backbone has no skip connections here, we pass skip=None
        self.dec4 = UNetDecoderBlock(decoder_dim, 0, decoder_dim) # Stride 4 -> Stride 2
        self.dec5 = UNetDecoderBlock(decoder_dim, 0, decoder_dim) # Stride 2 -> Stride 1

        # Define the channels of the output pyramid that the denoiser will sample from.
        # We output a multi-scale refined pyramid: [Stride 1, Stride 4, Stride 16]
        self.feature_channels = [decoder_dim, decoder_dim, decoder_dim * 4]

    def extract(self, image):
        """Passes image through backbone encoder and U-Net decoder to get refined features."""
        x = self._preprocess(image)
        
        # Forward pass through the backbone (Encoder)
        if self.freeze:
            with torch.no_grad():
                feats = list(self.backbone(x))
        else:
            feats = list(self.backbone(x))
            
        # Extract individual scale maps from the encoder pyramid
        skip_s4, skip_s8, skip_s16, skip_s32 = feats

        # Backward pass through the U-Net Decoder using skip connections
        s16_refined = self.dec1(skip_s32, skip_s16)
        s8_refined = self.dec2(s16_refined, skip_s8)
        s4_refined = self.dec3(s8_refined, skip_s4)
        
        # Upsample to full resolution without backbone skips
        s2_refined = self.dec4(s4_refined, skip=None)
        s1_refined = self.dec5(s2_refined, skip=None)

        # Return a balanced multi-scale pyramid back to the point sampler
        return [s1_refined, s4_refined, s16_refined]

    def fuse(self, feats, t):
        return feats

    def forward(self, image, t):
        return self.extract(image)


def build_conditioner(cfg):
    if cfg.encoder == "convnext":
        return ConvNeXtConditioner(
            backbone=cfg.backbone, pretrained=cfg.pretrained, freeze=cfg.freeze_backbone,
            stem_dim=cfg.stem_dim,
        )
    elif cfg.encoder == "pvt":
        return PVTConditioner(
            variant=cfg.pvt_variant,
            pretrained_path=(cfg.pvt_pretrained_path if cfg.pretrained else None),
            freeze=cfg.freeze_backbone, stem_dim=cfg.stem_dim,
        )
    elif cfg.encoder == "unet":
        return UNetConditioner(
            in_channels=cfg.in_channels, cond_dim=cfg.cond_channels,
            start_dim=cfg.unet_start_dim, dim_mults=cfg.unet_dim_mults,
            groups=cfg.unet_groupnorm_groups,
        )
    elif cfg.encoder == "convnext_unet":  # New option for the full U-Net architecture
        return BackboneUNetConditioner(
            backbone_name=cfg.backbone, 
            pretrained=cfg.pretrained, 
            freeze=cfg.freeze_backbone,
            decoder_dim=cfg.cond_channels  # Uses your config's base channel dimension
        )
    raise ValueError(f"Unknown encoder '{cfg.encoder}' (expected 'convnext', 'pvt', or 'unet').")
