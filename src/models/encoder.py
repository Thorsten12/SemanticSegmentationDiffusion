"""Image -> multi-scale feature pyramid (finest first).

Backbones only produce 2D features. DP2Seg consumes the list at contour points.

  * ConvNeXtConditioner : timm ConvNeXt (ImageNet)
  * PVTConditioner      : PVT-v2 pyramid
  * TimmConditioner     : any timm `features_only` model
  * UNetConditioner     : from-scratch encoder + bottleneck
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
        """Full-resolution stem so points can read pixel-precise cues (stride 1).

        Pretrained pyramids stop at stride 4. Always trainable, even if frozen.
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
            feats = [self.stem(x)] + feats
        return feats

    def forward(self, image, t=None):
        return self.extract(image)

    def train(self, mode=True):
        super().train(mode)
        if self.freeze:
            self.backbone.eval()
        return self


class ConvNeXtConditioner(_PretrainedPyramid):
    def __init__(self, backbone="convnext_tiny", pretrained=True, freeze=False, stem_dim=32):
        super().__init__()
        import timm
        self.backbone = timm.create_model(
            backbone, pretrained=pretrained, features_only=True, out_indices=(0, 1, 2, 3),
        )
        self.feature_channels = list(self.backbone.feature_info.channels())
        cfg = getattr(self.backbone, "pretrained_cfg", None) or {}
        self._setup_norm_freeze(
            cfg.get("mean", (0.485, 0.456, 0.406)),
            cfg.get("std", (0.229, 0.224, 0.225)), freeze,
        )
        self._setup_stem(stem_dim)


class TimmConditioner(ConvNeXtConditioner):
    """Any timm features_only pyramid (ResNet, EfficientNet, ...)."""


class PVTConditioner(_PretrainedPyramid):
    """PVT-v2-b2: channels [64,128,320,512] at strides [4,8,16,32]."""

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
        self._setup_norm_freeze((0.485, 0.456, 0.406), (0.229, 0.224, 0.225), freeze)
        self._setup_stem(stem_dim)


class UNetConditioner(nn.Module):
    """From-scratch U-Net encoder + bottleneck (no pixel decoder)."""

    def __init__(self, in_channels=3, start_dim=64, dim_mults=(1, 2, 4), groups=16):
        super().__init__()
        self.unet = FeatureUNet(in_channels, start_dim, dim_mults, groups)
        self.feature_channels = list(self.unet.feature_channels)
        self.freeze = False

    def extract(self, image):
        return self.unet(image)

    def forward(self, image, t=None):
        return self.unet(image)


def build_conditioner(cfg):
    stem = cfg.stem_dim
    if cfg.encoder == "convnext":
        return ConvNeXtConditioner(
            backbone=cfg.backbone, pretrained=cfg.pretrained,
            freeze=cfg.freeze_backbone, stem_dim=stem,
        )
    if cfg.encoder == "timm":
        return TimmConditioner(
            backbone=cfg.backbone, pretrained=cfg.pretrained,
            freeze=cfg.freeze_backbone, stem_dim=stem,
        )
    if cfg.encoder == "pvt":
        return PVTConditioner(
            variant=cfg.pvt_variant,
            pretrained_path=(cfg.pvt_pretrained_path if cfg.pretrained else None),
            freeze=cfg.freeze_backbone, stem_dim=stem,
        )
    if cfg.encoder == "unet":
        return UNetConditioner(
            in_channels=cfg.in_channels,
            start_dim=cfg.unet_start_dim, dim_mults=cfg.unet_dim_mults,
            groups=cfg.unet_groupnorm_groups,
        )
    raise ValueError(
        f"Unknown encoder '{cfg.encoder}' (expected convnext, pvt, unet, timm)."
    )
