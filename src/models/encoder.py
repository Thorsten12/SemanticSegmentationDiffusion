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
    raise ValueError(f"Unknown encoder '{cfg.encoder}' (expected 'convnext', 'pvt', or 'unet').")
