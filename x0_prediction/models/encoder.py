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
import re


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
        if backbone == "convnext_tiny":
            checkpoint_path="/loctmp/sit28238/SemanticSegmentationDiffusion/pretrained/convnext_tiny.pth"
        elif backbone == "convnext_base":
            checkpoint_path="/loctmp/sit28238/SemanticSegmentationDiffusion/pretrained/convnext_base.pth"
        elif backbone == "convnext_small":
            checkpoint_path="/loctmp/sit28238/SemanticSegmentationDiffusion/pretrained/convnext_small.pth"
        
        import timm  # local import: only needed for this encoder
        self.backbone = timm.create_model(
            backbone,
            pretrained=False,
            checkpoint_path=checkpoint_path,
            features_only=True,
            out_indices=(0, 1, 2, 3),
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

class ResNetConditioner(_PretrainedPyramid):
    """Frozen/fine-tuned timm ResNet-50 (ImageNet). Same interface as
    ConvNeXtConditioner. Feature maps come out NCHW already, no permute needed.
    Typical channels for resnet50 at out_indices (0,1,2,3): [256, 512, 1024, 2048]
    at strides [4, 8, 16, 32]."""

    def __init__(self, backbone="resnet50", pretrained=True, freeze=False, stem_dim=32):
        super().__init__()
        if backbone == "resnet50":
            checkpoint_path = "/loctmp/sit28238/SemanticSegmentationDiffusion/pretrained/resnet50_a1_0-14fe96d1.pth"
        else:
            raise ValueError(f"Unknown ResNet backbone '{backbone}'")

        import timm 
        self.backbone = timm.create_model(
            backbone,
            pretrained=False,
            # checkpoint_path entfernt, um den fehlerhaften Auto-Load zu umgehen
            features_only=True,
            # KORREKTUR: 1=layer1(256), 2=layer2(512), 3=layer3(1024), 4=layer4(2048)
            out_indices=(1, 2, 3, 4), 
        )

        # Gewichte manuell laden
        state_dict = torch.load(checkpoint_path, map_location="cpu")
        
        # Falls der Checkpoint ein "model"-Dict enthält (Sicherheitsabfrage)
        if "model" in state_dict:
            state_dict = state_dict["model"]

        # strict=False ignoriert den 'fc'-Head aus den Gewichten, der im Modell fehlt
        self.backbone.load_state_dict(state_dict, strict=False)

        self.feature_channels = list(self.backbone.feature_info.channels())
        cfg = getattr(self.backbone, "pretrained_cfg", None) or {}
        self._setup_norm_freeze(cfg.get("mean", (0.485, 0.456, 0.406)),
                                cfg.get("std", (0.229, 0.224, 0.225)), freeze)
        self._setup_stem(stem_dim)


class SwinConditioner(_PretrainedPyramid):
    """Frozen/fine-tuned timm Swin-T (ImageNet). Same interface as
    ConvNeXtConditioner, but Swin's `features_only` output is NHWC
    ([B, H, W, C]) instead of NCHW -- extract() is overridden to permute
    each scale before handing it to the point sampler.
    Typical channels for swin_tiny at out_indices (0,1,2,3): [96, 192, 384, 768]
    at strides [4, 8, 16, 32].

    Note: Swin needs input sizes divisible by 32 (window-partitioning
    constraint); this is generally fine at the 256px crops used elsewhere
    in the project, but keep it in mind if the crop size ever changes.
    """

    def __init__(self, backbone="swin_tiny_patch4_window7_224", pretrained=True,
                 freeze=False, stem_dim=32):
        super().__init__()
        if backbone == "swin_tiny_patch4_window7_224":
            checkpoint_path = "/loctmp/sit28238/SemanticSegmentationDiffusion/pretrained/swin_tiny_patch4_window7_224_22kto1k_finetune.pth"
        else:
            raise ValueError(f"Unknown Swin backbone '{backbone}'")

        import timm 
        # 1. Modell instanziieren, OHNE den checkpoint_path direkt zu übergeben
        self.backbone = timm.create_model(
            backbone,
            pretrained=False, 
            features_only=True,
            out_indices=(0, 1, 2, 3),
            img_size=256,
            dynamic_img_size=True
        )

        # 2. Checkpoint manuell in den Arbeitsspeicher laden
        state_dict = torch.load(checkpoint_path, map_location="cpu")
        # Offizielle Checkpoints verstecken die Parameter oft in einem "model"-Schlüssel
        if "model" in state_dict:
            state_dict = state_dict["model"]

        # 3. Mapping der Parameter-Namen durchführen
        mapped_state_dict = {}
        for k, v in state_dict.items():
            if "downsample" in k:
                # Verschiebt den Index für Downsampling um +1 (layers.0 -> layers_1)
                new_k = re.sub(r"^layers\.(\d+)\.downsample", 
                               lambda m: f"layers_{int(m.group(1)) + 1}.downsample", k)
            else:
                # Standard-Mapping für Attention-Blocks und andere Layer
                new_k = re.sub(r"^layers\.(\d+)\.", r"layers_\1.", k)
            
            mapped_state_dict[new_k] = v

        # 4. Angepasstes Dictionary laden
        # strict=False ist notwendig, da features_only=True den Classification-Head 
        # des Modells entfernt, dieser aber noch in den Gewichten existiert.
        self.backbone.load_state_dict(mapped_state_dict, strict=False)

        self.feature_channels = list(self.backbone.feature_info.channels())
        cfg = getattr(self.backbone, "pretrained_cfg", None) or {}
        self._setup_norm_freeze(cfg.get("mean", (0.485, 0.456, 0.406)),
                                cfg.get("std", (0.229, 0.224, 0.225)), freeze)
        self._setup_stem(stem_dim)

    def extract(self, image):
        x = self._preprocess(image)
        if self.freeze:
            with torch.no_grad():
                feats = list(self.backbone(x))
        else:
            feats = list(self.backbone(x))
        # NHWC -> NCHW per scale (Swin-specific; ConvNeXt/ResNet are NCHW already)
        feats = [f.permute(0, 3, 1, 2).contiguous() for f in feats]
        if getattr(self, "stem", None) is not None:
            feats = [self.stem(x)] + feats          # full-res scale, finest first
        return feats

class VMambaConditioner(_PretrainedPyramid):

    """Frozen/fine-tuned VMamba-T[s1l8] (ImageNet). Same interface as
    ConvNeXtConditioner. Backbone_VSSM already returns NCHW features when
    channel_first=True (norm_layer='ln2d'), so no permute needed.
    Channels for vmamba_tiny_s1l8 at out_indices (0,1,2,3): [96, 192, 384, 768]
    at strides [4, 8, 16, 32]."""

    def __init__(self, pretrained=True, freeze=False, stem_dim=32,
                 checkpoint_path="...", force_torch_backend=False):  # <-- Default jetzt True
        super().__init__()
        from . import vmamba

        #if force_torch_backend:
        #    vmamba.WITH_TRITON = False
        #    vmamba.WITH_SELECTIVESCAN_MAMBA = False  # ist eh schon False, schadet aber nicht

        self.backbone = vmamba.Backbone_VSSM(
            out_indices=(0, 1, 2, 3),
            pretrained=None,  # we load weights manually below
            depths=[2, 2, 8, 2],
            dims=96,
            ssm_d_state=1, ssm_ratio=1.0, ssm_dt_rank="auto", ssm_act_layer="silu",
            ssm_conv=3, ssm_conv_bias=False, ssm_drop_rate=0.0,
            ssm_init="v0", forward_type="v05_noz",
            mlp_ratio=4.0, mlp_act_layer="gelu", mlp_drop_rate=0.0,
            patch_norm=True, norm_layer="ln2d",
            downsample_version="v3", patchembed_version="v2",
            use_checkpoint=False, posembed=False, imgsize=224,
        )
        if pretrained:
            state_dict = torch.load(checkpoint_path, map_location="cpu")
            if "model" in state_dict:
                state_dict = state_dict["model"]
            incompatible = self.backbone.load_state_dict(state_dict, strict=False)
            print(f"VMambaConditioner: loaded {checkpoint_path}, incompatible keys: {incompatible}")
        self.feature_channels = list(self.backbone.dims)  # [96, 192, 384, 768]
        # VMamba normalizes like standard ImageNet models
        self._setup_norm_freeze((0.485, 0.456, 0.406), (0.229, 0.224, 0.225), freeze)
        self._setup_stem(stem_dim)



import torch
import torch.nn as nn
import torch.nn.functional as F



def build_conditioner(cfg):
    if cfg.encoder == "convnext":
        return ConvNeXtConditioner(
            backbone=cfg.backbone, pretrained=cfg.pretrained, freeze=cfg.freeze_backbone,
            stem_dim=cfg.stem_dim,
        )
    elif cfg.encoder == "pvt":
        return PVTConditioner(
            variant=cfg.pvt_variant,
            pretrained_path = "/loctmp/sit28238/SemanticSegmentationDiffusion/pretrained/pvt_v2_b2.pth",
            freeze=cfg.freeze_backbone, stem_dim=cfg.stem_dim,
        )
    elif cfg.encoder == "resnet":
        return ResNetConditioner(
            backbone="resnet50",
            pretrained=cfg.pretrained, freeze=cfg.freeze_backbone,
            stem_dim=cfg.stem_dim,
        )
    elif cfg.encoder == "swin":
        return SwinConditioner(
            backbone="swin_tiny_patch4_window7_224",
            pretrained=cfg.pretrained, freeze=cfg.freeze_backbone,
            stem_dim=cfg.stem_dim,
        )
    elif cfg.encoder == "vmamba":
        return VMambaConditioner(
            pretrained=cfg.pretrained, freeze=cfg.freeze_backbone,
            stem_dim=cfg.stem_dim,
            checkpoint_path=getattr(cfg, "vmamba_checkpoint",
                "/loctmp/sit28238/SemanticSegmentationDiffusion/pretrained/vssm1_tiny_0230s_ckpt_epoch_264.pth"),
        )
        
    raise ValueError(f"Unknown encoder '{cfg.encoder}' (expected 'convnext', 'pvt', or 'unet').")
