
import torch
import torch.nn as nn

from ..utils import timestep_encoding
from .LastVit import LastVit

class _PerPointFusion(nn.Module):
    """
    Gibt KEINEN gepoolten Vektor zurück, sondern die zeit-gewichteten,
    räumlich noch aufgelösten Feature-Maps -- für grid_sample pro Punkt.
    """
    def __init__(self, feature_channels, time_dim=128, hidden_dim=128):
        super().__init__()
        self.num_stages = len(feature_channels)
        self.time_dim = time_dim

        self.time_mlp = nn.Sequential(
            nn.Linear(time_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, self.num_stages)
        )

    def forward(self, feats, t):
        t_vec = timestep_encoding(t, dim=self.time_dim)
        stage_weights = self.time_mlp(t_vec)                # [B, num_stages]
        stage_weights = torch.softmax(stage_weights, dim=-1)

        weighted_feats = []
        for i, f in enumerate(feats):
            w = stage_weights[:, i].view(-1, 1, 1, 1)        # [B,1,1,1] -> broadcastbar auf (B,C,H,W)
            weighted_feats.append(f * w)

        return weighted_feats   # Liste von [B, C_i, H_i, W_i] 


class _Backbones(nn.Module):
    def _setup_norm_freeze(self, mean, std, freeze):
        self.register_buffer("mean", torch.tensor(mean).view(1,3,1,1))
        self.register_buffer("std", torch.tensor(std).view(1,3,1,1))
        self.freeze = freeze
        if freeze:
            for p in self.backbone.parameters():
                p.requires_grad = False
            self.backbone.eval()
        self.fusion = _PerPointFusion(self.feature_channels, hidden_dim=128)

    def _preprocess(self, x):
        x = x * 0.5 + 0.5  # Scale to [0, 1]
        x = (x - self.mean) / self.std  # Normalize
        return x

    def extract(self, x):
        x = self._preprocess(x)
        if self.freeze:
            with torch.no_grad():
                feats = list(self.backbone(x))
        else:
            feats = list(self.backbone(x))
        
        return feats

    def fuse(self, feats, t):
        return self.fusion(feats, t)

    def forward(self, x, t):
        feats = self.extract(x)
        return self.fuse(feats, t)
    
    def train(self, mode=True):
        super().train(mode)
        if self.freeze:
            self.backbone.eval()
        return self

class ConvNextConditioner(_Backbones):
    """
    returns B, channels, height, wight
    """
    def __init__(self, freeze = False):
        super().__init__()
        import timm
        self.backbone = timm.create_model("convnext_tiny", features_only=True, out_indices=(0, 1, 2, 3))

        self.feature_channels = self.backbone.feature_info.channels()
        cfg = getattr(self.backbone, "pretrained_cfg", None) or {}
        self._setup_norm_freeze(cfg.get("mean", (0.485, 0.456, 0.406)), 
                                cfg.get("std", (0.229, 0.224, 0.225)), freeze)

class LastVitConditioner(_Backbones):
    def __init__(
        self,
        ckpt_path: str = "/loctmp/sit28238/SemanticSegmentationDiffusion/pretrained/ViT_190k.pth",
        layers: list = [3, 6, 9, 12],
        freeze: bool = True,
        use_layer_norms: bool = True,
    ):
        super().__init__()

        self.backbone = LastVit(
            ckpt_path=ckpt_path,
            layers=layers,
            freeze=freeze,              
            use_layer_norms=use_layer_norms,
        )

        self.feature_channels = [768 for _ in layers]

        self._setup_norm_freeze(
            mean=(0.485, 0.456, 0.406),
            std=(0.229, 0.224, 0.225),
            freeze=freeze
        )

        if freeze and use_layer_norms:
            for p in self.backbone.layer_norms.parameters():
                p.requires_grad = True


def build_conditioner(cfg):
    if cfg.encoder == "convnext":
        return ConvNextConditioner(
            freeze=cfg.freeze
        )
    if cfg.encoder == "lastVit":
        return LastVitConditioner(  
            ckpt_path=getattr(cfg, "ckpt_path", "/loctmp/sit28238/SemanticSegmentationDiffusion/pretrained/ViT_190k.pth"),
            layers=getattr(cfg, "layers", [1,2,3,4,5,6,7,8,9,10,11,12]),  
            freeze=cfg.freeze
        )
    raise ValueError(f"Unknown encoder type: {cfg.encoder}")


