import sys
import os
import torch
import torch.nn as nn

# Pfad zum pretrained/-Ordner, der den dinov3-Paket-Ordner enthält
DINOV3_PKG_ROOT = "/loctmp/sit28238/SemanticSegmentationDiffusion/pretrained"
if DINOV3_PKG_ROOT not in sys.path:
    sys.path.insert(0, DINOV3_PKG_ROOT)

from dinov3.models.vision_transformer import vit_small, vit_base, vit_large  # bei Bedarf erweitern


_BUILDERS = {
    "dinov3_vits16": vit_small,
    "dinov3_vitb16": vit_base,
    "dinov3_vitl16": vit_large,
}


class DinoV3(nn.Module):
    """
    Analog zu LastVit: gibt eine Liste von [B, C, H, W]-Feature-Maps aus
    ausgewählten Transformer-Layern des DINOv3-Backbones zurück.

    Nutzt das Repo-eigene get_intermediate_layers(..., reshape=True), das
    Register-/CLS-Tokens bereits intern korrekt abschneidet.
    WICHTIG: n=[...] ist hier 0-indexiert (Block-Index), nicht 1-indexiert
    wie bei deinem LastVit!
    """

    def __init__(
        self,
        weights_path: str,
        model_name: str = "dinov3_vits16",
        layers: list = [2, 5, 8, 11],   # 1-indexiert, wie bei LastVit
        freeze: bool = True,
        use_layer_norms: bool = True,
    ):
        super().__init__()
        if model_name not in _BUILDERS:
            raise ValueError(f"Unbekannte Variante: {model_name}. Bekannt: {list(_BUILDERS.keys())}")

        self.layers = layers
        self._layers_0idx = [l - 1 for l in layers]
        self.freeze = freeze

        builder = _BUILDERS[model_name]
        self.model = builder(patch_size=16, n_storage_tokens=4)  # DINOv3 nutzt 4 Storage-Tokens
        self.hidden_dim = self.model.embed_dim

        state_dict = torch.load(weights_path, map_location="cpu")
        # Manche DINOv3-Checkpoints sind ein reines state_dict, manche in {"model": ...} verpackt --
        # hier defensiv beides abfangen, analog zum "model."-Prefix-Handling in deinem LastVit.
        if "model" in state_dict and isinstance(state_dict["model"], dict):
            state_dict = state_dict["model"]
        missing, unexpected = self.model.load_state_dict(state_dict, strict=False)
        print(f"[DINOv3] geladen aus {weights_path} | missing={len(missing)} unexpected={len(unexpected)}")

        self.patch_size = self.model.patch_size

        if self.freeze:
            self.model.eval()
            for p in self.model.parameters():
                p.requires_grad = False

        self.use_layer_norms = use_layer_norms
        if use_layer_norms:
            self.layer_norms = nn.ModuleList([
                nn.LayerNorm(self.hidden_dim) for _ in layers
            ])

    def forward(self, x: torch.Tensor) -> list:
        b, _, h, w = x.shape
        p = self.patch_size
        if h % p != 0 or w % p != 0:
            raise ValueError(f"Eingabegroesse {h}x{w} muss durch patch_size={p} teilbar sein.")

        ctx = torch.no_grad() if self.freeze else torch.enable_grad()
        with ctx:
            intermediates = self.model.get_intermediate_layers(
                x,
                n=self._layers_0idx,       # 0-indexierte Block-Indizes
                reshape=True,        # liefert direkt [B, C, H, W]
                return_class_token=False,
                norm=True,
            )

        selected_features = []
        for i, spatial_map in enumerate(intermediates):
            if self.use_layer_norms:
                bb, c, hh, ww = spatial_map.shape
                tokens = spatial_map.flatten(2).transpose(1, 2)      # [B, H*W, C]
                tokens = self.layer_norms[i](tokens)
                spatial_map = tokens.transpose(1, 2).reshape(bb, c, hh, ww)
            selected_features.append(spatial_map if self.freeze else spatial_map.clone())

        return selected_features