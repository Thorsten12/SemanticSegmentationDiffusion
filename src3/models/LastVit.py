import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import vit_b_16


class LastVit(nn.Module):
    def __init__(
        self,
        ckpt_path: str = "/loctmp/sit28238/SemanticSegmentationDiffusion/pretrained/ViT_190k.pth",
        layers: list = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12],
        freeze: bool = True,
        use_layer_norms: bool = True,
    ):
        super().__init__()
        self.ckpt_path = ckpt_path
        self.layers = layers
        self.freeze = freeze
        self.patch_size = 16
        self.hidden_dim = 768

        self.model = vit_b_16(weights=None)

        checkpoint = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        state_dict = {k.removeprefix("model."): v for k, v in checkpoint["model"].items()}
        self.model.load_state_dict(state_dict)

        if self.freeze:
            self.model.eval()
            for param in self.model.parameters():
                param.requires_grad = False

        self.use_layer_norms = use_layer_norms
        if use_layer_norms:
            self.layer_norms = nn.ModuleList([
                nn.LayerNorm(self.hidden_dim) for _ in layers
            ])

    def _interpolated_pos_embedding(self, gh: int, gw: int) -> torch.Tensor:
        """Bikubische Interpolation der Positional Embeddings für Eingaben != 224x224."""
        pos_embedding = self.model.encoder.pos_embedding
        cls_pos, patch_pos = pos_embedding[:, :1], pos_embedding[:, 1:]

        n0 = patch_pos.shape[1]
        gh0 = gw0 = int(n0 ** 0.5)
        if gh0 * gw0 != n0:
            raise ValueError("Urspruengliches Positional-Embedding ist nicht quadratisch.")

        if (gh0, gw0) == (gh, gw):
            return pos_embedding

        c = patch_pos.shape[-1]
        patch_pos = patch_pos.reshape(1, gh0, gw0, c).permute(0, 3, 1, 2)
        patch_pos = F.interpolate(patch_pos, size=(gh, gw), mode="bicubic", align_corners=False)
        patch_pos = patch_pos.permute(0, 2, 3, 1).reshape(1, gh * gw, c)
        return torch.cat([cls_pos, patch_pos], dim=1)

    def forward(self, x: torch.Tensor) -> list:
        b, _, h, w = x.shape
        p = self.patch_size
        if h % p != 0 or w % p != 0:
            raise ValueError(f"Eingabegroesse {h}x{w} muss durch patch_size={p} teilbar sein.")
        gh, gw = h // p, w // p

        ctx = torch.no_grad() if self.freeze else torch.enable_grad()
        with ctx:
            tokens = self.model.conv_proj(x)
            tokens = tokens.reshape(b, self.hidden_dim, gh * gw).permute(0, 2, 1)

            batch_class_token = self.model.class_token.expand(b, -1, -1)
            tokens = torch.cat([batch_class_token, tokens], dim=1)
            tokens = tokens + self._interpolated_pos_embedding(gh, gw)

            hidden_states = tokens
            raw_selected = []
            for idx, layer in enumerate(self.model.encoder.layers):
                hidden_states = layer(hidden_states)
                layer_num = idx + 1  # API bleibt 1-indexiert, wie bisher
                if layer_num in self.layers:
                    patch_tokens = hidden_states[:, 1:, :]
                    raw_selected.append(
                        patch_tokens if self.freeze else patch_tokens.clone()
                    )

        selected_features = []
        for i, patch_tokens in enumerate(raw_selected):
            if self.use_layer_norms:
                patch_tokens = self.layer_norms[i](patch_tokens)
            spatial_map = patch_tokens.transpose(1, 2).reshape(b, self.hidden_dim, gh, gw)
            selected_features.append(spatial_map)

        return selected_features


# --- TEST DER KLASSE ---
if __name__ == "__main__":
    backbone = LastVit(layers=[3, 6, 9, 12])
    x = torch.randn(2, 3, 224, 224)
    feats = backbone(x)
    for i, f in enumerate(feats):
        print(f"Layer {[3,6,9,12][i]}: mean={f.mean():.3f} std={f.std():.3f} abs_max={f.abs().max():.3f}")

    # Test mit anderer Eingabegröße
    x2 = torch.randn(2, 3, 384, 384)
    feats2 = backbone(x2)
    for i, f in enumerate(feats2):
        print(f"384x384 Layer {[3,6,9,12][i]}: shape={tuple(f.shape)}")