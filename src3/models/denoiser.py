import torch
import torch.nn as nn
import torch.nn.functional as F


import math

from ..utils import timestep_encoding, positional_encoding, order_encoding


class PerPointSampler(nn.Module):
    """
    Fragt für jeden Kontur-Punkt an seiner (x,y)-Position ALLE Backbone-Skalen ab
    und fasst sie zu einem einzigen Feature-Vektor PRO PUNKT zusammen.
    """
    def __init__(self, feature_channels, proj_dim=64, out_dim=128):
        super().__init__()
        self.projs = nn.ModuleList(nn.Linear(c, proj_dim) for c in feature_channels)
        self.mlp = nn.Sequential(
            nn.Linear(len(feature_channels) * proj_dim, out_dim),
            nn.GELU(),
            nn.Linear(out_dim, out_dim),
        )

    def forward(self, feats, points):
        grid = torch.clamp(points, -1.0, 1.0).unsqueeze(1)   # [B, 1, N, 2]

        pooled = []
        for proj, f in zip(self.projs, feats):
            s = F.grid_sample(f, grid, mode="bilinear",
                              padding_mode="border", align_corners=True)  # [B, C_i, 1, N]
            s = s.squeeze(2).transpose(1, 2)                              # [B, N, C_i]
            s = proj(s)                                                    # [B, N, proj_dim]
            pooled.append(s)

        x = torch.cat(pooled, dim=-1)
        return self.mlp(x)


class ContourDenoiser(nn.Module):
    def __init__(self, n_point=200, hidden_dim=128, coord_fourier_bands=12,
                 feature_channels=None, proj_dim=64, num_heads:int=4, num_layers:int=4):
        super().__init__()
        if feature_channels is None:
            raise ValueError("feature_channels darf nicht None sein — "
                             "encoder.feature_channels an build_models übergeben!")

        self.n_point = n_point
        self.hidden_dim = hidden_dim

        self.timestep_mlp = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim), nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.positional_mlp = nn.Sequential(
            nn.Linear(2 * coord_fourier_bands * 2, hidden_dim), nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.order_mlp = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim), nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.sampler = PerPointSampler(feature_channels, proj_dim=proj_dim, out_dim=hidden_dim)

        fused_dim = hidden_dim * 4

        self.local_conv1 = nn.Conv1d(fused_dim, fused_dim, 3, padding=1, padding_mode="circular")
        
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=fused_dim, nhead=num_heads, dim_feedforward=hidden_dim * 2, batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        self.proj_to_hidden = nn.Sequential(
            nn.Linear(fused_dim, hidden_dim),
            nn.GELU()
        )
        
        self.local_conv2 = nn.Conv1d(hidden_dim, hidden_dim, 3, padding=1, padding_mode="circular")    

        self.temp = nn.Parameter(torch.ones(fused_dim, 1) * 1e-3)

        self.out_mlp = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim), 
            nn.GELU(),
            nn.Linear(hidden_dim, 2),
        )

    def forward(self, x, t, cond, drop_mask=None):
        """cond: Liste von Feature-Maps [B, C_i, H_i, W_i], NICHT ein Vektor!"""
        B, N, _ = x.shape

        t_vec = self.timestep_mlp(timestep_encoding(t, self.hidden_dim))
        pos_vec = self.positional_mlp(positional_encoding(x, in_dim=2, num_bands=12))

        cond_vec = self.sampler(cond, x)
        # CFG Logik für das spätere Sampling integriert
        if drop_mask is not None:
            keep = (~drop_mask).float().view(B, 1, 1)
            cond_vec = cond_vec * keep

        order = torch.arange(N, device=x.device)
        order_vec = self.order_mlp(order_encoding(order, self.hidden_dim))
        order_vec = order_vec.unsqueeze(0).expand(B, -1, -1)

        t_vec = t_vec.unsqueeze(1).expand(-1, N, -1)

        # Features zusammenführen -> [B, N, fused_dim]
        feat = torch.cat([t_vec, pos_vec, cond_vec, order_vec], dim=-1)

        # 1. Local Conv 1
        feat_t = feat.transpose(1, 2)  # [B, fused_dim, N]
        feat_t = feat_t + self.temp * F.gelu(self.local_conv1(feat_t))
        feat = feat_t.transpose(1, 2)  # [B, N, fused_dim]

        # 2. Transformer
        feat = self.transformer(feat)  # [B, N, fused_dim]

        # Features von fused_dim auf hidden_dim projizieren
        feat = self.proj_to_hidden(feat)  # [B, N, hidden_dim]

        # 3. Local Conv 2 (jetzt mit passenden Kanälen)
        feat_t = feat.transpose(1, 2)  # [B, hidden_dim, N]
        feat_t = feat_t + F.gelu(self.local_conv2(feat_t))
        feat = feat_t.transpose(1, 2)  # [B, N, hidden_dim]

        # 4. Finales Output MLP am Stück aufrufen
        x = self.out_mlp(feat)  # [B, N, 2]
        
        pred_noise = torch.clamp(x, -3.0, 3.0)
        return pred_noise


if __name__ == "__main__":
    N = 200
    B = 1
    d_model = 128
    x_0 = torch.randn(B, N, 2)
    t = torch.tensor([0], dtype=torch.int64)

    # Dummy feature_channels + Dummy feature maps zum Testen
    feature_channels = [96, 192, 384, 768]
    dummy_feats = [
        torch.randn(B, 96, 56, 56),
        torch.randn(B, 192, 28, 28),
        torch.randn(B, 384, 14, 14),
        torch.randn(B, 768, 7, 7),
    ]

    cont = ContourDenoiser(n_point=N, hidden_dim=d_model, feature_channels=feature_channels)
    out = cont.forward(x_0, t, dummy_feats)
    print(out.shape)   # erwartet: [1, 200, 2]