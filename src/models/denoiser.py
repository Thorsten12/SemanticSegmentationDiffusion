"""Boundary-point denoiser with per-point, time-gated multi-scale conditioning.

Given the current (noisy) point set, the diffusion timestep, and the *raw
multi-scale* feature pyramid from the image backbone, predict the clean points x0.

Conditioning (the key design):
  * For each point, `grid_sample` (bilinear-interpolate) EVERY backbone scale at
    that point's (x, y). Coarse scales (e.g. 7x7) give global context; fine scales
    (e.g. 56x56) give precise local cues. No up-front fusion to a single map.
  * A softmax over scales, computed from the timestep, weights the scales per
    point ("coarse/global early near pure noise, fine/local late near the end of
    sampling"). The weighted per-scale features are concatenated and an MLP (also
    fed the timestep) summarizes them to one guidance vector per point.

Sequence model (closed ordered contour of N points):
  sinusoidal timestep embedding + fixed positional embedding over point index;
  circular Conv1d (the contour wraps around) -> Transformer encoder -> Conv1d;
  MLP head -> 2D coordinate prediction. Coordinates enter only via an additive
  Fourier path (kept OUT of the content fusion to avoid memorizing tiny datasets).
"""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from .positional import FourierFeatures


def timestep_embedding(t: torch.Tensor, dim: int, max_period: int = 10000) -> torch.Tensor:
    """Standard sinusoidal embedding of integer timesteps -> [B, dim]."""
    half = dim // 2
    freqs = torch.exp(
        -math.log(max_period) * torch.arange(half, device=t.device, dtype=torch.float32) / half
    )
    args = t.float()[:, None] * freqs[None]
    emb = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
    if dim % 2:  # pad odd dims
        emb = F.pad(emb, (0, 1))
    return emb


class MultiScalePointSampler(nn.Module):
    """Per-point query of a multi-scale feature pyramid, gated by the timestep."""

    def __init__(self, scale_channels, proj_dim=64, out_dim=128, time_dim=128):
        super().__init__()
        self.n_scales = len(scale_channels)
        # Project each scale to a common width so concatenation is balanced.
        self.projs = nn.ModuleList(nn.Linear(c, proj_dim) for c in scale_channels)
        # Timestep -> per-scale gate. Independent sigmoids (not a softmax) so the
        # gates aren't forced to compete: the model can read several scales at full
        # strength at once (coarse context AND fine edges) instead of a convex mix.
        self.scale_gate = nn.Linear(time_dim, self.n_scales)
        # Summarize the concatenated, time-weighted scales (+ time) to out_dim.
        self.mlp = nn.Sequential(
            nn.Linear(self.n_scales * proj_dim + time_dim, out_dim * 2), nn.GELU(),
            nn.Linear(out_dim * 2, out_dim),
        )

    def forward(self, maps, points, t_vec):
        """maps: list of [B,C_s,H,W]; points [B,N,2]; t_vec [B,time_dim] -> [B,N,out_dim]."""
        n = points.shape[1]
        grid = torch.clamp(points, -1.0, 1.0).unsqueeze(1)        # [B,1,N,2]
        w = torch.sigmoid(self.scale_gate(t_vec))                 # [B, S] independent

        feats = []
        for i, (proj, m) in enumerate(zip(self.projs, maps)):
            s = F.grid_sample(m, grid, mode="bilinear",
                              padding_mode="border", align_corners=True)
            s = s.squeeze(2).transpose(1, 2)                      # [B,N,C_s]
            s = proj(s)                                           # [B,N,proj_dim]
            s = s * w[:, i][:, None, None]                        # time-gate this scale
            feats.append(s)

        cat = torch.cat(feats, dim=-1)                            # [B,N, S*proj_dim]
        t_exp = t_vec.unsqueeze(1).expand(-1, n, -1)
        return self.mlp(torch.cat([cat, t_exp], dim=-1))         # [B,N,out_dim]


class ContourDenoiser(nn.Module):
    def __init__(
        self,
        n_points: int = 200,
        hidden_dim: int = 128,
        num_layers: int = 4,
        num_heads: int = 4,
        scale_channels=(64,),
        proj_dim: int = 64,
        coord_fourier_bands: int = 6,
    ):
        super().__init__()
        self.n_points = n_points
        self.hidden_dim = hidden_dim

        # Timestep embedding MLP -> [B, hidden].
        self.time_mlp = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim), nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
        )

        # Per-point, time-gated multi-scale guidance.
        self.sampler = MultiScalePointSampler(
            list(scale_channels), proj_dim=proj_dim, out_dim=hidden_dim, time_dim=hidden_dim,
        )

        # NeRF-style Fourier embedding of point coordinates (additive path only).
        self.coord_ff = FourierFeatures(in_dim=2, num_bands=coord_fourier_bands)
        self.coord_mlp = nn.Sequential(
            nn.Linear(self.coord_ff.out_dim, hidden_dim), nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
        )

        self.curve_proj = nn.Sequential(
            nn.Linear(1, hidden_dim // 4),
            nn.GELU(),
            nn.Linear(hidden_dim // 4, hidden_dim)
        )

        # Fixed sinusoidal positional embedding over the ordered point index.
        pe = torch.zeros(1, n_points, hidden_dim)
        position = torch.arange(0, n_points, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, hidden_dim, 2).float() * (-math.log(10000.0) / hidden_dim))
        pe[0, :, 0::2] = torch.sin(position * div_term)
        pe[0, :, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pos_emb", pe)

        # Local smoothing along the closed contour (circular padding).
        self.local_conv1 = nn.Conv1d(hidden_dim, hidden_dim, 3, padding=1, padding_mode="circular")
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim, nhead=num_heads, dim_feedforward=hidden_dim * 2, batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.local_conv2 = nn.Conv1d(hidden_dim, hidden_dim, 3, padding=1, padding_mode="circular")

        # Prediction head (skip-connected with guidance and raw coords).
        self.out_mlp = nn.Sequential(
            nn.Linear(hidden_dim + hidden_dim + 2, hidden_dim // 2), nn.GELU(),
            nn.Linear(hidden_dim // 2, 2),
        )

        self.temp1 = nn.Parameter(torch.ones(hidden_dim, 1)*1e-3)
        self.temp2 = nn.Parameter(torch.ones(hidden_dim, 1)*1e-3)
        
    def forward(self, points, t, cond_maps):
        """points [B,N,2], t [B], cond_maps = raw pyramid (list of [B,C,H,W]) -> x0 [B,N,2]."""
        t_vec = self.time_mlp(timestep_embedding(t, self.hidden_dim))   # [B,H]
        t_emb = t_vec.unsqueeze(1)                                      # [B,1,H]

        guidance = self.sampler(cond_maps, points, t_vec)              # [B,N,H]
        coord_pe = self.coord_ff(points)                               # [B,N,coord_feat]

        # --- NEU: Compute curvature bias from current point coordinates ---
        # Get immediate neighbors in the closed loop
        prev_p = torch.roll(points, shifts=1, dims=1)
        next_p = torch.roll(points, shifts=-1, dims=1)
        
        # Discrete Laplacian magnitude: ||x_{i-1} + x_{i+1} - 2x_i||
        current_curve = torch.norm(next_p + prev_p - 2 * points, dim=-1, keepdim=True) # [B, N, 1]
        
        # Project to hidden dimension
        curve_bias = self.curve_proj(current_curve)                    # [B, N, H]

        # --- REVISED: Inject the curvature bias additively into the point features ---
        x = self.coord_mlp(coord_pe) + guidance + self.pos_emb + t_emb + curve_bias

        # The rest of your architecture remains completely global and untouched
        x = x.transpose(1, 2)
        x = x + self.temp1 * F.gelu(self.local_conv1(x))
        x = x.transpose(1, 2)

        x = self.transformer(x)

        x = x.transpose(1, 2)
        x = x + self.temp2 * F.gelu(self.local_conv2(x))
        x = x.transpose(1, 2)

        out = self.out_mlp(torch.cat([x, guidance, points], dim=-1))
        out = torch.clamp(out, -3.0, 3.0)
        return out