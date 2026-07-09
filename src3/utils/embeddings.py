import torch
import math

def timestep_encoding(t: torch.Tensor, dim: int, max_period: int = 10000):
    half = dim // 2
    freqs = torch.exp(
        -math.log(max_period) * torch.arange(start=0, end=half, device=t.device, dtype=torch.float32) / half
    )

    args = t.float()[:, None] * freqs[None] # jeder Punkt mal alle Frequenzen
    emb = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)

    if dim % 2:
        raise ValueError("dim must be even")
    return emb

def order_encoding(o: torch.Tensor, dim: int, max_period: int = 10000):
    if dim % 2:
        raise ValueError("dim must be even")
    half = dim // 2
    freqs = torch.exp(
        -math.log(max_period) * torch.arange(start=0, end=half, device=o.device, dtype=torch.float32) / half
    )
    args = o.float()[:, None] * freqs[None]  # (N, half)
    emb = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)  # (N, dim)
    return emb

def positional_encoding(x: torch.Tensor, in_dim = 2, num_bands: int = 12, max_freq: int = 32, include_input: bool = True):
    exps    = torch.linspace(start = 0.0, end = math.log2(max_freq), steps = num_bands, device=x.device)
    freqs   = 2.0 ** exps * math.pi

    x       = x[..., None]                  # B, N, 2, 1
    freqs   = freqs[None, None, None, :]    # 1, 1, 1, num_bands

    args    = x * freqs             # N, 2, num_bands
    emb = torch.cat([torch.sin(args), torch.cos(args)], dim=-1)

    emb = emb.flatten(2)            # N, 2 * num_bands

    return emb