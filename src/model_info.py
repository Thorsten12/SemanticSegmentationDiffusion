"""Print model size for a chosen encoder without loading a dataset."""

import argparse
import torch

from .config import Config
from .train import build_models, str2bool


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--encoder', choices=['unet','convnext','pvt','timm'], default='unet')
    p.add_argument('--backbone', default='convnext_tiny')
    p.add_argument('--pretrained', type=str2bool, default=False)
    p.add_argument('--unet_start_dim', type=int, default=64)
    p.add_argument('--stem_dim', type=int, default=32)
    p.add_argument('--proposal_type', choices=['ellipse', 'fourier'], default='ellipse')
    p.add_argument('--diffusion_target', choices=['absolute', 'residual'], default='absolute')
    p.add_argument('--fourier_harmonics', type=int, default=4)
    a = p.parse_args()
    cfg = Config(
        encoder=a.encoder, backbone=a.backbone, pretrained=a.pretrained,
        unet_start_dim=a.unet_start_dim, stem_dim=a.stem_dim,
        proposal_type=a.proposal_type, diffusion_target=a.diffusion_target,
        fourier_harmonics=a.fourier_harmonics,
    )
    enc, den = build_models(cfg, torch.device('cpu'))
    n_enc = sum(x.numel() for x in enc.parameters())
    n_den = sum(x.numel() for x in den.parameters())
    print(f'encoder: {n_enc/1e6:.3f} M')
    print(f'sparse diffusion decoder: {n_den/1e6:.3f} M')
    print(f'total: {(n_enc+n_den)/1e6:.3f} M')


if __name__ == '__main__':
    main()
