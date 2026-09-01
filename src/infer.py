"""Run a trained P2SDiff V5.2 checkpoint on one RGB image.

Example:
    python -m src.infer --ckpt runs/best.pth --image lesion.jpg --out_mask mask.png
"""

import argparse

import numpy as np
import torch
import torchvision.transforms as transforms
from PIL import Image

from .config import Config
from .diffusion import GaussianDiffusion
from .sample import load_checkpoint
from .utils import points_to_mask


@torch.no_grad()
def main():
    parser = argparse.ArgumentParser(description="Single-image inference with P2SDiff V5.2")
    parser.add_argument("--ckpt", required=True, type=str)
    parser.add_argument("--image", required=True, type=str)
    parser.add_argument("--out_mask", required=True, type=str)
    parser.add_argument("--device", type=str)
    parser.add_argument("--ddim_steps", type=int)
    parser.add_argument("--guidance_scale", type=float)
    args = parser.parse_args()

    cfg = Config.from_args(args)
    device = torch.device(cfg.device if torch.cuda.is_available() else "cpu")
    encoder, denoiser = load_checkpoint(args.ckpt, cfg, device)

    image = Image.open(args.image).convert("RGB")
    orig_w, orig_h = image.size
    tfm = transforms.Compose([
        transforms.Resize(cfg.img_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
    ])
    x = tfm(image).unsqueeze(0).to(device)

    encoder.eval(); denoiser.eval()
    maps = encoder.extract(x)
    condition = denoiser.prepare_condition(maps, image=x)
    proposal = denoiser.proposal_points(condition)
    diffusion = GaussianDiffusion(
        cfg.timesteps, cfg.beta_start, cfg.beta_end, device=device,
    )
    coarse_points = diffusion.ddim_sample(
        denoiser, lambda _t: condition, (1, cfg.n_points, 2),
        ddim_steps=cfg.ddim_steps,
        guidance_scale=cfg.guidance_scale,
        latent_clamp=cfg.latent_clamp,
        proposal_points=(proposal if cfg.diffusion_target == "residual" else None),
        residual_scale=getattr(cfg, "residual_scale", 1.0),
    )
    pred_points = coarse_points  # V5.2: no post-snapper

    mask = points_to_mask(pred_points[0], (orig_h, orig_w))
    Image.fromarray((mask * 255).astype(np.uint8), mode="L").save(args.out_mask)
    print(f"saved mask -> {args.out_mask}")


if __name__ == "__main__":
    main()
