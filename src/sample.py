"""Evaluate / sample from a trained P2SDiff model.

As a library: `evaluate(...)` runs DDIM sampling over a loader, rasterizes the
predicted points and returns mean Dice / IoU (used by training for validation).

As a CLI:
    python -m src.sample --ckpt src/runs/baseline/best.pth --split test
"""

import argparse

import numpy as np
import torch
from torch.utils.data import DataLoader

from .config import Config
from .data import build_contour_dataset, split_counts
from .diffusion import GaussianDiffusion
from .models import ContourDenoiser, build_conditioner
from .utils import dice_score, iou_score, points_to_mask, save_prediction_grid


@torch.no_grad()
def evaluate(encoder, denoiser, diffusion, loader, cfg, device, viz_path=None):
    """Sample boundaries, rasterize, and return (mean Dice, mean IoU)."""
    encoder.eval(); denoiser.eval()
    dices, ious = [], []
    viz_cache = None

    for images, gt_points, gt_masks in loader:
        images = images.to(device)
        raw = encoder.extract(images)                  # backbone runs once
        cond_fn = lambda t_b: encoder.fuse(raw, t_b)   # time-conditioned per step
        shape = (images.shape[0], cfg.n_points, 2)
        pred_points = diffusion.ddim_sample(
            denoiser, cond_fn, shape,
            ddim_steps=cfg.ddim_steps, guidance_scale=cfg.guidance_scale, clamp=1.0,
        )

        pred_masks_np, batch_scores = [], []
        for i in range(images.shape[0]):
            pred_mask = points_to_mask(pred_points[i], cfg.img_size)
            gt_mask = gt_masks[i].squeeze().cpu().numpy().astype(np.uint8)
            d = dice_score(pred_mask, gt_mask)
            j = iou_score(pred_mask, gt_mask)
            dices.append(d); ious.append(j)
            pred_masks_np.append(pred_mask)
            batch_scores.append({"dice": d, "iou": j})

        if viz_path is not None and viz_cache is None:
            viz_cache = (images.cpu(), gt_points, pred_points.cpu(),
                         gt_masks, pred_masks_np, batch_scores)

    if viz_path is not None and viz_cache is not None:
        imgs, gtp, pp, gtm, pm, sc = viz_cache
        save_prediction_grid(imgs, gtp, pp, gtm, pm, viz_path,
                             max_samples=min(4, imgs.shape[0]), scores=sc)

    return float(np.mean(dices)), float(np.mean(ious))


def load_checkpoint(ckpt_path, cfg, device):
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    # Prefer the config stored in the checkpoint so the architecture matches.
    saved = ckpt.get("config")
    if saved:
        for k in ("encoder", "backbone", "cond_channels", "stem_dim", "n_points", "hidden_dim",
                  "n_transformer_layers", "n_heads", "pretrained", "freeze_backbone",
                  "in_channels", "unet_start_dim", "unet_dim_mults", "unet_groupnorm_groups",
                  "coord_fourier_bands", "pos_grid_bands", "img_size", "npy_size",
                  "pvt_variant", "pvt_pretrained_path"):
            if k in saved:
                setattr(cfg, k, saved[k])
        # Checkpoints saved before the high-res stem existed have no stem weights.
        if "stem_dim" not in saved:
            cfg.stem_dim = 0

    # Don't re-download ImageNet weights at load time; the checkpoint already has them.
    cfg.pretrained = False
    encoder = build_conditioner(cfg).to(device)
    denoiser = ContourDenoiser(
        n_points=cfg.n_points, hidden_dim=cfg.hidden_dim,
        num_layers=cfg.n_transformer_layers, num_heads=cfg.n_heads,
        scale_channels=encoder.feature_channels, proj_dim=cfg.cond_channels,
        coord_fourier_bands=cfg.coord_fourier_bands,
    ).to(device)
    encoder.load_state_dict(ckpt["encoder"])
    denoiser.load_state_dict(ckpt["denoiser"])
    return encoder, denoiser


def main():
    parser = argparse.ArgumentParser(description="Evaluate / sample P2SDiff")
    parser.add_argument("--ckpt", type=str, required=True)
    parser.add_argument("--dataset", choices=["ph2", "isic2017", "isic2018", "ham10000"])
    parser.add_argument("--skin_root", type=str)
    parser.add_argument("--encoder", choices=["convnext", "pvt", "unet"])
    parser.add_argument("--split", choices=["val", "test"], default="test")
    parser.add_argument("--device", type=str)
    parser.add_argument("--guidance_scale", type=float)
    parser.add_argument("--ddim_steps", type=int)
    parser.add_argument("--viz", type=str, default="prediction_grid.png")
    args = parser.parse_args()

    cfg = Config.from_args(args)
    device = torch.device(cfg.device if torch.cuda.is_available() else "cpu")

    split = "vl" if args.split == "val" else "te"
    ds = build_contour_dataset(cfg.skin_root, cfg.dataset, split, cfg.n_points,
                               cfg.img_size, augment=False, npy_size=cfg.npy_size)
    loader = DataLoader(ds, batch_size=cfg.batch_size, shuffle=False)

    encoder, denoiser = load_checkpoint(args.ckpt, cfg, device)
    diffusion = GaussianDiffusion(cfg.timesteps, cfg.beta_start, cfg.beta_end, device=device)

    dice, iou = evaluate(encoder, denoiser, diffusion, loader, cfg, device, viz_path=args.viz)
    print(f"[{args.split}] Dice {dice:.4f} | IoU {iou:.4f} | viz -> {args.viz}")


if __name__ == "__main__":
    main()
