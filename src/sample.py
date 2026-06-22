"""Evaluate / sample from a trained P2SDiff model.

As a library: `evaluate(...)` runs DDIM sampling over a loader, rasterizes the
predicted points and returns mean Dice / IoU (used by training for validation).

As a CLI:
    python -m src.sample --ckpt src/runs/baseline/best.pth --split test --tta
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
from .utils.helper_funcs import enhance_frequencies
from .utils.post_processing import remove_contour_outliers, smooth_closed_contour, taubin_smooth_closed_contour


def align_contour_to_reference(ref: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """
    Rotiert 'target' zyklisch so, dass die Summe der quadratischen Abstände
    zu 'ref' minimiert wird. Arbeitet auf (N, 2) Tensoren.
    """
    N = ref.shape[0]
    best_shift, best_cost = 0, float("inf")
    for shift in range(N):
        rolled = torch.roll(target, shift, dims=0)
        cost = ((ref - rolled) ** 2).sum().item()
        if cost < best_cost:
            best_cost = cost
            best_shift = shift
    return torch.roll(target, best_shift, dims=0)

@torch.no_grad()
def evaluate(encoder, denoiser, diffusion, loader, cfg, device, viz_path=None):
    """Sample boundaries, rasterize, and return (mean Dice, mean IoU)."""
    encoder.eval(); denoiser.eval()
    dices, ious = [], []
    viz_cache = None

    for images, gt_points, gt_masks in loader:
        images = images.to(device)

        images = enhance_frequencies(images, mid_gain=getattr(cfg, "mid_frequency_gain", 1.5), edge_gain=getattr(cfg, "edge_frequency_gain", 1.2))

        shape = (images.shape[0], cfg.n_points, 2)

        # -----------------------------------------------------------------
        # PASS 1: Original Image Run
        # -----------------------------------------------------------------
        raw_orig = encoder.extract(images)
        cond_fn_orig = lambda t_b: encoder.fuse(raw_orig, t_b)
        pred_points = diffusion.ddim_sample(
            denoiser, cond_fn_orig, shape,
            ddim_steps=cfg.ddim_steps, guidance_scale=cfg.guidance_scale, clamp=1.0,
        )

        # -----------------------------------------------------------------
        # OPTIONAL: Test-Time Augmentation (TTA) → Mask-Level Majority Vote
        # -----------------------------------------------------------------
        if getattr(cfg, "tta", False):
            COORD_SCALE = -1.0
            COORD_OFFSET = 0.0

            # --- TTA 1: Horizontal Flip ---
            images_hf = torch.flip(images, dims=[3])
            raw_hf = encoder.extract(images_hf)
            cond_fn_hf = lambda t_b: encoder.fuse(raw_hf, t_b)
            pred_points_hf = diffusion.ddim_sample(
                denoiser, cond_fn_hf, shape,
                ddim_steps=cfg.ddim_steps, guidance_scale=cfg.guidance_scale, clamp=1.0,
            )
            pred_points_hf_rect = pred_points_hf.clone()
            pred_points_hf_rect[:, :, 0] = COORD_SCALE * pred_points_hf_rect[:, :, 0] + COORD_OFFSET

            # --- TTA 2: Vertical Flip ---
            images_vf = torch.flip(images, dims=[2])
            raw_vf = encoder.extract(images_vf)
            cond_fn_vf = lambda t_b: encoder.fuse(raw_vf, t_b)
            pred_points_vf = diffusion.ddim_sample(
                denoiser, cond_fn_vf, shape,
                ddim_steps=cfg.ddim_steps, guidance_scale=cfg.guidance_scale, clamp=1.0,
            )
            pred_points_vf_rect = pred_points_vf.clone()
            pred_points_vf_rect[:, :, 1] = COORD_SCALE * pred_points_vf_rect[:, :, 1] + COORD_OFFSET

            # Post-Processing auf jede Variante einzeln
            pp_orig = taubin_smooth_closed_contour(
                remove_contour_outliers(pred_points, threshold_sigma=2.0),
                iterations=5, lamb=0.5, mu=-0.53)
            pp_hf = taubin_smooth_closed_contour(
                remove_contour_outliers(pred_points_hf_rect, threshold_sigma=2.0),
                iterations=5, lamb=0.5, mu=-0.53)
            pp_vf = taubin_smooth_closed_contour(
                remove_contour_outliers(pred_points_vf_rect, threshold_sigma=2.0),
                iterations=5, lamb=0.5, mu=-0.53)

            # Mask-Level Majority Vote
            pred_masks_np, batch_scores = [], []
            for i in range(images.shape[0]):
                mask_orig = points_to_mask(pp_orig[i], cfg.img_size)
                mask_hf   = points_to_mask(pp_hf[i],   cfg.img_size)
                mask_vf   = points_to_mask(pp_vf[i],   cfg.img_size)

                vote = mask_orig.astype(np.int32) + mask_hf.astype(np.int32) + mask_vf.astype(np.int32)
                pred_mask = (vote >= 2).astype(np.uint8)

                gt_mask = gt_masks[i].squeeze().cpu().numpy().astype(np.uint8)
                d = dice_score(pred_mask, gt_mask)
                j = iou_score(pred_mask, gt_mask)
                dices.append(d); ious.append(j)
                pred_masks_np.append(pred_mask)
                batch_scores.append({"dice": d, "iou": j})

            if viz_path is not None and viz_cache is None:
                viz_cache = (images.cpu(), gt_points, pp_orig.cpu(),
                             gt_masks, pred_masks_np, batch_scores)

        # -----------------------------------------------------------------
        # Kein TTA: normaler Pfad
        # -----------------------------------------------------------------
        else:
            pred_points = remove_contour_outliers(pred_points, threshold_sigma=2.0)
            pred_points = taubin_smooth_closed_contour(pred_points, iterations=5, lamb=0.5, mu=-0.53)

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

    # -----------------------------------------------------------------
    # Visualization (außerhalb der Batch-Schleife)
    # -----------------------------------------------------------------
    if viz_path is not None and viz_cache is not None:
        imgs, gtp, pp, gtm, pm, sc = viz_cache
        save_prediction_grid(imgs, gtp, pp, gtm, pm, viz_path,
                             max_samples=min(4, imgs.shape[0]), scores=sc)

    return float(np.mean(dices)), float(np.mean(ious))

    
def load_checkpoint(ckpt_path, cfg, device):
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    saved = ckpt.get("config")
    if saved:
        for k in ("encoder", "backbone", "cond_channels", "stem_dim", "n_points", "hidden_dim",
                  "n_transformer_layers", "n_heads", "pretrained", "freeze_backbone",
                  "in_channels", "unet_start_dim", "unet_dim_mults", "unet_groupnorm_groups",
                  "coord_fourier_bands", "pos_grid_bands", "img_size", "npy_size",
                  "pvt_variant", "pvt_pretrained_path", "adaptive_uniformity"): # <--- HIER HINZUFÜGEN
            if k in saved:
                setattr(cfg, k, saved[k])
        if "stem_dim" not in saved:
            cfg.stem_dim = 0

    cfg.pretrained = False
    encoder = build_conditioner(cfg).to(device)
    denoiser = ContourDenoiser(
        n_points=cfg.n_points, hidden_dim=cfg.hidden_dim,
        num_layers=cfg.n_transformer_layers, num_heads=cfg.n_heads,
        scale_channels=encoder.feature_channels, proj_dim=cfg.cond_channels,
        coord_fourier_bands=cfg.coord_fourier_bands,
    ).to(device)
    
    encoder.load_state_dict(ckpt["encoder"], strict=False)
    denoiser.load_state_dict(ckpt["denoiser"], strict=False)
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
    # --- NEW: Added TTA flag ---
    parser.add_argument("--tta", action="store_true", help="Enable Test-Time Augmentation (H-Flip & V-Flip)")
    args = parser.parse_args()

    cfg = Config.from_args(args)
    device = torch.device(cfg.device if torch.cuda.is_available() else "cpu")

    split = "vl" if args.split == "val" else "te"
    ds = build_contour_dataset(cfg.skin_root, cfg.dataset, split, cfg.n_points,
            cfg.img_size, augment=False, npy_size=cfg.npy_size,
            adaptive_sampling=getattr(cfg, "adaptive_uniformity", True)) 
    loader = DataLoader(ds, batch_size=cfg.batch_size, shuffle=False)

    encoder, denoiser = load_checkpoint(args.ckpt, cfg, device)
    diffusion = GaussianDiffusion(cfg.timesteps, cfg.beta_start, cfg.beta_end, device=device)

    dice, iou = evaluate(encoder, denoiser, diffusion, loader, cfg, device, viz_path=args.viz)
    print(f"[{args.split}] Dice {dice:.4f} | IoU {iou:.4f} | viz -> {args.viz}")


if __name__ == "__main__":
    main()