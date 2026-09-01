"""Evaluate / sample from a trained P2SDiff model.

As a library: `evaluate(...)` runs DDIM sampling over a loader, rasterizes the
predicted points and returns mean Dice / IoU (used by training for validation).

As a CLI:
    python -m src.sample --ckpt src/runs/baseline/best.pth --split test
"""

import argparse
import csv

import numpy as np
import torch
from torch.utils.data import DataLoader

from .config import Config
from .data import DATASET_NAMES, build_contour_dataset
from .diffusion import GaussianDiffusion
from .models import ContourDenoiser, build_conditioner
from .utils import dice_score, iou_score, points_to_mask, save_prediction_grid


@torch.no_grad()
def evaluate(encoder, denoiser, diffusion, loader, cfg, device, viz_path=None,
             metrics_path=None):
    """Sample boundaries, rasterize, and return (mean Dice, mean IoU)."""
    encoder.eval(); denoiser.eval()
    dices, ious = [], []
    viz_cache = None
    case_metrics = []
    eval_seed = int(getattr(cfg, "eval_seed", -1))
    generator = None
    if eval_seed >= 0:
        generator = torch.Generator(device=device)
        generator.manual_seed(eval_seed)

    for images, gt_points, gt_masks in loader:
        images = images.to(device)
        maps = encoder.extract(images)                     # backbone once
        condition = denoiser.prepare_condition(maps, image=images)  # sparse condition cache
        proposal = denoiser.proposal_points(condition)
        cond_fn = lambda t_b, condition=condition: condition
        shape = (images.shape[0], cfg.n_points, 2)
        coarse_points = diffusion.ddim_sample(
            denoiser, cond_fn, shape,
            ddim_steps=cfg.ddim_steps, guidance_scale=cfg.guidance_scale,
            latent_clamp=cfg.latent_clamp,
            proposal_points=(proposal if cfg.diffusion_target == "residual" else None),
            residual_scale=getattr(cfg, "residual_scale", 1.0),
            generator=generator,
        )
        pred_points = coarse_points  # V5.2: no post-snapper

        pred_masks_np, batch_scores = [], []
        for i in range(images.shape[0]):
            pred_mask = points_to_mask(pred_points[i], cfg.img_size)
            coarse_mask = points_to_mask(coarse_points[i], cfg.img_size)
            proposal_mask = points_to_mask(proposal[i], cfg.img_size)
            gt_mask = gt_masks[i].squeeze().cpu().numpy().astype(np.uint8)
            d = dice_score(pred_mask, gt_mask)
            j = iou_score(pred_mask, gt_mask)
            dices.append(d); ious.append(j)
            gt_bool = gt_mask.astype(bool)
            touches = bool(
                gt_bool[0].any() or gt_bool[-1].any()
                or gt_bool[:, 0].any() or gt_bool[:, -1].any()
            )
            case_metrics.append({
                "index": len(case_metrics),
                "gt_area_fraction": float(gt_bool.mean()),
                "gt_touches_border": int(touches),
                "proposal_dice": dice_score(proposal_mask, gt_mask),
                "coarse_dice": dice_score(coarse_mask, gt_mask),
                "dice": d,
                "iou": j,
            })
            pred_masks_np.append(pred_mask)
            batch_scores.append({"dice": d, "iou": j})

        if viz_path is not None and viz_cache is None:
            viz_cache = (images.cpu(), gt_points, pred_points.cpu(),
                         gt_masks, pred_masks_np, batch_scores)

    if viz_path is not None and viz_cache is not None:
        imgs, gtp, pp, gtm, pm, sc = viz_cache
        save_prediction_grid(imgs, gtp, pp, gtm, pm, viz_path,
                             max_samples=min(4, imgs.shape[0]), scores=sc)

    if metrics_path is not None:
        with open(metrics_path, "w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=[
                "index", "gt_area_fraction", "gt_touches_border",
                "proposal_dice", "coarse_dice", "dice", "iou",
            ])
            writer.writeheader()
            writer.writerows(case_metrics)

    return float(np.mean(dices)), float(np.mean(ious))


def load_checkpoint(ckpt_path, cfg, device):
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    # Prefer the config stored in the checkpoint so the architecture matches.
    saved = ckpt.get("config")
    if saved:
        version = saved.get("model_version", "v1")
        if version not in ("v3_1_curriculum_snap", "v5_spatial_residual", "v5_2_exact_boundary"):
            raise ValueError(
                f"{ckpt_path} is a {version!r} checkpoint, but this source evaluates "
                "V3.1/V5/V5.2 checkpoints. To upgrade an older checkpoint, warm-start training "
                "with --init_checkpoint /path/to/best.pth."
            )
        for k in ("encoder", "backbone", "stem_dim", "n_points", "hidden_dim",
                  "n_transformer_layers", "n_heads", "pretrained", "freeze_backbone",
                  "in_channels", "unet_start_dim", "unet_dim_mults", "unet_groupnorm_groups",
                  "tag_maps", "tag_mode",
                  "coord_fourier_bands", "contour_phase_bands",
                  "deformable_samples", "local_radius_min", "local_radius_max",
                  "global_levels", "global_grid", "latent_clamp",
                  "proposal_type", "diffusion_target", "fourier_harmonics",
                  "proposal_arclength_resample", "residual_scale", "eval_seed",
                  "exact_boundary_enabled", "exact_boundary_levels", "exact_boundary_samples",
                  "exact_boundary_radius", "exact_boundary_profile_dim", "exact_boundary_hidden",
                  "exact_boundary_ring_bands", "exact_boundary_relative_bias",
                  "exact_boundary_confidence_power", "exact_boundary_low_t_fraction",
                  "exact_confidence_radius", "exact_tangent_tolerance", "exact_use_rgb",
                  "boundary_dim", "boundary_levels", "boundary_thickness",
                  "snap_candidates", "snap_iterations", "snap_radius_max",
                  "snap_radius_min", "snap_temperature", "snap_hard_inference",
                  "snap_teacher_offset", "snap_teacher_smooth", "snap_profile_radius",
                  "low_t_fraction", "low_t_max_fraction",
                  "lambda_boundary_band", "lambda_boundary_head", "lambda_snap_teacher",
                  "lambda_hd", "hd_fraction", "lambda_curvature", "boundary_band_width",
                  "lambda_hard_boundary", "hard_boundary_fraction",
                  "snap_teacher_hard_weight", "snap_teacher_hard_fraction",
                  "snap_geometry_warmup_epochs", "snap_geometry_ramp_epochs",
                  "timesteps", "beta_start", "beta_end",
                  "img_size", "npy_size",
                  "pvt_variant", "pvt_pretrained_path", "dataset", "skin_root",
                  "data_root", "polyp_test"):
            if k in saved:
                setattr(cfg, k, saved[k])
        if "stem_dim" not in saved:
            cfg.stem_dim = 0
        if saved.get("fusion") == "flat":
            raise ValueError(
                f"{ckpt_path} uses the removed flat sampler. "
                "Current src only has DP2Seg (cascade contour decoder)."
            )

    # Don't re-download ImageNet weights at load time; the checkpoint already has them.
    cfg.pretrained = False
    encoder = build_conditioner(cfg).to(device)
    denoiser = ContourDenoiser(
        n_points=cfg.n_points, hidden_dim=cfg.hidden_dim,
        num_layers=cfg.n_transformer_layers, num_heads=cfg.n_heads,
        scale_channels=encoder.feature_channels,
        coord_fourier_bands=cfg.coord_fourier_bands,
        tag_maps=getattr(cfg, "tag_maps", True),
        tag_mode=getattr(cfg, "tag_mode", "concat"),
        timesteps=getattr(cfg, "timesteps", 1000),
        deformable_samples=getattr(cfg, "deformable_samples", 5),
        local_radius_min=getattr(cfg, "local_radius_min", 0.025),
        local_radius_max=getattr(cfg, "local_radius_max", 0.22),
        global_levels=getattr(cfg, "global_levels", 3),
        global_grid=getattr(cfg, "global_grid", 14),
        contour_phase_bands=getattr(cfg, "contour_phase_bands", 4),
        boundary_dim=getattr(cfg, "boundary_dim", 32),
        boundary_levels=getattr(cfg, "boundary_levels", 3),
        snap_candidates=getattr(cfg, "snap_candidates", 15),
        snap_iterations=getattr(cfg, "snap_iterations", 3),
        snap_radius_max=getattr(cfg, "snap_radius_max", 0.16),
        snap_radius_min=getattr(cfg, "snap_radius_min", 0.015),
        snap_temperature=getattr(cfg, "snap_temperature", 0.16),
        snap_profile_radius=getattr(cfg, "snap_profile_radius", 0.018),
        proposal_type=getattr(cfg, "proposal_type", "ellipse"),
        diffusion_target=getattr(cfg, "diffusion_target", "absolute"),
        fourier_harmonics=getattr(cfg, "fourier_harmonics", 4),
        proposal_arclength_resample=getattr(cfg, "proposal_arclength_resample", False),
        residual_scale=getattr(cfg, "residual_scale", 1.0),
        exact_boundary_enabled=getattr(cfg, "exact_boundary_enabled", True),
        exact_boundary_levels=getattr(cfg, "exact_boundary_levels", 2),
        exact_boundary_samples=getattr(cfg, "exact_boundary_samples", 11),
        exact_boundary_radius=getattr(cfg, "exact_boundary_radius", 0.10),
        exact_boundary_profile_dim=getattr(cfg, "exact_boundary_profile_dim", 20),
        exact_boundary_hidden=getattr(cfg, "exact_boundary_hidden", 64),
        exact_boundary_ring_bands=getattr(cfg, "exact_boundary_ring_bands", 4),
        exact_boundary_relative_bias=getattr(cfg, "exact_boundary_relative_bias", 0.12),
        exact_boundary_confidence_power=getattr(cfg, "exact_boundary_confidence_power", 2.0),
        exact_boundary_low_t_fraction=getattr(cfg, "exact_boundary_low_t_fraction", 0.30),
        exact_confidence_radius=getattr(cfg, "exact_confidence_radius", 0.060),
        exact_tangent_tolerance=getattr(cfg, "exact_tangent_tolerance", 0.040),
        exact_use_rgb=getattr(cfg, "exact_use_rgb", True),
    ).to(device)
    encoder.load_state_dict(ckpt["encoder"], strict=False)
    missing, unexpected = denoiser.load_state_dict(ckpt["denoiser"], strict=False)
    if missing or unexpected:
        print(f"  denoiser: missing={len(missing)} unexpected={len(unexpected)}")
    return encoder, denoiser


def main():
    parser = argparse.ArgumentParser(description="Evaluate / sample P2SDiff")
    parser.add_argument("--ckpt", type=str, required=True)
    parser.add_argument("--dataset", choices=list(DATASET_NAMES))
    parser.add_argument("--skin_root", type=str)
    parser.add_argument("--data_root", type=str)
    parser.add_argument("--polyp_test", type=str)
    parser.add_argument("--encoder", choices=["convnext", "pvt", "unet", "timm"])
    parser.add_argument("--split", choices=["val", "test"], default="test")
    parser.add_argument("--device", type=str)
    parser.add_argument("--guidance_scale", type=float)
    parser.add_argument("--ddim_steps", type=int)
    parser.add_argument("--eval_seed", type=int)
    parser.add_argument("--viz", type=str, default="prediction_grid.png")
    parser.add_argument("--metrics", type=str, default=None)
    args = parser.parse_args()

    cfg = Config.from_args(args)
    device = torch.device(cfg.device if torch.cuda.is_available() else "cpu")

    split = "vl" if args.split == "val" else "te"
    ds = build_contour_dataset(
        cfg.skin_root, cfg.dataset, split, cfg.n_points,
        cfg.img_size, augment=False, npy_size=cfg.npy_size,
        data_root=cfg.data_root, polyp_test=cfg.polyp_test,
    )
    loader = DataLoader(ds, batch_size=cfg.batch_size, shuffle=False)

    encoder, denoiser = load_checkpoint(args.ckpt, cfg, device)
    diffusion = GaussianDiffusion(cfg.timesteps, cfg.beta_start, cfg.beta_end, device=device)

    dice, iou = evaluate(
        encoder, denoiser, diffusion, loader, cfg, device,
        viz_path=args.viz, metrics_path=args.metrics,
    )
    print(f"[{args.split}] Dice {dice:.4f} | IoU {iou:.4f} | viz -> {args.viz}")


if __name__ == "__main__":
    main()
