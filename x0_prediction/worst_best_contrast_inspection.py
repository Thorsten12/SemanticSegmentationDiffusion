"""Visual inspection of worst/best-decile samples ranked by image_contrast.

Follow-up to failure_mode_correlation.py, which found image_contrast as the
strongest (but only moderate, rho ~ 0.3) correlate of per-sample Dice.

This script does NOT just report numbers -- it saves a grid of actual images
(input, GT contour, predicted contour, per-sample dice + contrast value) for
the worst-decile and best-decile samples so you can eyeball whether the
correlation:
  (a) holds fairly uniformly across the worst-decile images, or
  (b) is actually driven by a small subgroup (hair artifacts, vignetting,
      ink markings, specific lesion types, ruler marks, etc.) that happens
      to also have low local contrast, in which case "low contrast" would be
      a symptom/proxy of those artifacts rather than the causal factor itself.

Usage:
    python -m x0_prediction_V6.worst_best_contrast_inspection \
        --ckpt x0_prediction_V6/runs/ablation_v7_laziness_on_snapboth030/baseline_seed_01/best.pth \
        --dataset isic2018 --split test \
        --out_dir x0_prediction_V6/analysis_out/contrast_inspection
"""

import argparse
import os

import numpy as np
import torch
import cv2
from torch.utils.data import DataLoader
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from .config import Config
from .data import build_contour_dataset
from .sample import load_checkpoint
from .diffusion import GaussianDiffusion
from .utils import dice_score, iou_score, points_to_mask
from .failure_mode_correlation import per_sample_mask_stats, per_sample_image_stats


def denorm_to_uint8(image_chw: np.ndarray) -> np.ndarray:
    """image_chw assumed roughly in [-1, 1]. Returns [H,W,3] uint8 for display."""
    img = np.transpose(image_chw, (1, 2, 0))
    img = (img + 1.0) / 2.0
    img = np.clip(img, 0.0, 1.0)
    if img.shape[-1] == 1:
        img = np.repeat(img, 3, axis=-1)
    return (img * 255).astype(np.uint8)


def mask_to_contour_points(mask: np.ndarray):
    contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    if not contours:
        return None
    return max(contours, key=cv2.contourArea)


def make_panel(image_u8, gt_mask, pred_mask, dice, contrast, area_frac, solidity):
    """One sample's inspection panel: raw image | GT overlay | pred overlay."""
    h, w = gt_mask.shape
    panel = np.zeros((h, w * 3 + 20, 3), dtype=np.uint8)
    panel[:, :w] = image_u8

    gt_overlay = image_u8.copy()
    gt_c = mask_to_contour_points(gt_mask)
    if gt_c is not None:
        cv2.drawContours(gt_overlay, [gt_c], -1, (0, 255, 0), 2)
    panel[:, w + 10:w * 2 + 10] = gt_overlay

    pred_overlay = image_u8.copy()
    pred_c = mask_to_contour_points(pred_mask)
    if pred_c is not None:
        cv2.drawContours(pred_overlay, [pred_c], -1, (255, 60, 60), 2)
    if gt_c is not None:
        cv2.drawContours(pred_overlay, [gt_c], -1, (0, 255, 0), 1)
    panel[:, w * 2 + 20:] = pred_overlay

    return panel


@torch.no_grad()
def collect_records_with_media(ckpt_path, dataset, split, device, max_samples=None):
    cfg = Config()
    device = torch.device(device)

    encoder, denoiser, snapper, proposal_head = load_checkpoint(ckpt_path, cfg, device)
    diffusion = GaussianDiffusion(cfg.timesteps, cfg.beta_start, cfg.beta_end, device=device)
    encoder.eval(); denoiser.eval(); snapper.eval()
    if proposal_head is not None:
        proposal_head.eval()

    split_key = "te" if split == "test" else "vl"
    ds = build_contour_dataset(cfg.skin_root, dataset, split_key, cfg.n_points,
                               cfg.img_size, augment=False, npy_size=cfg.npy_size)
    loader = DataLoader(ds, batch_size=8, shuffle=False)

    records = []
    snap_t_threshold = int(round(getattr(cfg, "snap_t_threshold_frac", 0.15) * cfg.timesteps))
    n_seen = 0

    for images, gt_points, gt_masks in loader:
        if max_samples is not None and n_seen >= max_samples:
            break
        images = images.to(device)
        raw = encoder.extract(images)
        proposal = proposal_head(raw) if proposal_head is not None else None
        cond_fn = lambda t_b: encoder.fuse(raw, t_b)
        shape = (images.shape[0], cfg.n_points, 2)

        pred_points = diffusion.ddim_sample(
            denoiser, cond_fn, shape,
            proposal=proposal, proposal_target=cfg.proposal_target,
            ddim_steps=cfg.ddim_steps, guidance_scale=cfg.guidance_scale, clamp=1.0,
            snapper=snapper if cfg.snap_mode in ("loop", "both") else None,
            snap_maps=raw if cfg.snap_mode in ("loop", "both") else None,
            snap_image=images if cfg.snap_mode in ("loop", "both") else None,
            snap_t_threshold=snap_t_threshold if cfg.snap_mode in ("loop", "both") else None,
            snap_every=getattr(cfg, "snap_every", 1),
            collect_every=0,
        )
        if cfg.snap_mode in ("post", "both"):
            pred_points = snapper(pred_points, raw, image=images, hard=True)

        images_cpu = images.cpu().numpy()
        for i in range(images.shape[0]):
            gt_mask = gt_masks[i].squeeze().cpu().numpy().astype(np.uint8)
            pred_mask = points_to_mask(pred_points[i], cfg.img_size)
            d = dice_score(pred_mask, gt_mask)
            j = iou_score(pred_mask, gt_mask)

            mask_stats = per_sample_mask_stats(gt_mask)
            img_stats = per_sample_image_stats(images_cpu[i], gt_mask)

            records.append({
                "dice": d, "iou": j, **mask_stats, **img_stats,
                "image_u8": denorm_to_uint8(images_cpu[i]),
                "gt_mask": gt_mask,
                "pred_mask": pred_mask,
            })
            n_seen += 1
            if max_samples is not None and n_seen >= max_samples:
                break

    return records


def save_grid(records, indices, out_path, title, cols=4):
    n = len(indices)
    rows = int(np.ceil(n / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 5, rows * 2.2))
    axes = np.array(axes).reshape(-1)

    for ax_idx, rec_idx in enumerate(indices):
        r = records[rec_idx]
        panel = make_panel(r["image_u8"], r["gt_mask"], r["pred_mask"],
                            r["dice"], r["image_contrast"], r["lesion_area_frac"],
                            r["mask_solidity"])
        ax = axes[ax_idx]
        ax.imshow(panel)
        ax.set_title(
            f"dice={r['dice']:.2f} contrast={r['image_contrast']:.3f}\n"
            f"area={r['lesion_area_frac']:.3f} solidity={r['mask_solidity']:.2f} "
            f"compact={r['mask_compactness']:.2f}",
            fontsize=8,
        )
        ax.axis("off")

    for ax_idx in range(n, len(axes)):
        axes[ax_idx].axis("off")

    fig.suptitle(title, fontsize=12)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(out_path, dpi=140)
    plt.close(fig)


def report_and_save(records, out_dir):
    os.makedirs(out_dir, exist_ok=True)
    dice = np.array([r["dice"] for r in records])
    contrast = np.array([r["image_contrast"] for r in records])
    n = len(records)
    worst_n = max(1, n // 10)

    # Rank by contrast (ascending = lowest contrast first), NOT by dice.
    # This directly visualizes the factor under test rather than re-deriving
    # a dice-sorted view, so you can see what "low contrast" images actually
    # look like and whether their dice is uniformly bad or scattered.
    contrast_order = np.argsort(contrast)
    low_contrast_idx = contrast_order[:worst_n]
    high_contrast_idx = contrast_order[-worst_n:]

    # Also keep the dice-sorted worst/best decile for direct comparison.
    dice_order = np.argsort(dice)
    worst_dice_idx = dice_order[:worst_n]
    best_dice_idx = dice_order[-worst_n:]

    overlap_low_contrast_worst_dice = len(set(low_contrast_idx) & set(worst_dice_idx))

    print(f"N = {n}, decile size = {worst_n}")
    print(f"Overlap between 'lowest-contrast decile' and 'worst-dice decile': "
          f"{overlap_low_contrast_worst_dice}/{worst_n} samples "
          f"({100*overlap_low_contrast_worst_dice/worst_n:.0f}%)")
    print("(High overlap => contrast plausibly drives failure directly. "
          "Low overlap => the correlation is weak/driven by a few points; "
          "many low-contrast images are actually fine, and/or many failures "
          "are NOT low-contrast -- look at both grids below.)\n")

    save_grid(records, low_contrast_idx, os.path.join(out_dir, "lowest_contrast_decile.png"),
               "Lowest-contrast decile (ranked by image_contrast, ascending)")
    save_grid(records, high_contrast_idx, os.path.join(out_dir, "highest_contrast_decile.png"),
               "Highest-contrast decile (ranked by image_contrast, descending)")
    save_grid(records, worst_dice_idx, os.path.join(out_dir, "worst_dice_decile.png"),
               "Worst-dice decile (for comparison)")
    save_grid(records, best_dice_idx, os.path.join(out_dir, "best_dice_decile.png"),
               "Best-dice decile (for comparison)")

    print(f"Saved 4 grids to {out_dir}:")
    print("  lowest_contrast_decile.png  <- eyeball these for hair/ink/vignetting/artifact subgroups")
    print("  highest_contrast_decile.png")
    print("  worst_dice_decile.png       <- eyeball these for what's actually failing")
    print("  best_dice_decile.png")
    print(
        "\nWhat to look for:\n"
        "  - If lowest_contrast_decile images share a visible confound (dense hair,\n"
        "    dark corner vignetting, surgical ink, ruler marks) -> image_contrast may be\n"
        "    a proxy for that artifact rather than the causal factor. Consider adding an\n"
        "    explicit 'has_hair'/'has_ink' flag if you can label a subsample, and re-check\n"
        "    whether contrast still correlates after controlling for it (or just look at\n"
        "    whether the flagged subgroup alone accounts for the correlation).\n"
        "  - If worst_dice_decile images don't visually overlap much with\n"
        "    lowest_contrast_decile images, that's consistent with the modest rho=0.3:\n"
        "    contrast explains part of the failures but plenty of failures have other\n"
        "    causes (check those images for GT mask errors, multi-lesion images, etc.)."
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ckpt", type=str, required=True)
    parser.add_argument("--dataset", type=str, default="isic2018")
    parser.add_argument("--split", type=str, default="test")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--max_samples", type=int, default=None)
    parser.add_argument("--out_dir", type=str, default="analysis_out/contrast_inspection")
    args = parser.parse_args()

    records = collect_records_with_media(args.ckpt, args.dataset, args.split, args.device,
                                          max_samples=args.max_samples)
    report_and_save(records, args.out_dir)