"""Center-bias vs. size-bias diagnosis for outlier samples.

Motivating case (idx 485): the model picked a LARGER, more diffuse patch
over the correct, smaller, GT-labeled one. Two competing explanations:

    (a) SIZE-BIAS:   the model just prefers "biggest blob" regardless of
                      position -- it would pick the same wrong region even
                      if it were off-center.
    (b) CENTER-BIAS:  the model prefers regions close to the image center
                      (dermoscopy images are typically centered on the
                      target lesion) -- it picked the wrong region because
                      it happened to be more central, not just because it
                      was bigger.

These aren't mutually exclusive, but which one dominates changes the fix:
(a) -> hard-negative augmentation / distractor training.
(b) -> only relevant if GT lesions are NOT reliably centered themselves,
       in which case adding an explicit center-prior would actively hurt
       the many off-center legitimate cases. So this script also reports
       whether GT lesions in this dataset even tend to be centered, which
       is a precondition for a center-prior to make sense at all.

For each requested sample:
    - GT centroid distance to image center (normalized to [0,1], where 1 =
      corner-to-center distance)
    - Prediction centroid distance to image center
    - GT area (fraction of image)
    - Prediction area (fraction of image)
    - area_ratio = pred_area / gt_area (>1 = over-segmented / bigger blob)
    - centroid_shift = distance between GT centroid and prediction centroid
      (normalized) -- large shift = model looked at a genuinely different
      region, not just a fuzzy boundary around the same object

Then aggregates across the outlier set to answer directly:
    "Are predictions on outliers systematically MORE CENTERED than their
     GT? Are they systematically LARGER than their GT? Which effect is
     bigger?"

And, as the precondition check:
    "Are GT lesions in general (not just outliers) reliably centered?"
    (mean/median GT centroid distance across ALL samples, not just outliers)

Usage:
    python -m src.analyze_center_size_bias \
        --runs_root x0_prediction_backup/runs/ablation_v15a_local_drop \
        --skin_root data/datasets \
        --dataset isic2018 --split test \
        --indices 213 445 485 417 413 465 482 430 171 433 3 84 477 219 323 394 348 \
        --out center_size_bias.csv
"""

import argparse
import csv

import numpy as np
import torch
from torch.utils.data import DataLoader

from .config import Config
from .data import build_contour_dataset
from .diffusion import GaussianDiffusion
from .sample import load_checkpoint
from .utils import points_to_mask
from .postprocess import combine_masks
from .ensemble_predict import find_checkpoints, _predict_points_single_checkpoint

CROSS_RUN_EXPERIMENTS = ["drop_1", "drop_2", "drop_2_topk_none"]


# ---------------------------------------------------------------------------
# Geometry helpers (pure numpy, operate on a single binary mask)
# ---------------------------------------------------------------------------

def centroid(mask):
    """Pixel-area centroid (row, col) of a binary mask, or None if empty."""
    ys, xs = np.nonzero(mask)
    if len(xs) == 0:
        return None
    return (float(ys.mean()), float(xs.mean()))


def normalized_center_distance(mask):
    """Distance from mask centroid to image center, normalized so that the
    image corner is at distance 1.0. Returns None if mask is empty."""
    h, w = mask.shape
    c = centroid(mask)
    if c is None:
        return None
    cy, cx = c
    center_y, center_x = h / 2, w / 2
    dist = np.hypot(cy - center_y, cx - center_x)
    max_dist = np.hypot(center_y, center_x)  # corner distance
    return float(dist / max_dist) if max_dist > 0 else 0.0


def area_fraction(mask):
    return float(mask.sum() / mask.size)


def normalized_centroid_shift(mask_a, mask_b):
    """Distance between two masks' centroids, normalized by image diagonal
    (so 1.0 = full corner-to-corner shift). Returns None if either is empty."""
    h, w = mask_a.shape
    ca, cb = centroid(mask_a), centroid(mask_b)
    if ca is None or cb is None:
        return None
    dist = np.hypot(ca[0] - cb[0], ca[1] - cb[1])
    diag = np.hypot(h, w)
    return float(dist / diag)


# ---------------------------------------------------------------------------
# Prediction collection (mirrors visualize_outliers.py's approach)
# ---------------------------------------------------------------------------

def collect_masks_for_indices(ckpt_map_cross, cfg, device, loader, wanted_indices):
    diffusion = GaussianDiffusion(cfg.timesteps, cfg.beta_start, cfg.beta_end, device=device)
    cross_ckpts = [p for paths in ckpt_map_cross.values() for p in paths]

    wanted = set(wanted_indices)
    masks_per_ckpt = {p: {} for p in cross_ckpts}
    gt_masks = {}

    for ckpt_path in cross_ckpts:
        print(f"  predicting with {ckpt_path} ...")
        encoder, denoiser, snapper, proposal_head = load_checkpoint(ckpt_path, cfg, device)

        idx = 0
        for images, gt_points, gt_batch in loader:
            b = images.shape[0]
            batch_indices = list(range(idx, idx + b))
            if wanted.isdisjoint(batch_indices):
                idx += b
                continue

            point_sets = _predict_points_single_checkpoint(
                encoder, denoiser, diffusion, images, cfg, device,
                snapper=snapper, proposal_head=proposal_head, tta_variants=("none",),
            )
            for i in range(b):
                gi = idx + i
                if gi not in wanted:
                    continue
                if gi not in gt_masks:
                    gt_masks[gi] = gt_batch[i].squeeze().cpu().numpy().astype(np.uint8)
                masks_per_ckpt[ckpt_path][gi] = points_to_mask(point_sets[0][i], cfg.img_size)
            idx += b

        del encoder, denoiser, snapper, proposal_head
        if device.type == "cuda":
            torch.cuda.empty_cache()

    return masks_per_ckpt, gt_masks, cross_ckpts


def collect_gt_masks_all(loader):
    """Cheap pass (no model inference) to get GT masks for EVERY sample, used
    for the 'are GT lesions centered in general' precondition check."""
    gt_masks = []
    for _, _, gt_batch in loader:
        for i in range(gt_batch.shape[0]):
            gt_masks.append(gt_batch[i].squeeze().cpu().numpy().astype(np.uint8))
    return gt_masks


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Center-bias vs size-bias diagnosis")
    parser.add_argument("--runs_root", type=str, required=True)
    parser.add_argument("--skin_root", type=str, required=True)
    parser.add_argument("--dataset", choices=["ph2", "isic2017", "isic2018", "ham10000"], required=True)
    parser.add_argument("--split", choices=["val", "test"], default="test")
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--indices", type=int, nargs="+", required=True,
                        help="Outlier sample indices to diagnose (e.g. the "
                             "unanimous-wrong ones from analyze_agreement.py).")
    parser.add_argument("--out", type=str, default="center_size_bias.csv")
    parser.add_argument("--skip_precondition_check", action="store_true",
                        help="Skip the 'are GT lesions centered in general' pass "
                             "over the full test set (saves a bit of time; that "
                             "pass needs no model inference, just mask geometry, "
                             "so it's cheap, but you can skip it if the loader is slow).")
    args = parser.parse_args()

    cfg = Config.from_args(args)
    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))

    split = "vl" if args.split == "val" else "te"
    ds = build_contour_dataset(cfg.skin_root, args.dataset, split, cfg.n_points,
                               cfg.img_size, augment=False, npy_size=cfg.npy_size)
    loader = DataLoader(ds, batch_size=cfg.batch_size, shuffle=False)

    ckpt_map_cross = find_checkpoints(args.runs_root, CROSS_RUN_EXPERIMENTS)

    print(f"Collecting predictions for {len(args.indices)} requested indices...")
    masks_per_ckpt, gt_masks, ckpts = collect_masks_for_indices(
        ckpt_map_cross, cfg, device, loader, args.indices)

    rows = []
    for idx in args.indices:
        gt = gt_masks[idx]
        pred = combine_masks([masks_per_ckpt[p][idx] for p in ckpts], mode="majority")

        gt_center_dist = normalized_center_distance(gt)
        pred_center_dist = normalized_center_distance(pred)
        gt_area = area_fraction(gt)
        pred_area = area_fraction(pred)
        shift = normalized_centroid_shift(gt, pred)

        area_ratio = (pred_area / gt_area) if gt_area > 0 else None
        # Positive -> prediction is MORE central than GT (evidence for center-bias)
        center_bias_signal = (gt_center_dist - pred_center_dist) if (
            gt_center_dist is not None and pred_center_dist is not None) else None

        rows.append({
            "sample_idx": idx,
            "gt_center_dist": round(gt_center_dist, 4) if gt_center_dist is not None else None,
            "pred_center_dist": round(pred_center_dist, 4) if pred_center_dist is not None else None,
            "center_bias_signal": round(center_bias_signal, 4) if center_bias_signal is not None else None,
            "gt_area_frac": round(gt_area, 4),
            "pred_area_frac": round(pred_area, 4),
            "area_ratio_pred_over_gt": round(area_ratio, 3) if area_ratio is not None else None,
            "centroid_shift": round(shift, 4) if shift is not None else None,
        })

    with open(args.out, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    print(f"\nPer-sample geometry written to {args.out}")

    # ---- aggregate diagnosis ----
    valid = [r for r in rows if r["center_bias_signal"] is not None]
    center_signals = np.array([r["center_bias_signal"] for r in valid])
    area_ratios = np.array([r["area_ratio_pred_over_gt"] for r in valid if r["area_ratio_pred_over_gt"] is not None])
    shifts = np.array([r["centroid_shift"] for r in valid if r["centroid_shift"] is not None])

    print(f"\n=== Aggregate diagnosis over {len(valid)} outlier samples ===")
    print(f"  center_bias_signal (positive = prediction more central than GT):")
    print(f"    mean={center_signals.mean():+.4f}  median={np.median(center_signals):+.4f}"
         f"  n_positive={np.sum(center_signals > 0.02)}/{len(center_signals)}"
         f"  n_negative={np.sum(center_signals < -0.02)}/{len(center_signals)}")
    print(f"  area_ratio_pred_over_gt (>1 = prediction bigger than GT):")
    print(f"    mean={area_ratios.mean():.3f}  median={np.median(area_ratios):.3f}"
         f"  n_bigger={np.sum(area_ratios > 1.1)}/{len(area_ratios)}"
         f"  n_smaller={np.sum(area_ratios < 0.9)}/{len(area_ratios)}")
    print(f"  centroid_shift (0=same location, 1=opposite corners):")
    print(f"    mean={shifts.mean():.4f}  median={np.median(shifts):.4f}"
         f"  n_large_shift(>0.1)={np.sum(shifts > 0.1)}/{len(shifts)}")

    corr = np.corrcoef(center_signals, np.log(area_ratios + 1e-6))[0, 1] if len(area_ratios) == len(center_signals) else float("nan")
    print(f"\n  corr(center_bias_signal, log(area_ratio)) = {corr:.3f}")
    print("  (near 0 -> the two effects are independent; strongly positive ->")
    print("   'bigger' and 'more central' tend to co-occur in these errors,")
    print("   i.e. hard to tell them apart from this data alone.)")

    print(f"\n=== Per-sample detail (sorted by |center_bias_signal|, descending) ===")
    order = sorted(valid, key=lambda r: -abs(r["center_bias_signal"]))
    for r in order:
        tag = "MORE CENTRAL" if r["center_bias_signal"] > 0.02 else (
              "LESS CENTRAL" if r["center_bias_signal"] < -0.02 else "similar")
        print(f"  idx {r['sample_idx']:5d}  center_signal={r['center_bias_signal']:+.3f} [{tag:>13}]"
             f"  area_ratio={r['area_ratio_pred_over_gt']:.2f}x  shift={r['centroid_shift']:.3f}")

    if not args.skip_precondition_check:
        print(f"\nChecking precondition: are GT lesions centered in general "
             f"(not just outliers)? Scanning all {len(ds)} test samples "
             f"(mask geometry only, no model inference)...")
        all_gt = collect_gt_masks_all(loader)
        all_center_dists = np.array([
            d for d in (normalized_center_distance(m) for m in all_gt) if d is not None
        ])
        print(f"  GT centroid distance to image center, over ALL {len(all_center_dists)} test samples:")
        print(f"    mean={all_center_dists.mean():.4f}  median={np.median(all_center_dists):.4f}"
             f"  (0=perfectly centered, ~0.5=halfway to edge, 1=corner)")
        if all_center_dists.mean() < 0.15:
            print("  -> GT lesions ARE reliably centered in this dataset.")
            print("     A center-prior is a reasonable, low-risk thing to try.")
        else:
            print("  -> GT lesions are NOT reliably centered in this dataset.")
            print("     A center-prior would likely help these outliers but HURT")
            print("     the (many) legitimately off-center cases -- do not add one")
            print("     without weighting it very lightly, or without further checks.")


if __name__ == "__main__":
    main()