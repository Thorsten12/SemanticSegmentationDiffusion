"""Per-sample diagnostic analysis: single-checkpoint vs ensemble/TTA predictions.

Answers the actual question that matters before doing more global tuning:
    "Is the +0.5 Dice from ensembling/TTA coming from fixing the known
     dual_candidate_ambiguity outliers (Dice 0.38-0.58), or just nudging
     already-good mid-field samples up by a fraction of a point?"

For every test image, computes Dice/IoU for:
    - single_best        (one checkpoint, no tricks)
    - seed_ensemble       (5 seeds, majority vote)
    - cross_run_ensemble  (15 checkpoints, majority vote)
    - cross_run_ensemble+tta (15 checkpoints x 4 TTA views)

...then writes a per-sample CSV and prints a breakdown:
    1. Outlier bucket (Dice < 0.70 under single_best) -- how many remain
       outliers under each stronger config, and by how much did they move.
    2. Mid-field bucket (Dice in [0.85, 0.95] under single_best) -- how much
       did ensembling/TTA move these, on average.
    3. Top movers -- the N samples with the largest Dice delta between
       single_best and the strongest config, in both directions (biggest
       fixes AND any regressions introduced by ensembling).
    4. A scatter-style bucketed summary (mean delta per single_best-Dice
       decile) so you can see whether gains concentrate at the low end.

This reuses run_ensemble()'s per-checkpoint prediction machinery directly
(rather than re-implementing sampling) so the numbers are guaranteed
consistent with what run_postprocessing_ablation.py reported.

Usage:
    python -m src.analyze_outliers \
        --runs_root x0_prediction_backup/runs/ablation_v15a_local_drop \
        --skin_root /path/to/data_root \
        --dataset isic2018 --split test \
        --out per_sample_results.csv
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
from .utils import dice_score, iou_score, points_to_mask
from .postprocess import combine_masks
from .ensemble_predict import find_checkpoints, _predict_points_single_checkpoint

BEST_EXPERIMENT = "drop_1"
CROSS_RUN_EXPERIMENTS = ["drop_1", "drop_2", "drop_2_topk_none"]

OUTLIER_THRESHOLD = 0.70          # matches the "compromise case" ballpark (0.38-0.58) with margin
MIDFIELD_RANGE = (0.85, 0.95)
N_TOP_MOVERS = 15


# ---------------------------------------------------------------------------
# Collect per-sample masks for every config, in one pass over checkpoints
# ---------------------------------------------------------------------------

def collect_predictions(ckpt_map_best, ckpt_map_cross, cfg, device, loader):
    """Single pass over all needed checkpoints; returns a dict of
    {config_name: [mask_per_image, ...]} plus the cached GT masks.

    Runs each checkpoint exactly once (with and without TTA views computed
    together), and reuses drop_1's 5 checkpoints for both `seed_ensemble`
    and as part of `cross_run_ensemble` rather than loading them twice.
    """
    diffusion = GaussianDiffusion(cfg.timesteps, cfg.beta_start, cfg.beta_end, device=device)
    n_images = len(loader.dataset)

    best_ckpts = ckpt_map_best[BEST_EXPERIMENT]
    cross_ckpts = [p for paths in ckpt_map_cross.values() for p in paths]
    all_ckpts = sorted(set(best_ckpts) | set(cross_ckpts))
    is_best = {p: p in best_ckpts for p in all_ckpts}

    # masks_no_tta[ckpt][i]   -> single binary mask (no TTA), image i
    # masks_tta[ckpt][i]      -> list of 4 binary masks (one per TTA view), image i
    masks_no_tta = {p: [None] * n_images for p in all_ckpts}
    masks_tta = {p: [None] * n_images for p in all_ckpts}
    gt_masks = [None] * n_images

    for ckpt_path in all_ckpts:
        print(f"  predicting with {ckpt_path} ...")
        encoder, denoiser, snapper, proposal_head = load_checkpoint(ckpt_path, cfg, device)

        idx = 0
        for images, gt_points, gt_batch in loader:
            b = images.shape[0]
            point_sets = _predict_points_single_checkpoint(
                encoder, denoiser, diffusion, images, cfg, device,
                snapper=snapper, proposal_head=proposal_head,
                tta_variants=("none", "hflip", "vflip", "hvflip"),
            )
            # point_sets[0] = no-flip view -> doubles as the "no TTA" prediction
            for i in range(b):
                gi = idx + i
                if gt_masks[gi] is None:
                    gt_masks[gi] = gt_batch[i].squeeze().cpu().numpy().astype(np.uint8)

                no_tta_mask = points_to_mask(point_sets[0][i], cfg.img_size)
                masks_no_tta[ckpt_path][gi] = no_tta_mask

                tta_masks = [points_to_mask(vp[i], cfg.img_size) for vp in point_sets]
                masks_tta[ckpt_path][gi] = tta_masks
            idx += b

        del encoder, denoiser, snapper, proposal_head
        if device.type == "cuda":
            torch.cuda.empty_cache()

    # ---- assemble the four config variants from the collected masks ----
    results = {}

    # single_best: first checkpoint of drop_1, no TTA
    single_ckpt = best_ckpts[0]
    results["single_best"] = [masks_no_tta[single_ckpt][i] for i in range(n_images)]

    # seed_ensemble: majority vote over drop_1's 5 checkpoints, no TTA
    results["seed_ensemble"] = [
        combine_masks([masks_no_tta[p][i] for p in best_ckpts], mode="majority")
        for i in range(n_images)
    ]

    # cross_run_ensemble: majority vote over all 15 checkpoints, no TTA
    results["cross_run_ensemble"] = [
        combine_masks([masks_no_tta[p][i] for p in cross_ckpts], mode="majority")
        for i in range(n_images)
    ]

    # cross_run_ensemble+tta: majority vote over all 15 checkpoints x 4 views (60 votes)
    results["cross_run_ensemble+tta"] = [
        combine_masks(
            [m for p in cross_ckpts for m in masks_tta[p][i]], mode="majority"
        )
        for i in range(n_images)
    ]

    return results, gt_masks


# ---------------------------------------------------------------------------
# Scoring + analysis
# ---------------------------------------------------------------------------

def score_all(results, gt_masks):
    """{config: [dice_per_image]}, {config: [iou_per_image]}"""
    dices, ious = {}, {}
    for name, masks in results.items():
        d = [dice_score(masks[i], gt_masks[i]) for i in range(len(gt_masks))]
        j = [iou_score(masks[i], gt_masks[i]) for i in range(len(gt_masks))]
        dices[name] = d
        ious[name] = j
    return dices, ious


def print_outlier_breakdown(dices, best_config="cross_run_ensemble+tta"):
    base = np.array(dices["single_best"])
    strong = np.array(dices[best_config])
    delta = strong - base

    outlier_idx = np.where(base < OUTLIER_THRESHOLD)[0]
    mid_idx = np.where((base >= MIDFIELD_RANGE[0]) & (base <= MIDFIELD_RANGE[1]))[0]

    print(f"\n=== Outlier bucket (single_best Dice < {OUTLIER_THRESHOLD}) ===")
    print(f"  n = {len(outlier_idx)} / {len(base)} samples")
    if len(outlier_idx) > 0:
        print(f"  single_best mean Dice here:  {base[outlier_idx].mean():.4f}")
        print(f"  {best_config} mean Dice here: {strong[outlier_idx].mean():.4f}")
        print(f"  mean delta: {delta[outlier_idx].mean():+.4f}")
        still_outlier = np.sum(strong[outlier_idx] < OUTLIER_THRESHOLD)
        print(f"  still below {OUTLIER_THRESHOLD} after {best_config}: {still_outlier} / {len(outlier_idx)}")
    else:
        print("  (none found at this threshold -- consider lowering OUTLIER_THRESHOLD)")

    print(f"\n=== Mid-field bucket (single_best Dice in {MIDFIELD_RANGE}) ===")
    print(f"  n = {len(mid_idx)} / {len(base)} samples")
    if len(mid_idx) > 0:
        print(f"  mean delta: {delta[mid_idx].mean():+.4f}")

    print(f"\n=== Where does the aggregate gain come from? ===")
    total_gain = delta.sum()
    outlier_gain = delta[outlier_idx].sum() if len(outlier_idx) else 0.0
    mid_gain = delta[mid_idx].sum() if len(mid_idx) else 0.0
    other_idx = np.setdiff1d(np.arange(len(base)),
                             np.union1d(outlier_idx, mid_idx))
    other_gain = delta[other_idx].sum() if len(other_idx) else 0.0
    print(f"  total summed delta:        {total_gain:+.3f}  (n={len(base)})")
    print(f"  from outlier bucket:       {outlier_gain:+.3f}  ({100*outlier_gain/total_gain if total_gain else 0:.1f}% of total)")
    print(f"  from mid-field bucket:     {mid_gain:+.3f}  ({100*mid_gain/total_gain if total_gain else 0:.1f}% of total)")
    print(f"  from everything else:      {other_gain:+.3f}  ({100*other_gain/total_gain if total_gain else 0:.1f}% of total)")

    print(f"\n=== Top {N_TOP_MOVERS} biggest FIXES ({best_config} vs single_best) ===")
    order = np.argsort(-delta)
    for rank, i in enumerate(order[:N_TOP_MOVERS]):
        print(f"  idx {i:5d}: single_best={base[i]:.4f} -> {best_config}={strong[i]:.4f}  (delta {delta[i]:+.4f})")

    print(f"\n=== Top {N_TOP_MOVERS} biggest REGRESSIONS ({best_config} vs single_best) ===")
    for rank, i in enumerate(order[-N_TOP_MOVERS:][::-1]):
        if delta[i] >= 0:
            break
        print(f"  idx {i:5d}: single_best={base[i]:.4f} -> {best_config}={strong[i]:.4f}  (delta {delta[i]:+.4f})")

    print(f"\n=== Mean delta by single_best-Dice decile ===")
    deciles = np.digitize(base, np.linspace(0, 1, 11)) - 1
    deciles = np.clip(deciles, 0, 9)
    for d in range(10):
        mask = deciles == d
        if mask.sum() == 0:
            continue
        lo, hi = d / 10, (d + 1) / 10
        print(f"  Dice in [{lo:.1f},{hi:.1f}): n={mask.sum():4d}  mean single_best={base[mask].mean():.4f}"
             f"  mean delta={delta[mask].mean():+.4f}")


def write_csv(path, dices, ious, configs):
    n = len(next(iter(dices.values())))
    fieldnames = ["sample_idx"]
    for c in configs:
        fieldnames += [f"{c}_dice", f"{c}_iou"]
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for i in range(n):
            row = {"sample_idx": i}
            for c in configs:
                row[f"{c}_dice"] = round(dices[c][i], 4)
                row[f"{c}_iou"] = round(ious[c][i], 4)
            writer.writerow(row)
    print(f"\nPer-sample results written to {path}")


def main():
    parser = argparse.ArgumentParser(description="Per-sample outlier / gain-source analysis")
    parser.add_argument("--runs_root", type=str, required=True)
    parser.add_argument("--skin_root", type=str, required=True)
    parser.add_argument("--dataset", choices=["ph2", "isic2017", "isic2018", "ham10000"], required=True)
    parser.add_argument("--split", choices=["val", "test"], default="test")
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--out", type=str, default="per_sample_results.csv")
    parser.add_argument("--best_config", type=str, default="cross_run_ensemble+tta",
                        choices=["seed_ensemble", "cross_run_ensemble", "cross_run_ensemble+tta"],
                        help="Which strong config to compare single_best against in the "
                             "outlier/mover breakdown printed to stdout.")
    args = parser.parse_args()

    cfg = Config.from_args(args)
    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))

    split = "vl" if args.split == "val" else "te"
    ds = build_contour_dataset(cfg.skin_root, args.dataset, split, cfg.n_points,
                               cfg.img_size, augment=False, npy_size=cfg.npy_size)
    loader = DataLoader(ds, batch_size=cfg.batch_size, shuffle=False)

    ckpt_map_best = find_checkpoints(args.runs_root, [BEST_EXPERIMENT])
    ckpt_map_cross = find_checkpoints(args.runs_root, CROSS_RUN_EXPERIMENTS)

    print("Collecting predictions (single pass over all needed checkpoints)...")
    results, gt_masks = collect_predictions(ckpt_map_best, ckpt_map_cross, cfg, device, loader)

    dices, ious = score_all(results, gt_masks)

    print("\n=== Aggregate summary (sanity check vs. ablation script) ===")
    for name in results:
        print(f"  {name:<28} Dice {np.mean(dices[name]):.4f} | IoU {np.mean(ious[name]):.4f}")

    print_outlier_breakdown(dices, best_config=args.best_config)
    write_csv(args.out, dices, ious, list(results.keys()))


if __name__ == "__main__":
    main()