"""Per-sample checkpoint AGREEMENT analysis.

Extends analyze_outliers.py with the missing piece: for each image, how much
do the 15 individual checkpoints (before any ensembling) actually disagree
with each other? This distinguishes two structurally different failure
modes that look identical in the aggregate Dice/IoU numbers but need
completely different fixes:

    (a) HIGH-DISAGREEMENT outliers: checkpoints disagree a lot with each
        other on this image (e.g. two genuinely different plausible
        contours get roughly equal votes -- the dual_candidate_ambiguity
        pattern). Majority voting on these MUST hurt or do nothing useful,
        since it averages between two incompatible hypotheses instead of
        picking one. Fix: needs a per-sample "pick one hypothesis" strategy
        (e.g. pick the checkpoint whose x0-confidence is highest for this
        sample, or a learned selector), not more voting.

    (b) UNANIMOUS-BUT-WRONG outliers: all 15 checkpoints agree closely with
        each other AND are all wrong in the same way. This is not an
        ensembling problem at all -- it's a shared model/data bias (e.g. a
        lesion type or artifact the whole training setup handles badly).
        Fix: needs actual data/architecture work (hard-example mining,
        targeted augmentation, or accepting it as a dataset limit), not
        post-processing.

Two agreement metrics are computed per image, both purely from the 15
binary masks (no GT needed for the metric itself, only for bucketing):

    mean_pairwise_iou   -- average IoU between every pair of the 15
        checkpoint masks. High (~1.0) = checkpoints agree strongly.
        Low (<0.5) = checkpoints disagree substantially.
    vote_entropy        -- per-pixel vote fraction p (0..1, how many of the
        15 checkpoints call this pixel foreground), turned into a binary
        entropy -p*log(p) - (1-p)*log(1-p), averaged over pixels near the
        object (see `_boundary_band`). High entropy = lots of pixels where
        the vote was close to a coin flip (contested boundary pixels).
        Low entropy = checkpoints agree pixel-by-pixel almost everywhere.

This reuses the exact same prediction-collection pass as analyze_outliers.py
(same checkpoints, same TTA-off masks) so results line up directly with
that script's per-sample CSV -- run this AFTER analyze_outliers.py, or
standalone; it recomputes predictions itself either way.

Usage:
    python -m src.analyze_agreement \
        --runs_root x0_prediction_backup/runs/ablation_v15a_local_drop \
        --skin_root data/datasets \
        --dataset isic2018 --split test \
        --out per_sample_agreement.csv
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

OUTLIER_THRESHOLD = 0.70
HIGH_DISAGREEMENT_IOU = 0.60     # mean pairwise IoU below this -> "high disagreement" bucket
N_SHOW = 20


# ---------------------------------------------------------------------------
# Agreement metrics (pure function of a list of binary masks, no GT)
# ---------------------------------------------------------------------------

def mean_pairwise_iou(mask_list):
    """Average IoU over all C(n,2) unordered pairs of binary masks."""
    n = len(mask_list)
    if n < 2:
        return 1.0
    stack = [np.asarray(m).astype(bool) for m in mask_list]
    ious = []
    for i in range(n):
        for j in range(i + 1, n):
            inter = np.logical_and(stack[i], stack[j]).sum()
            union = np.logical_or(stack[i], stack[j]).sum()
            ious.append(inter / union if union > 0 else 1.0)
    return float(np.mean(ious))


def vote_entropy(mask_list, eps=1e-6):
    """Mean binary entropy of the per-pixel vote fraction, restricted to
    pixels where at least one checkpoint disagreed with the rest (i.e. the
    contested region) -- averaging entropy over the whole image would mostly
    just measure lesion size (huge uncontested background/foreground area
    both have p in {0,1}, entropy 0), diluting the signal we actually want.
    """
    stack = np.stack([np.asarray(m).astype(np.float32) for m in mask_list], axis=0)
    p = stack.mean(axis=0)  # [H, W], vote fraction per pixel

    contested = (p > 0) & (p < 1)
    if not np.any(contested):
        return 0.0  # perfect unanimous agreement everywhere

    pc = p[contested]
    ent = -pc * np.log(pc + eps) - (1 - pc) * np.log(1 - pc + eps)
    return float(np.mean(ent))


def contested_area_fraction(mask_list):
    """Fraction of the image's foreground-union area where checkpoints
    disagree (any-vote minus all-vote, divided by any-vote) -- an intuitive
    companion to vote_entropy: what fraction of "the lesion, generously
    defined" is actually contested rather than agreed upon."""
    stack = np.stack([np.asarray(m).astype(bool) for m in mask_list], axis=0)
    any_vote = np.any(stack, axis=0)
    all_vote = np.all(stack, axis=0)
    union_area = any_vote.sum()
    if union_area == 0:
        return 0.0
    contested = any_vote.sum() - all_vote.sum()
    return float(contested / union_area)


# ---------------------------------------------------------------------------
# Collect masks (mirrors analyze_outliers.collect_predictions, no-TTA only
# since agreement here is about cross-checkpoint disagreement, not TTA views)
# ---------------------------------------------------------------------------

def collect_no_tta_masks(ckpt_map_cross, cfg, device, loader):
    diffusion = GaussianDiffusion(cfg.timesteps, cfg.beta_start, cfg.beta_end, device=device)
    n_images = len(loader.dataset)

    cross_ckpts = [p for paths in ckpt_map_cross.values() for p in paths]
    masks_per_ckpt = {p: [None] * n_images for p in cross_ckpts}
    gt_masks = [None] * n_images

    for ckpt_path in cross_ckpts:
        print(f"  predicting with {ckpt_path} ...")
        encoder, denoiser, snapper, proposal_head = load_checkpoint(ckpt_path, cfg, device)

        idx = 0
        for images, gt_points, gt_batch in loader:
            b = images.shape[0]
            point_sets = _predict_points_single_checkpoint(
                encoder, denoiser, diffusion, images, cfg, device,
                snapper=snapper, proposal_head=proposal_head, tta_variants=("none",),
            )
            for i in range(b):
                gi = idx + i
                if gt_masks[gi] is None:
                    gt_masks[gi] = gt_batch[i].squeeze().cpu().numpy().astype(np.uint8)
                masks_per_ckpt[ckpt_path][gi] = points_to_mask(point_sets[0][i], cfg.img_size)
            idx += b

        del encoder, denoiser, snapper, proposal_head
        if device.type == "cuda":
            torch.cuda.empty_cache()

    return masks_per_ckpt, gt_masks, cross_ckpts


# ---------------------------------------------------------------------------
# Main analysis
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Per-sample checkpoint agreement analysis")
    parser.add_argument("--runs_root", type=str, required=True)
    parser.add_argument("--skin_root", type=str, required=True)
    parser.add_argument("--dataset", choices=["ph2", "isic2017", "isic2018", "ham10000"], required=True)
    parser.add_argument("--split", choices=["val", "test"], default="test")
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--out", type=str, default="per_sample_agreement.csv")
    args = parser.parse_args()

    cfg = Config.from_args(args)
    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))

    split = "vl" if args.split == "val" else "te"
    ds = build_contour_dataset(cfg.skin_root, args.dataset, split, cfg.n_points,
                               cfg.img_size, augment=False, npy_size=cfg.npy_size)
    loader = DataLoader(ds, batch_size=cfg.batch_size, shuffle=False)

    ckpt_map_cross = find_checkpoints(args.runs_root, CROSS_RUN_EXPERIMENTS)

    print("Collecting per-checkpoint predictions (no TTA)...")
    masks_per_ckpt, gt_masks, ckpts = collect_no_tta_masks(ckpt_map_cross, cfg, device, loader)
    n_images = len(gt_masks)

    print("Computing per-sample agreement metrics + majority-vote Dice...")
    rows = []
    for i in range(n_images):
        mask_list = [masks_per_ckpt[p][i] for p in ckpts]

        mv_mask = combine_masks(mask_list, mode="majority")
        mv_dice = dice_score(mv_mask, gt_masks[i])
        mv_iou = iou_score(mv_mask, gt_masks[i])

        # Best/worst single checkpoint on this sample, for reference.
        single_dices = [dice_score(m, gt_masks[i]) for m in mask_list]

        rows.append({
            "sample_idx": i,
            "mean_pairwise_iou": round(mean_pairwise_iou(mask_list), 4),
            "vote_entropy": round(vote_entropy(mask_list), 4),
            "contested_area_frac": round(contested_area_fraction(mask_list), 4),
            "majority_vote_dice": round(mv_dice, 4),
            "majority_vote_iou": round(mv_iou, 4),
            "best_single_dice": round(max(single_dices), 4),
            "worst_single_dice": round(min(single_dices), 4),
            "single_dice_spread": round(max(single_dices) - min(single_dices), 4),
        })

    with open(args.out, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    print(f"Per-sample agreement metrics written to {args.out}")

    # ---- bucketed interpretation ----
    mv_dice_arr = np.array([r["majority_vote_dice"] for r in rows])
    pairwise_iou_arr = np.array([r["mean_pairwise_iou"] for r in rows])
    entropy_arr = np.array([r["vote_entropy"] for r in rows])
    spread_arr = np.array([r["single_dice_spread"] for r in rows])

    outlier_idx = np.where(mv_dice_arr < OUTLIER_THRESHOLD)[0]
    good_idx = np.where(mv_dice_arr >= OUTLIER_THRESHOLD)[0]

    print(f"\n=== Agreement stats: outliers (majority-vote Dice < {OUTLIER_THRESHOLD}, n={len(outlier_idx)}) "
         f"vs rest (n={len(good_idx)}) ===")
    print(f"  mean pairwise IoU  -- outliers: {pairwise_iou_arr[outlier_idx].mean():.4f}"
         f"   rest: {pairwise_iou_arr[good_idx].mean():.4f}")
    print(f"  vote entropy       -- outliers: {entropy_arr[outlier_idx].mean():.4f}"
         f"   rest: {entropy_arr[good_idx].mean():.4f}")
    print(f"  single-ckpt spread -- outliers: {spread_arr[outlier_idx].mean():.4f}"
         f"   rest: {spread_arr[good_idx].mean():.4f}")

    print(f"\n=== Outlier breakdown by disagreement level (mean pairwise IoU < {HIGH_DISAGREEMENT_IOU} = high disagreement) ===")
    high_dis = outlier_idx[pairwise_iou_arr[outlier_idx] < HIGH_DISAGREEMENT_IOU]
    low_dis = outlier_idx[pairwise_iou_arr[outlier_idx] >= HIGH_DISAGREEMENT_IOU]
    print(f"  HIGH-disagreement outliers (likely dual-candidate ambiguity): {len(high_dis)} / {len(outlier_idx)}")
    print(f"    -> voting structurally can't fix these; needs a 'pick one hypothesis' strategy")
    print(f"  UNANIMOUS-but-wrong outliers (checkpoints agree, all wrong): {len(low_dis)} / {len(outlier_idx)}")
    print(f"    -> not an ensembling problem; needs data/model-level work (hard-example mining, etc.)")

    print(f"\n=== Detail: all {len(outlier_idx)} outlier samples, sorted by disagreement (most disagreement first) ===")
    order = outlier_idx[np.argsort(pairwise_iou_arr[outlier_idx])]
    for i in order:
        r = rows[i]
        tag = "HIGH-DISAGREEMENT" if r["mean_pairwise_iou"] < HIGH_DISAGREEMENT_IOU else "unanimous-wrong"
        print(f"  idx {i:5d} [{tag:>18}]  pairwise_iou={r['mean_pairwise_iou']:.3f}"
             f"  entropy={r['vote_entropy']:.3f}  mv_dice={r['majority_vote_dice']:.3f}"
             f"  best_single={r['best_single_dice']:.3f}  worst_single={r['worst_single_dice']:.3f}")

    print(f"\n=== Correlation check: does disagreement predict low Dice in general? ===")
    corr = np.corrcoef(pairwise_iou_arr, mv_dice_arr)[0, 1]
    print(f"  corr(mean_pairwise_iou, majority_vote_dice) = {corr:.3f}  (positive = agreement predicts quality)")


if __name__ == "__main__":
    main()