"""Seed-ensemble / cross-run-ensemble / TTA prediction for P2SDiff.

Combines predictions from multiple checkpoints (same config, different
seeds, and/or different configs) and/or multiple test-time-augmented views
of each image, via pixel-wise majority-vote mask combination
(see postprocess.combine_masks). No retraining required.

Checkpoint layout assumed (matches the ablation_v15a_local_drop example):

    <runs_root>/<run_name>/<experiment>_seed_<NN>/best.pth

e.g.
    x0_prediction_backup/runs/ablation_v15a_local_drop/drop_1_seed_01/best.pth
    x0_prediction_backup/runs/ablation_v15a_local_drop/drop_1_seed_02/best.pth
    ...

Usage (seed-ensemble over one experiment's 5 seeds):
    python -m src.ensemble_predict \
        --runs_root x0_prediction_backup/runs/ablation_v15a_local_drop \
        --experiments drop_1 \
        --dataset isic2018 --split test

Usage (cross-run ensemble over the 3 best experiments, all seeds pooled):
    python -m src.ensemble_predict \
        --runs_root x0_prediction_backup/runs/ablation_v15a_local_drop \
        --experiments drop_1 drop_2 drop_2_topk_none \
        --dataset isic2018 --split test

Add --tta to additionally average each checkpoint's prediction over
horizontal-flip / vertical-flip / both-flip views of the input image.
Add --keep_largest_component and/or --smooth_contour for the extra
post-processing steps (composable with any of the above).
"""

import argparse
import glob
import os
from itertools import product

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from .config import Config
from .data import build_contour_dataset
from .diffusion import GaussianDiffusion
from .sample import load_checkpoint
from .utils import dice_score, iou_score, points_to_mask
from .postprocess import combine_masks, keep_largest_component, smooth_contour_points


# ---------------------------------------------------------------------------
# Checkpoint discovery
# ---------------------------------------------------------------------------

def find_checkpoints(runs_root, experiments):
    """Return {experiment_name: [ckpt_path_seed01, ckpt_path_seed02, ...]}.

    Globs "<runs_root>/<experiment>_seed_*/best.pth" for each requested
    experiment name and sorts by seed number so results are reproducible.
    Raises if an experiment has zero matching checkpoints (fail loudly
    rather than silently ensembling over fewer seeds than intended).
    """
    found = {}
    for exp in experiments:
        pattern = os.path.join(runs_root, f"{exp}_seed_*", "best.pth")
        paths = sorted(glob.glob(pattern))
        if not paths:
            raise FileNotFoundError(
                f"No checkpoints found for experiment '{exp}' with pattern:\n  {pattern}\n"
                f"Check --runs_root and the experiment name spelling."
            )
        found[exp] = paths
    return found


# ---------------------------------------------------------------------------
# TTA: flip an image, sample, flip the predicted points back
# ---------------------------------------------------------------------------

TTA_VARIANTS = {
    "none": (False, False),
    "hflip": (True, False),
    "vflip": (False, True),
    "hvflip": (True, True),
}


def _apply_flip(images, hflip, vflip):
    if hflip:
        images = torch.flip(images, dims=[3])  # flip width
    if vflip:
        images = torch.flip(images, dims=[2])  # flip height
    return images


def _unflip_points(points, hflip, vflip):
    """points: [B, N, 2] in [-1, 1] (x, y). Undo the same flip applied to the image."""
    points = points.clone()
    if hflip:
        points[..., 0] = -points[..., 0]
    if vflip:
        points[..., 1] = -points[..., 1]
    return points


@torch.no_grad()
def _predict_points_single_checkpoint(encoder, denoiser, diffusion, images, cfg, device,
                                      snapper=None, proposal_head=None, tta_variants=("none",)):
    """Run DDIM sampling for one checkpoint, optionally over several TTA views.

    Returns a list of [B, N, 2] point tensors (CPU), one per requested TTA
    variant, already flipped back into the original (unflipped) coordinate
    frame so they're directly comparable/combinable across variants.
    """
    encoder.eval(); denoiser.eval()
    if snapper is not None:
        snapper.eval()
    if proposal_head is not None:
        proposal_head.eval()

    results = []
    for variant in tta_variants:
        hflip, vflip = TTA_VARIANTS[variant]
        imgs_v = _apply_flip(images, hflip, vflip).to(device)

        raw = encoder.extract(imgs_v)
        proposal = proposal_head(raw) if proposal_head is not None else None
        cond_fn = lambda t_b: encoder.fuse(raw, t_b)
        shape = (imgs_v.shape[0], cfg.n_points, 2)

        pred_points = diffusion.ddim_sample(
            denoiser, cond_fn, shape,
            proposal=proposal, proposal_target=cfg.proposal_target,
            ddim_steps=cfg.ddim_steps, guidance_scale=cfg.guidance_scale, clamp=1.0,
            snapper=None, snap_maps=None, snap_image=None, snap_t_threshold=None,
            snap_every=1, collect_every=0,
        )
        if snapper is not None:
            pred_points = snapper(pred_points, raw, image=imgs_v, hard=True)

        pred_points = _unflip_points(pred_points, hflip, vflip).cpu()
        results.append(pred_points)
    return results


# ---------------------------------------------------------------------------
# Main ensembling loop
# ---------------------------------------------------------------------------

def run_ensemble(ckpt_paths, cfg, device, loader,
                 tta_variants=("none",), vote_threshold=0.5,
                 keep_largest=False, keep_ratio=0.0,
                 smooth=False, smooth_window=9, smooth_polyorder=2):
    """Evaluate an ensemble of checkpoints (+ optional TTA views each) over
    `loader`, combining all resulting binary masks per image via majority
    vote, with optional post-processing. Returns (mean_dice, mean_iou).

    Loads and discards one checkpoint fully before loading the next, to keep
    peak GPU memory to a single model regardless of ensemble size (trades
    some redundant encoder.extract() work across images vs. holding every
    checkpoint resident -- fine for offline evaluation).
    """
    diffusion = GaussianDiffusion(cfg.timesteps, cfg.beta_start, cfg.beta_end, device=device)

    # per_image_masks[i] accumulates one binary mask per (checkpoint, TTA
    # variant) combination for dataset image i, across the whole loader.
    n_images = len(loader.dataset)
    per_image_masks = [[] for _ in range(n_images)]
    gt_masks_cache = [None] * n_images

    for ckpt_path in ckpt_paths:
        encoder, denoiser, snapper, proposal_head = load_checkpoint(ckpt_path, cfg, device)

        idx = 0
        for images, gt_points, gt_masks in loader:
            b = images.shape[0]
            point_sets = _predict_points_single_checkpoint(
                encoder, denoiser, diffusion, images, cfg, device,
                snapper=snapper, proposal_head=proposal_head, tta_variants=tta_variants,
            )
            for variant_points in point_sets:  # one [B,N,2] tensor per TTA variant
                for i in range(b):
                    pts = variant_points[i]
                    if smooth:
                        pts = torch.from_numpy(
                            smooth_contour_points(pts.numpy(), smooth_window, smooth_polyorder)
                        )
                    mask = points_to_mask(pts, cfg.img_size)
                    per_image_masks[idx + i].append(mask)
            for i in range(b):
                if gt_masks_cache[idx + i] is None:
                    gt_masks_cache[idx + i] = gt_masks[i].squeeze().cpu().numpy().astype(np.uint8)
            idx += b

        del encoder, denoiser, snapper, proposal_head
        if device.type == "cuda":
            torch.cuda.empty_cache()

    dices, ious = [], []
    for i in range(n_images):
        combined = combine_masks(per_image_masks[i], mode="majority", threshold=vote_threshold)
        if keep_largest:
            combined = keep_largest_component(combined, keep_ratio=keep_ratio)
        d = dice_score(combined, gt_masks_cache[i])
        j = iou_score(combined, gt_masks_cache[i])
        dices.append(d); ious.append(j)

    return float(np.mean(dices)), float(np.mean(ious))


def main():
    parser = argparse.ArgumentParser(description="Seed / cross-run ensemble + TTA evaluation for P2SDiff")
    parser.add_argument("--runs_root", type=str, required=True,
                        help="Directory containing <experiment>_seed_NN/ subfolders, "
                             "e.g. x0_prediction_backup/runs/ablation_v15a_local_drop")
    parser.add_argument("--experiments", type=str, nargs="+", required=True,
                        help="One experiment name -> seed-ensemble over its seeds. "
                             "Multiple names -> cross-run ensemble pooling ALL seeds "
                             "from ALL listed experiments into one majority vote.")
    parser.add_argument("--dataset", choices=["ph2", "isic2017", "isic2018", "ham10000"], required=True)
    parser.add_argument("--skin_root", type=str, required=True)
    parser.add_argument("--split", choices=["val", "test"], default="test")
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--vote_threshold", type=float, default=0.5,
                        help="Fraction of votes (checkpoints x TTA views) required for a "
                             "pixel to be foreground. 0.5 = simple majority.")
    parser.add_argument("--tta", action="store_true",
                        help="Additionally average each checkpoint's prediction over "
                             "hflip/vflip/hvflip views (4x the sampling cost per checkpoint).")
    parser.add_argument("--keep_largest_component", action="store_true")
    parser.add_argument("--keep_ratio", type=float, default=0.0,
                        help="See postprocess.keep_largest_component. 0.0 = strictly "
                             "single largest blob kept.")
    parser.add_argument("--smooth_contour", action="store_true")
    parser.add_argument("--smooth_window", type=int, default=9)
    parser.add_argument("--smooth_polyorder", type=int, default=2)
    args = parser.parse_args()

    cfg = Config.from_args(args)
    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))

    split = "vl" if args.split == "val" else "te"
    ds = build_contour_dataset(cfg.skin_root, args.dataset, split, cfg.n_points,
                               cfg.img_size, augment=False, npy_size=cfg.npy_size)
    loader = DataLoader(ds, batch_size=cfg.batch_size, shuffle=False)

    ckpt_map = find_checkpoints(args.runs_root, args.experiments)
    all_ckpts = [p for paths in ckpt_map.values() for p in paths]
    mode = "seed-ensemble" if len(args.experiments) == 1 else "cross-run-ensemble"
    print(f"[{mode}] experiments={args.experiments} -> {len(all_ckpts)} checkpoint(s):")
    for exp, paths in ckpt_map.items():
        print(f"  {exp}: {len(paths)} seed(s)")

    tta_variants = ("none", "hflip", "vflip", "hvflip") if args.tta else ("none",)

    dice, iou = run_ensemble(
        all_ckpts, cfg, device, loader,
        tta_variants=tta_variants, vote_threshold=args.vote_threshold,
        keep_largest=args.keep_largest_component, keep_ratio=args.keep_ratio,
        smooth=args.smooth_contour, smooth_window=args.smooth_window,
        smooth_polyorder=args.smooth_polyorder,
    )
    n_votes = len(all_ckpts) * len(tta_variants)
    print(f"[{args.split}] {mode} ({n_votes} votes/image, tta={args.tta}, "
         f"components={args.keep_largest_component}, smooth={args.smooth_contour}) "
         f"Dice {dice:.4f} | IoU {iou:.4f}")


if __name__ == "__main__":
    main()