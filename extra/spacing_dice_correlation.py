"""Analysis: per-sample Dice vs. stem-scale tangential point spacing.

Question this answers: does the tangential sampling gap on the full-resolution
stem (mean ~5.5 feature cells between neighboring contour points, per
point_space_diagnostic.py) actually predict WHICH samples the model gets
wrong -- or is the gap uniform across the dataset and unrelated to the
model's actual failure modes?

If per-sample spacing (mean or max) correlates with per-sample Dice (worse
Dice on samples with wider spacing), that's evidence the gap is a real,
exploitable failure mode -- worth fixing with the tangential-sampling /
blur / attention ablation. If there's no correlation, the gap may be a red
herring: uniformly present but not actually limiting, e.g. because the
denoiser's later mixing (transformer/circular conv) already compensates,
or because failures are dominated by something else entirely (topology,
low contrast, etc).

Usage (run on the server, adjust the wiring block near the bottom to match
your actual sample.py-style checkpoint loading):

    python -m x0_prediction_V6.spacing_dice_correlation \
        --ckpt x0_prediction_V6/runs/ablation_v7_laziness_on_snapboth030/baseline_seed_01/best.pth \
        --split test
"""

import argparse

import numpy as np
import torch
from torch.utils.data import DataLoader

from scipy.stats import pearsonr, spearmanr  # already a transitive dep via matplotlib/sklearn-adjacent tooling; if missing: pip install scipy --break-system-packages

from .config import Config
from .data import build_contour_dataset
from .sample import load_checkpoint
from .diffusion import GaussianDiffusion
from .utils import dice_score, iou_score, points_to_mask


def per_contour_stem_spacing(points_xy: np.ndarray, img_size) -> dict:
    """points_xy: [N,2] in [-1,1]. Returns mean/max spacing in stem pixels
    (stem = full image resolution, stride 1 -> 1 cell == 1 pixel)."""
    if isinstance(img_size, (tuple, list)):
        img_h, img_w = float(img_size[0]), float(img_size[1])
    else:
        img_h = img_w = float(img_size)

    px = (points_xy + 1.0) * 0.5 * np.array([img_w, img_h])
    next_px = np.roll(px, shift=-1, axis=0)
    seg_len_px = np.linalg.norm(next_px - px, axis=-1)

    return {
        "mean_spacing_px": float(seg_len_px.mean()),
        "max_spacing_px": float(seg_len_px.max()),
        "p90_spacing_px": float(np.percentile(seg_len_px, 90)),
    }


@torch.no_grad()
def run_analysis(ckpt_path, dataset, split, device, max_samples=None):
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

    records = []  # one dict per sample: dice, iou, mean_spacing_px, max_spacing_px, p90_spacing_px

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

        for i in range(images.shape[0]):
            pred_mask = points_to_mask(pred_points[i], cfg.img_size)
            gt_mask = gt_masks[i].squeeze().cpu().numpy().astype(np.uint8)
            d = dice_score(pred_mask, gt_mask)
            j = iou_score(pred_mask, gt_mask)

            gt_np = gt_points[i].cpu().numpy()
            spacing = per_contour_stem_spacing(gt_np, cfg.img_size)

            records.append({
                "dice": d, "iou": j,
                **spacing,
            })
            n_seen += 1
            if max_samples is not None and n_seen >= max_samples:
                break

    return records


def report(records):
    dice = np.array([r["dice"] for r in records])
    mean_sp = np.array([r["mean_spacing_px"] for r in records])
    max_sp = np.array([r["max_spacing_px"] for r in records])
    p90_sp = np.array([r["p90_spacing_px"] for r in records])

    print(f"N = {len(records)} test samples\n")
    print(f"Dice:            mean={dice.mean():.4f}  std={dice.std():.4f}  min={dice.min():.4f}  max={dice.max():.4f}")
    print(f"mean spacing:    mean={mean_sp.mean():.2f}px  std={mean_sp.std():.2f}px")
    print(f"max spacing:     mean={max_sp.mean():.2f}px  std={max_sp.std():.2f}px")
    print(f"p90 spacing:     mean={p90_sp.mean():.2f}px  std={p90_sp.std():.2f}px\n")

    for name, arr in [("mean_spacing", mean_sp), ("max_spacing", max_sp), ("p90_spacing", p90_sp)]:
        pear_r, pear_p = pearsonr(arr, dice)
        spear_r, spear_p = spearmanr(arr, dice)
        print(f"{name:>14} vs Dice -> Pearson r={pear_r:+.3f} (p={pear_p:.4f})  "
              f"Spearman rho={spear_r:+.3f} (p={spear_p:.4f})")

    print(
        "\nInterpretation: a negative, statistically significant correlation "
        "(r/rho meaningfully below 0, p < 0.05) means samples with wider "
        "tangential point spacing tend to have LOWER Dice -- i.e. the gap "
        "identified in point_space_diagnostic.py is predictive of where the "
        "model actually fails, which supports building a targeted fix. A "
        "near-zero or non-significant correlation means spacing alone does "
        "not explain failure cases well -- the mixing that already happens "
        "downstream (transformer / circular conv / snapper) may already be "
        "compensating, or failures are dominated by other factors."
    )

    # Bottom/top decile comparison -- often more informative than a single
    # global correlation coefficient, since correlations can be washed out
    # by many "easy" samples where spacing doesn't matter either way.
    order = np.argsort(dice)
    n = len(records)
    worst_n = max(1, n // 10)
    worst_idx = order[:worst_n]
    best_idx = order[-worst_n:]

    print(f"\nWorst {worst_n} samples (lowest Dice): mean spacing = {mean_sp[worst_idx].mean():.2f}px, "
          f"max spacing = {max_sp[worst_idx].mean():.2f}px")
    print(f"Best  {worst_n} samples (highest Dice): mean spacing = {mean_sp[best_idx].mean():.2f}px, "
          f"max spacing = {max_sp[best_idx].mean():.2f}px")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ckpt", type=str, required=True)
    parser.add_argument("--dataset", type=str, default="isic2018")
    parser.add_argument("--split", type=str, default="test")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--max_samples", type=int, default=None)
    args = parser.parse_args()

    records = run_analysis(args.ckpt, args.dataset, args.split, args.device,
                           max_samples=args.max_samples)
    report(records)