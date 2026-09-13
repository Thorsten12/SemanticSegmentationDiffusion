"""Analysis: what actually predicts per-sample Dice failure?

Follow-up to spacing_dice_correlation.py, which showed tangential point
spacing does NOT meaningfully predict failure (Pearson r ~ -0.03 to -0.05,
worst/best decile nearly identical). This script tests several other
plausible candidates in one pass:

  - lesion_area_frac:   GT mask area / image area (small lesions harder?)
  - mask_compactness:   4*pi*area / perimeter^2 (1.0 = perfect circle;
                        lower = more irregular/elongated shape -> plausibly
                        harder for a low-frequency Fourier/ellipse proposal)
  - mask_solidity:      area / convex_hull_area (< 1 = concave shape,
                        e.g. two blobs, notches -- likely hard for a single
                        closed contour of fixed point count)
  - image_contrast:     std of pixel intensity inside a dilated ring around
                        the GT boundary (low = low-contrast / ambiguous edge)
  - image_brightness:   mean pixel intensity of the whole image
  - gt_perimeter_px:    contour perimeter in pixels (longer/more complex
                        boundary -> more opportunity to get parts wrong)

Usage:
    python -m x0_prediction_V6.failure_mode_correlation \
        --ckpt x0_prediction_V6/runs/ablation_v7_laziness_on_snapboth030/baseline_seed_01/best.pth \
        --dataset isic2018 --split test
"""

import argparse

import numpy as np
import torch
import cv2
from scipy.stats import pearsonr, spearmanr
from torch.utils.data import DataLoader

from .config import Config
from .data import build_contour_dataset
from .sample import load_checkpoint
from .diffusion import GaussianDiffusion
from .utils import dice_score, iou_score, points_to_mask


def per_sample_mask_stats(gt_mask: np.ndarray) -> dict:
    """gt_mask: [H,W] binary uint8. Returns area fraction, compactness,
    solidity, and perimeter (all computed from contours -- robust to small
    masks / multiple blobs by using the largest contour)."""
    h, w = gt_mask.shape
    area_total = float(h * w)
    contours, _ = cv2.findContours(gt_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

    if not contours:
        return {"lesion_area_frac": 0.0, "mask_compactness": 0.0,
                "mask_solidity": 0.0, "gt_perimeter_px": 0.0}

    c = max(contours, key=cv2.contourArea)
    area = cv2.contourArea(c)
    perimeter = cv2.arcLength(c, closed=True)
    hull = cv2.convexHull(c)
    hull_area = cv2.contourArea(hull)

    compactness = (4.0 * np.pi * area / (perimeter ** 2)) if perimeter > 0 else 0.0
    solidity = (area / hull_area) if hull_area > 0 else 0.0

    return {
        "lesion_area_frac": float(area / area_total),
        "mask_compactness": float(compactness),
        "mask_solidity": float(solidity),
        "gt_perimeter_px": float(perimeter),
    }


def per_sample_image_stats(image_chw: np.ndarray, gt_mask: np.ndarray) -> dict:
    """image_chw: [C,H,W] in the model's normalized range (roughly [-1,1]).
    Returns whole-image brightness and local boundary-ring contrast."""
    img = np.transpose(image_chw, (1, 2, 0))
    gray = img.mean(axis=-1)  # [-1,1]-ish grayscale proxy

    brightness = float(gray.mean())

    # Ring around the boundary: dilate minus erode of the GT mask.
    kernel = np.ones((9, 9), np.uint8)
    dilated = cv2.dilate(gt_mask, kernel, iterations=1)
    eroded = cv2.erode(gt_mask, kernel, iterations=1)
    ring = (dilated - eroded).astype(bool)

    if ring.sum() > 0:
        contrast = float(gray[ring].std())
    else:
        contrast = float(gray.std())

    return {"image_contrast": contrast, "image_brightness": brightness}


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

            records.append({"dice": d, "iou": j, **mask_stats, **img_stats})
            n_seen += 1
            if max_samples is not None and n_seen >= max_samples:
                break

    return records


def report(records):
    dice = np.array([r["dice"] for r in records])
    n = len(records)
    print(f"N = {n} test samples")
    print(f"Dice: mean={dice.mean():.4f}  std={dice.std():.4f}  min={dice.min():.4f}  max={dice.max():.4f}\n")

    factors = ["lesion_area_frac", "mask_compactness", "mask_solidity",
               "gt_perimeter_px", "image_contrast", "image_brightness"]

    results = []
    for name in factors:
        arr = np.array([r[name] for r in records])
        pear_r, pear_p = pearsonr(arr, dice)
        spear_r, spear_p = spearmanr(arr, dice)
        results.append((name, pear_r, pear_p, spear_r, spear_p, arr.mean(), arr.std()))

    # Sort by absolute Spearman rho, strongest candidate first.
    results.sort(key=lambda r: -abs(r[3]))

    print(f"{'factor':>20} | {'mean':>8} | {'std':>8} | {'Pearson r':>10} | {'p':>8} | {'Spearman rho':>13} | {'p':>8}")
    print("-" * 95)
    for name, pr, pp, sr, sp, mean, std in results:
        flag = "  <-- candidate" if abs(sr) > 0.20 and sp < 0.01 else ""
        print(f"{name:>20} | {mean:8.3f} | {std:8.3f} | {pr:+10.3f} | {pp:8.4f} | {sr:+13.3f} | {sp:8.4f}{flag}")

    print(
        "\nInterpretation: rho magnitude roughly < 0.1 = negligible, 0.1-0.3 = weak, "
        "0.3-0.5 = moderate, > 0.5 = strong. Flagged rows (|rho| > 0.20, p < 0.01) "
        "are worth a follow-up worst/best decile breakdown -- for those, rerun with "
        "the strongest factor and inspect a handful of worst-decile images directly "
        "to confirm the correlation reflects a real, fixable pattern rather than a "
        "dataset artifact (e.g. a handful of mislabeled or ambiguous GT masks)."
    )

    strongest = results[0]
    name = strongest[0]
    arr = np.array([r[name] for r in records])
    order = np.argsort(dice)
    worst_n = max(1, n // 10)
    worst_idx = order[:worst_n]
    best_idx = order[-worst_n:]
    print(f"\nStrongest candidate: {name}")
    print(f"  Worst {worst_n} samples (lowest Dice): {name} mean = {arr[worst_idx].mean():.3f}")
    print(f"  Best  {worst_n} samples (highest Dice): {name} mean = {arr[best_idx].mean():.3f}")


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