"""Visual inspection panels for outlier samples.

For each requested sample index, renders a 4-panel figure:
    1. Original image (as seen by the model)
    2. Ground-truth mask overlaid on the image (green)
    3. Majority-vote prediction overlaid on the image (red)
    4. Difference map: true positive (gray), false positive (red),
       false negative (blue) -- so you can see AT A GLANCE whether the
       model over-segments, under-segments, or is offset/rotated.

Also writes one combined contact sheet (all samples in a grid) for quick
scrolling, plus the individual full-resolution PNGs for close inspection.

This re-collects per-checkpoint masks itself (same cross_run_ensemble
checkpoint set as analyze_agreement.py) so it can be run standalone -- pass
--indices to target specific samples (e.g. the ones printed by
analyze_agreement.py), or --top_outliers N to auto-select the N worst
majority-vote Dice samples.

Usage (target specific known outliers):
    python -m src.visualize_outliers \
        --runs_root x0_prediction_backup/runs/ablation_v15a_local_drop \
        --skin_root data/datasets \
        --dataset isic2018 --split test \
        --indices 393 370 506 141 37 213 445 485 417 413 465 482 430 171 433 3 84 477 219 323 394 348 \
        --out_dir outlier_panels

Usage (auto-select worst N by majority-vote Dice):
    python -m src.visualize_outliers \
        --runs_root x0_prediction_backup/runs/ablation_v15a_local_drop \
        --skin_root data/datasets \
        --dataset isic2018 --split test \
        --top_outliers 22 --out_dir outlier_panels
"""

import argparse
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.utils.data import DataLoader

from .config import Config
from .data import build_contour_dataset
from .diffusion import GaussianDiffusion
from .sample import load_checkpoint
from .utils import dice_score, points_to_mask
from .postprocess import combine_masks
from .ensemble_predict import find_checkpoints, _predict_points_single_checkpoint

CROSS_RUN_EXPERIMENTS = ["drop_1", "drop_2", "drop_2_topk_none"]


# ---------------------------------------------------------------------------
# Prediction collection (only for the requested indices, to keep this fast --
# still has to run every checkpoint over the full loader since sampling is
# batched, but skips building visualizations for anything not requested)
# ---------------------------------------------------------------------------

def collect_masks_for_indices(ckpt_map_cross, cfg, device, loader, wanted_indices):
    diffusion = GaussianDiffusion(cfg.timesteps, cfg.beta_start, cfg.beta_end, device=device)
    cross_ckpts = [p for paths in ckpt_map_cross.values() for p in paths]

    wanted = set(wanted_indices)
    n_images = len(loader.dataset)
    masks_per_ckpt = {p: {} for p in cross_ckpts}   # ckpt -> {idx: mask}
    gt_masks = {}
    images_cache = {}

    for ckpt_path in cross_ckpts:
        print(f"  predicting with {ckpt_path} ...")
        encoder, denoiser, snapper, proposal_head = load_checkpoint(ckpt_path, cfg, device)

        idx = 0
        for images, gt_points, gt_batch in loader:
            b = images.shape[0]
            batch_indices = list(range(idx, idx + b))
            if wanted.isdisjoint(batch_indices):
                idx += b
                continue  # skip sampling cost for batches with nothing we need
                # NOTE: still runs full-batch DDIM if ANY sample in the batch
                # is wanted (sampling is inherently batched) -- only whole
                # batches with zero overlap are skipped.

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
                if gi not in images_cache:
                    images_cache[gi] = images[i].cpu()
                masks_per_ckpt[ckpt_path][gi] = points_to_mask(point_sets[0][i], cfg.img_size)
            idx += b

        del encoder, denoiser, snapper, proposal_head
        if device.type == "cuda":
            torch.cuda.empty_cache()

    return masks_per_ckpt, gt_masks, images_cache, cross_ckpts


# ---------------------------------------------------------------------------
# Rendering
# ---------------------------------------------------------------------------

def _to_display_image(img_chw):
    img = img_chw.numpy()
    if img.ndim == 3:
        img = np.transpose(img, (1, 2, 0))
    img = img.astype(np.float32)
    img = (img - img.min()) / (img.max() - img.min() + 1e-8)
    return img


def render_panel(idx, image_chw, gt_mask, pred_mask, dice, save_path):
    """4-panel figure for one sample; saved to save_path."""
    img = _to_display_image(image_chw)
    gt = np.asarray(gt_mask).astype(bool)
    pred = np.asarray(pred_mask).astype(bool)

    tp = gt & pred
    fp = pred & ~gt
    fn = gt & ~pred

    diff = np.zeros((*gt.shape, 3), dtype=np.float32)
    diff[tp] = [0.6, 0.6, 0.6]   # gray: correct
    diff[fp] = [1.0, 0.2, 0.2]   # red: false positive (over-segmentation)
    diff[fn] = [0.2, 0.4, 1.0]   # blue: false negative (under-segmentation)

    fig, axes = plt.subplots(1, 4, figsize=(16, 4.2), dpi=110)

    axes[0].imshow(img)
    axes[0].set_title("image")
    axes[0].axis("off")

    axes[1].imshow(img)
    axes[1].contour(gt.astype(float), levels=[0.5], colors="lime", linewidths=1.5)
    axes[1].set_title("GT boundary")
    axes[1].axis("off")

    axes[2].imshow(img)
    axes[2].contour(pred.astype(float), levels=[0.5], colors="red", linewidths=1.5)
    axes[2].set_title("majority-vote prediction")
    axes[2].axis("off")

    axes[3].imshow(diff)
    axes[3].set_title("gray=correct  red=FP  blue=FN")
    axes[3].axis("off")

    fig.suptitle(f"idx {idx}  |  majority-vote Dice = {dice:.3f}", fontsize=12)
    fig.tight_layout(rect=[0, 0, 1, 0.93])
    fig.savefig(save_path)
    plt.close(fig)


def render_contact_sheet(panel_paths, save_path, cols=3):
    """Stitches the saved panel PNGs into one big scrollable-by-eye sheet."""
    import imageio.v2 as imageio
    imgs = [imageio.imread(p) for p in panel_paths]
    rows = int(np.ceil(len(imgs) / cols))
    h, w = imgs[0].shape[:2]
    sheet = np.full((rows * h, cols * w, 3), 255, dtype=np.uint8)
    for i, im in enumerate(imgs):
        r, c = divmod(i, cols)
        im3 = im[:, :, :3] if im.shape[-1] == 4 else im
        sheet[r * h:(r + 1) * h, c * w:(c + 1) * w] = im3
    imageio.imwrite(save_path, sheet)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Visual inspection panels for outlier samples")
    parser.add_argument("--runs_root", type=str, required=True)
    parser.add_argument("--skin_root", type=str, required=True)
    parser.add_argument("--dataset", choices=["ph2", "isic2017", "isic2018", "ham10000"], required=True)
    parser.add_argument("--split", choices=["val", "test"], default="test")
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--out_dir", type=str, default="outlier_panels")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--indices", type=int, nargs="+", help="Specific sample indices to render.")
    group.add_argument("--top_outliers", type=int,
                       help="Auto-select the N worst majority-vote-Dice samples "
                            "(requires one extra pass to score every sample first).")
    args = parser.parse_args()

    cfg = Config.from_args(args)
    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))

    split = "vl" if args.split == "val" else "te"
    ds = build_contour_dataset(cfg.skin_root, args.dataset, split, cfg.n_points,
                               cfg.img_size, augment=False, npy_size=cfg.npy_size)
    loader = DataLoader(ds, batch_size=cfg.batch_size, shuffle=False)

    ckpt_map_cross = find_checkpoints(args.runs_root, CROSS_RUN_EXPERIMENTS)

    if args.top_outliers is not None:
        print(f"Scoring all {len(ds)} samples to find the {args.top_outliers} worst "
             f"majority-vote-Dice cases (one full pass over all checkpoints)...")
        all_indices = list(range(len(ds)))
        masks_per_ckpt, gt_masks, images_cache, ckpts = collect_masks_for_indices(
            ckpt_map_cross, cfg, device, loader, all_indices)
        dice_by_idx = {}
        for i in all_indices:
            mv = combine_masks([masks_per_ckpt[p][i] for p in ckpts], mode="majority")
            dice_by_idx[i] = dice_score(mv, gt_masks[i])
        wanted_indices = sorted(dice_by_idx, key=dice_by_idx.get)[:args.top_outliers]
        print(f"Selected indices: {wanted_indices}")
    else:
        wanted_indices = args.indices
        print(f"Collecting predictions for {len(wanted_indices)} requested indices...")
        masks_per_ckpt, gt_masks, images_cache, ckpts = collect_masks_for_indices(
            ckpt_map_cross, cfg, device, loader, wanted_indices)

    os.makedirs(args.out_dir, exist_ok=True)
    panel_paths = []
    for idx in wanted_indices:
        mask_list = [masks_per_ckpt[p][idx] for p in ckpts]
        mv_mask = combine_masks(mask_list, mode="majority")
        dice = dice_score(mv_mask, gt_masks[idx])

        save_path = os.path.join(args.out_dir, f"sample_{idx}_dice{dice:.3f}.png")
        render_panel(idx, images_cache[idx], gt_masks[idx], mv_mask, dice, save_path)
        panel_paths.append(save_path)
        print(f"  saved {save_path}")

    contact_sheet_path = os.path.join(args.out_dir, "_contact_sheet.png")
    render_contact_sheet(panel_paths, contact_sheet_path)
    print(f"\nContact sheet (all samples at a glance): {contact_sheet_path}")
    print(f"Individual panels in: {args.out_dir}/")


if __name__ == "__main__":
    main()