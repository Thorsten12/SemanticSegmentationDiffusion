"""Diagnostic: contour point spacing vs. feature-grid resolution.

Question this answers: for a typical/ground-truth contour, how far apart
(in image pixels, and in feature-grid cells per scale) are neighboring
points -- and how does that compare to the receptive "footprint" of a single
grid_sample lookup at each scale? If points are already closer together than
one feature-grid cell at the coarsest scale, neither a neighbor-aware sampler
nor a Gaussian blur is likely to add much: adjacent points would already be
reading from the same (or immediately adjacent) grid cell.

Usage (adapt the two loader lines to your actual dataset/config):

    python point_spacing_diagnostic.py --dataset isic2018 --split te

Or import `analyze_dataset(...)` / `analyze_points(...)` directly.
"""

import argparse
import numpy as np


def analyze_points(points_xy, img_size, scale_resolutions):
    """points_xy: [N,2] array in [-1,1] normalized coords (closed contour,
    in point order). img_size: either a single int (square image side length)
    or an (H, W) tuple -- the pixel extent the [-1,1] range maps to.
    scale_resolutions: list of (h, w) feature-grid sizes, finest first.

    Returns a dict of stats: pixel spacing between consecutive points, and
    for each scale, the spacing expressed in units of "feature cells" (i.e.
    spacing / cell_size_in_pixels). A value < 1.0 means neighboring points
    fall inside the same feature cell at that scale; > 1.0 means there is
    a real resolvable gap between what each point's grid_sample call sees.
    """
    if isinstance(img_size, (tuple, list)):
        img_h, img_w = float(img_size[0]), float(img_size[1])
    else:
        img_h = img_w = float(img_size)

    points_xy = np.asarray(points_xy, dtype=np.float64)
    n = points_xy.shape[0]

    # Map [-1,1] normalized coords to pixel coords in [0, img_w]/[0, img_h].
    px = (points_xy + 1.0) * 0.5 * np.array([img_w, img_h])

    next_px = np.roll(px, shift=-1, axis=0)
    seg_len_px = np.linalg.norm(next_px - px, axis=-1)

    stats = {
        "n_points": n,
        "img_size": (img_h, img_w),
        "mean_spacing_px": float(seg_len_px.mean()),
        "median_spacing_px": float(np.median(seg_len_px)),
        "min_spacing_px": float(seg_len_px.min()),
        "max_spacing_px": float(seg_len_px.max()),
        "std_spacing_px": float(seg_len_px.std()),
        "per_scale": [],
    }

    for (h, w) in scale_resolutions:
        # Cell size in pixels for this scale (assume square-ish cells; use
        # the mean of x/y cell size if img is non-square in feature terms).
        cell_px_x = img_w / float(w)
        cell_px_y = img_h / float(h)
        cell_px = 0.5 * (cell_px_x + cell_px_y)

        ratio_mean = stats["mean_spacing_px"] / cell_px
        ratio_median = stats["median_spacing_px"] / cell_px
        ratio_max = stats["max_spacing_px"] / cell_px

        stats["per_scale"].append({
            "resolution": (h, w),
            "cell_size_px": cell_px,
            "mean_spacing_in_cells": ratio_mean,
            "median_spacing_in_cells": ratio_median,
            "max_spacing_in_cells": ratio_max,
        })

    return stats


def analyze_dataset(loader, img_size, scale_resolutions, max_batches=None):
    """loader: yields (images, gt_points, gt_masks) as in your train/sample
    scripts -- gt_points assumed [B,N,2] in [-1,1]. Aggregates per-scale
    spacing-in-cells ratios across the whole (or a subset of the) dataset.
    """
    all_mean_ratios = {tuple(res): [] for res in scale_resolutions}
    all_max_ratios = {tuple(res): [] for res in scale_resolutions}
    n_seen = 0

    for bi, (_, gt_points, _) in enumerate(loader):
        if max_batches is not None and bi >= max_batches:
            break
        pts = gt_points.detach().cpu().numpy()  # [B,N,2]
        for b in range(pts.shape[0]):
            stats = analyze_points(pts[b], img_size, scale_resolutions)
            for s in stats["per_scale"]:
                key = tuple(s["resolution"])
                all_mean_ratios[key].append(s["mean_spacing_in_cells"])
                all_max_ratios[key].append(s["max_spacing_in_cells"])
            n_seen += 1

    if isinstance(img_size, (tuple, list)):
        img_h, img_w = float(img_size[0]), float(img_size[1])
    else:
        img_h = img_w = float(img_size)

    print(f"Analyzed {n_seen} contours.\n")
    print(f"{'scale (h,w)':>14} | {'cell_px':>8} | {'mean spacing (cells)':>22} | {'max spacing (cells)':>20}")
    print("-" * 74)
    for res in scale_resolutions:
        key = tuple(res)
        mean_r = np.mean(all_mean_ratios[key])
        max_r = np.mean(all_max_ratios[key])  # mean of per-contour max, not global max
        cell_px = 0.5 * (img_w / float(res[1]) + img_h / float(res[0]))
        print(f"{str(key):>14} | {cell_px:8.2f} | {mean_r:22.3f} | {max_r:20.3f}")

    print(
        "\nInterpretation: a 'mean spacing (cells)' value < 1.0 at a given scale "
        "means neighboring contour points typically fall within the SAME feature "
        "cell at that resolution -- a single grid_sample call there already "
        "captures the local neighborhood implicitly, and neither multi-point "
        "sampling nor blurring is likely to add information at that scale. "
        "Values > 1.0 mean there is a real, resolvable gap between what "
        "adjacent points' samples see -- that's where a neighbor-aware "
        "sampler or blur could plausibly help. 'max spacing' shows the worst "
        "case (e.g. near sharp corners / low point density regions), which "
        "matters more than the mean if you care about failure cases rather "
        "than the average behavior."
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=str, default="isic2018")
    parser.add_argument("--split", type=str, default="te")
    parser.add_argument("--img_size", type=int, default=256,
                        help="Pixel side length your [-1,1] contour coords map to "
                             "(cfg.img_size in your codebase).")
    parser.add_argument("--n_points", type=int, default=100)
    parser.add_argument("--max_batches", type=int, default=20)
    # Adjust to your actual encoder's feature_channels / pyramid resolutions.
    # These are placeholders -- replace with e.g. encoder.feature_map_sizes
    # or hard-code from your ConvNeXt-Tiny / PVT config.
    parser.add_argument("--scale_resolutions", type=str, default="7x7,14x14,28x28,56x56",
                        help="Comma-separated HxW feature grid sizes, finest LAST "
                             "(matches typical finest-to-coarsest encoder order reversed "
                             "for readability here; adjust to match your actual pyramid).")
    args = parser.parse_args()

    scale_resolutions = []
    for tok in args.scale_resolutions.split(","):
        h, w = tok.lower().split("x")
        scale_resolutions.append((int(h), int(w)))

    from .data import build_contour_dataset
    from torch.utils.data import DataLoader
    from .config import Config
    cfg = Config()
    img_size_tuple = (args.img_size, args.img_size) if isinstance(args.img_size, int) else tuple(args.img_size)
    ds = build_contour_dataset(cfg.skin_root, args.dataset, args.split,
                               args.n_points, img_size_tuple, augment=False)
    loader = DataLoader(ds, batch_size=8, shuffle=False)
    analyze_dataset(loader, img_size_tuple, scale_resolutions,
                    max_batches=args.max_batches)
    # ------------------------------------------------------------------