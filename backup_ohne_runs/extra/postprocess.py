"""Post-processing utilities for P2SDiff predictions.

Three independent, composable pieces:

1. combine_masks   -- merge N binary masks (from different seeds/configs/TTA
                       views) into one, via majority vote or soft averaging.
2. keep_largest_component -- morphological cleanup: drop small disconnected
                       blobs, keeping only the largest connected component
                       (optionally within a relative-area tolerance so two
                       genuinely close-in-size lobes aren't wrongly merged
                       away -- see `keep_ratio`).
3. smooth_contour_points -- circular Savitzky-Golay smoothing applied to the
                       ordered [N,2] boundary points *before* rasterization,
                       matching the model's circular/closed-contour
                       parametrization.

All functions operate on plain numpy arrays so they can be dropped into
sample.py's `evaluate()` or a standalone ensembling script without any torch
dependency.
"""

import numpy as np
import cv2
from scipy.signal import savgol_filter


# ---------------------------------------------------------------------------
# 1. Mask combination (ensembling across seeds / configs / TTA views)
# ---------------------------------------------------------------------------

def combine_masks(mask_list, mode="majority", threshold=0.5):
    """Combine a list of binary {0,1} masks (same H,W) into one.

    mode="majority" (default): pixel-wise fraction of masks voting "1" must
        be >= threshold (0.5 = strict majority; use 0.5 with an even number
        of masks to mean ">=50%", i.e. ties count as foreground -- pass a
        value like 0.5+eps if you want ties to go the other way).
    mode="soft": same computation, but returns the continuous [0,1] coverage
        map instead of thresholding it -- useful if a caller wants to defer
        thresholding (e.g. to combine with another soft signal first).

    Only binary-mask majority voting is supported here, not true
    probability averaging, since points_to_mask() only ever produces hard
    0/1 masks in this pipeline (no per-pixel model confidence is available
    downstream of the point predictions).
    """
    if len(mask_list) == 0:
        raise ValueError("combine_masks: empty mask_list")
    stack = np.stack([np.asarray(m).astype(np.float32) for m in mask_list], axis=0)
    coverage = stack.mean(axis=0)  # fraction of masks voting foreground, per pixel
    if mode == "soft":
        return coverage
    if mode == "majority":
        return (coverage >= threshold).astype(np.uint8)
    raise ValueError(f"combine_masks: unknown mode '{mode}'")


# ---------------------------------------------------------------------------
# 2. Connected-component cleanup
# ---------------------------------------------------------------------------

def keep_largest_component(mask, keep_ratio=0.0):
    """Zero out every connected component except the largest.

    mask: binary {0,1} uint8/bool array, single channel [H, W].
    keep_ratio: if > 0, additionally keep any OTHER component whose area is
        at least `keep_ratio` * (area of the largest component). Default 0.0
        reproduces the simple "keep only the single largest blob" behavior,
        appropriate for ISIC-style single-lesion images where genuine
        multi-blob ground truth is essentially never the case. Raise this
        (e.g. 0.3) only if your dataset can have two genuinely comparable
        lesions in one image.

    If the mask is empty (no foreground), it is returned unchanged.
    """
    mask = np.asarray(mask).astype(np.uint8)
    if mask.sum() == 0:
        return mask

    n_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    if n_labels <= 2:  # background (0) + at most one foreground component
        return mask

    areas = stats[1:, cv2.CC_STAT_AREA]  # skip background label 0
    largest_area = areas.max()
    if keep_ratio <= 0.0:
        # Strict mode: only the single largest component survives.
        keep_labels = {1 + int(np.argmax(areas))}
    else:
        min_area = keep_ratio * largest_area
        keep_labels = {1 + int(i) for i, a in enumerate(areas) if a >= min_area}

    out = np.zeros_like(mask)
    for lbl in keep_labels:
        out[labels == lbl] = 1
    return out


# ---------------------------------------------------------------------------
# 3. Circular contour smoothing (on the ordered boundary points, pre-raster)
# ---------------------------------------------------------------------------

def smooth_contour_points(points, window_length=9, polyorder=2):
    """Savitzky-Golay smoothing of a closed, ordered contour.

    points: [N, 2] array (x, y), ordered around the boundary (as produced by
        the dataset's uniform/curvature-adaptive sampling -- same ordering
        convention the model is trained to predict).
    window_length: must be odd and <= N; reduced automatically if N is too
        small (falls back to no smoothing if N < 5).
    polyorder: polynomial order fit within each window; must be < window_length.

    Wraps the signal periodically (mode="wrap") so the seam between the last
    and first point is smoothed consistently with every other point on the
    contour, matching the circular/padding_mode="circular" convention used
    elsewhere in this pipeline.
    """
    points = np.asarray(points, dtype=np.float32)
    n = points.shape[0]

    wl = min(window_length, n if n % 2 == 1 else n - 1)
    if wl < 5:
        return points  # too few points to smooth meaningfully
    if wl <= polyorder:
        polyorder = wl - 1 if wl > 1 else 1

    x_smooth = savgol_filter(points[:, 0], window_length=wl, polyorder=polyorder, mode="wrap")
    y_smooth = savgol_filter(points[:, 1], window_length=wl, polyorder=polyorder, mode="wrap")
    return np.stack([x_smooth, y_smooth], axis=1).astype(np.float32)