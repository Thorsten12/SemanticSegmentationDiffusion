"""Qualitative visualization of predictions (saved to disk, no display needed)."""

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import numpy as np


def _denorm_img(img_tensor):
    img = img_tensor.detach().cpu().numpy().transpose(1, 2, 0)
    return np.clip(img * 0.5 + 0.5, 0, 1)


def _to_pixels(points, H, W):
    px = (points[:, 0] + 1.0) / 2.0 * (W - 1)
    py = (points[:, 1] + 1.0) / 2.0 * (H - 1)
    return px, py


def _sigma_to_heatmap(pred_points_np, sigma_np, H, W, blur_radius=7):
    """Rasterize per-point sigma values into a dense HxW heatmap.

    Each contour point votes into its pixel location; a Gaussian blur
    spreads the signal to neighbouring pixels so the result reads as a
    continuous density rather than isolated dots.

    Args:
        pred_points_np : [N, 2]  normalised coords in [-1, 1]
        sigma_np       : [N]     per-point aleatoric uncertainty (exp(log_sigma))
        H, W           : output resolution
        blur_radius    : std-dev of the Gaussian spread in pixels

    Returns:
        heatmap : [H, W]  float32, values in [0, 1]
    """
    from scipy.ndimage import gaussian_filter

    canvas = np.zeros((H, W), dtype=np.float32)
    count  = np.zeros((H, W), dtype=np.float32)

    px = (pred_points_np[:, 0] + 1.0) / 2.0 * (W - 1)
    py = (pred_points_np[:, 1] + 1.0) / 2.0 * (H - 1)

    xi = np.clip(np.round(px).astype(int), 0, W - 1)
    yi = np.clip(np.round(py).astype(int), 0, H - 1)

    for x, y, s in zip(xi, yi, sigma_np):
        canvas[y, x] += s
        count[y, x]  += 1.0

    # Average where multiple points land on the same pixel
    mask = count > 0
    canvas[mask] /= count[mask]

    # Spread to neighbouring pixels via Gaussian blur
    canvas = gaussian_filter(canvas, sigma=blur_radius)

    # Normalise to [0, 1] for display (per-image relative scale)
    vmax = canvas.max()
    if vmax > 1e-8:
        canvas /= vmax

    return canvas


def save_prediction_grid(images, gt_points, pred_points, gt_masks, pred_masks,
                         out_path, max_samples=4, scores=None, log_sigma=None):
    """Save a grid: GT boundary | predicted boundary | predicted mask | uncertainty.

    images        : [B, 3, H, W]
    gt_points     : [B, N, 2]
    pred_points   : [B, N, 2]
    gt_masks      : [B, H, W] or [B, 1, H, W]
    pred_masks    : list of np arrays
    scores        : optional list of per-sample dicts with 'dice' / 'iou'
    log_sigma     : [B, N] tensor or None – per-point log aleatoric uncertainty
                    from the denoiser uncertainty head.  When provided a 4th
                    column is added showing exp(log_sigma) as a colour-coded
                    density map overlaid on the image.  High sigma (red) marks
                    ambiguous boundary segments; low sigma (blue) marks
                    confident regions.
    """
    has_uncertainty = log_sigma is not None
    n_cols = 4 if has_uncertainty else 3
    n = min(max_samples, images.shape[0])

    fig, axes = plt.subplots(n, n_cols, figsize=(5 * n_cols, 5 * n))
    if n == 1:
        axes = axes[np.newaxis, :]

    for i in range(n):
        img = _denorm_img(images[i])
        H, W = img.shape[:2]
        gp = gt_points[i].detach().cpu().numpy()
        pp = pred_points[i].detach().cpu().numpy()
        gx, gy = _to_pixels(gp, H, W)
        px, py = _to_pixels(pp, H, W)
        pmask = pred_masks[i]
        pmask = pmask.squeeze() if hasattr(pmask, "squeeze") else pmask

        # ---- col 0: GT boundary ----
        axes[i, 0].imshow(img)
        axes[i, 0].plot(np.append(gx, gx[0]), np.append(gy, gy[0]),
                        "g.-", lw=1.5, ms=4)
        axes[i, 0].set_title(f"Sample {i + 1}: GT boundary")
        axes[i, 0].axis("off")

        # ---- col 1: predicted boundary ----
        axes[i, 1].imshow(img)
        axes[i, 1].plot(np.append(gx, gx[0]), np.append(gy, gy[0]),
                        "g-", alpha=0.4, lw=2)
        axes[i, 1].scatter(px, py, c="red", s=25)
        title = "Prediction"
        if scores is not None:
            title += f"  Dice={scores[i]['dice']:.3f} IoU={scores[i]['iou']:.3f}"
        axes[i, 1].set_title(title)
        axes[i, 1].axis("off")

        # ---- col 2: rasterized mask ----
        axes[i, 2].imshow(pmask, cmap="gray")
        axes[i, 2].set_title("Predicted mask (rasterized)")
        axes[i, 2].axis("off")

        # ---- col 3: uncertainty heatmap (optional) ----
        if has_uncertainty:
            sigma_i = log_sigma[i].detach().cpu().numpy()  # [N]
            sigma_i = np.exp(sigma_i)                       # log → linear scale

            heatmap = _sigma_to_heatmap(pp, sigma_i, H, W, blur_radius=7)

            axes[i, 3].imshow(img)
            im = axes[i, 3].imshow(
                heatmap,
                cmap="RdYlBu_r",   # blue=confident, red=uncertain
                alpha=0.55,
                vmin=0.0, vmax=1.0,
            )
            # Scatter the contour points coloured by their raw sigma
            sc = axes[i, 3].scatter(
                px, py,
                c=sigma_i,
                cmap="RdYlBu_r",
                s=18,
                vmin=sigma_i.min(),
                vmax=sigma_i.max(),
                edgecolors="none",
            )
            plt.colorbar(sc, ax=axes[i, 3], fraction=0.03, pad=0.02,
                         label="σ (aleatoric)")
            mean_s = sigma_i.mean()
            axes[i, 3].set_title(f"Uncertainty  σ̄={mean_s:.3f}")
            axes[i, 3].axis("off")

    plt.tight_layout()
    fig.savefig(out_path, bbox_inches="tight", dpi=150)
    plt.close(fig)