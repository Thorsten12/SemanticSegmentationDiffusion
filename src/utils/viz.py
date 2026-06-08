"""Qualitative visualization of predictions (saved to disk, no display needed)."""

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def _denorm_img(img_tensor):
    img = img_tensor.detach().cpu().numpy().transpose(1, 2, 0)
    return np.clip(img * 0.5 + 0.5, 0, 1)


def _to_pixels(points, H, W):
    px = (points[:, 0] + 1.0) / 2.0 * (W - 1)
    py = (points[:, 1] + 1.0) / 2.0 * (H - 1)
    return px, py


def save_prediction_grid(images, gt_points, pred_points, gt_masks, pred_masks,
                         out_path, max_samples=4, scores=None):
    """Save a 3-column grid: GT boundary | predicted boundary | predicted mask.

    images     : [B,3,H,W];  gt/pred_points : [B,N,2];
    gt/pred_masks : [B,H,W] or [B,1,H,W];  scores : optional list of per-sample dicts.
    """
    n = min(max_samples, images.shape[0])
    fig, axes = plt.subplots(n, 3, figsize=(15, 5 * n))
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

        axes[i, 0].imshow(img)
        axes[i, 0].plot(np.append(gx, gx[0]), np.append(gy, gy[0]), "g.-", lw=1.5, ms=4)
        axes[i, 0].set_title(f"Sample {i + 1}: GT boundary")
        axes[i, 0].axis("off")

        axes[i, 1].imshow(img)
        axes[i, 1].plot(np.append(gx, gx[0]), np.append(gy, gy[0]), "g-", alpha=0.4, lw=2)
        axes[i, 1].scatter(px, py, c="red", s=25)
        title = "Prediction"
        if scores is not None:
            title += f"  Dice={scores[i]['dice']:.3f} IoU={scores[i]['iou']:.3f}"
        axes[i, 1].set_title(title)
        axes[i, 1].axis("off")

        axes[i, 2].imshow(pmask, cmap="gray")
        axes[i, 2].set_title("Predicted mask (rasterized)")
        axes[i, 2].axis("off")

    plt.tight_layout()
    fig.savefig(out_path, bbox_inches="tight", dpi=150)
    plt.close(fig)
