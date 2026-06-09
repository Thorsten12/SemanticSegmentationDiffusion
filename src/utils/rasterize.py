"""Turn predicted boundary points into a binary mask and score it.

This closes the loop of the method: diffusion produces an ordered set of boundary
points in [-1, 1]; we map them to pixel coordinates and fill the polygon to get a
0/1 segmentation, then compare against the ground-truth mask with Dice / IoU.
"""

import math

import cv2
import numpy as np
import torch
import torch.nn.functional as F


def soft_rasterize(points, size=64, eps=1e-7):
    """Differentiable polygon fill via the winding number.

    For every pixel we sum the signed angle subtended by each polygon edge; the
    winding number is ~+/-1 for pixels inside the closed contour and ~0 outside,
    with a smooth transition across the boundary. Unlike `cv2.fillPoly` this is
    differentiable w.r.t. the vertex coordinates, so a mask-level (soft-Dice)
    loss can push the boundary points to match the target shape.

    points : tensor [B, N, 2] in [-1, 1], (x, y) order, ordered around the contour.
    returns: tensor [B, size, size] in [0, 1] (soft occupancy).
    """
    B, N, _ = points.shape
    device, dtype = points.device, points.dtype
    ys = torch.linspace(-1.0, 1.0, size, device=device, dtype=dtype)
    xs = torch.linspace(-1.0, 1.0, size, device=device, dtype=dtype)
    gy, gx = torch.meshgrid(ys, xs, indexing="ij")
    grid = torch.stack([gx, gy], dim=-1).reshape(1, size * size, 1, 2)  # [1,P,1,2]

    v = points.unsqueeze(1)                       # [B,1,N,2]
    vn = torch.roll(v, shifts=-1, dims=2)         # next vertex (contour wraps)
    a = v - grid                                  # [B,P,N,2]  pixel -> vertex_i
    b = vn - grid                                 # [B,P,N,2]  pixel -> vertex_{i+1}
    cross = a[..., 0] * b[..., 1] - a[..., 1] * b[..., 0]
    dot = a[..., 0] * b[..., 0] + a[..., 1] * b[..., 1] + eps  # eps avoids atan2(0,0)
    ang = torch.atan2(cross, dot)                 # [B,P,N] signed subtended angle
    winding = ang.sum(dim=-1) / (2.0 * math.pi)   # [B,P] ~+/-1 inside, ~0 outside
    return winding.abs().clamp(0.0, 1.0).reshape(B, size, size)


def soft_dice_loss(pred_points, gt_masks, size=64, eps=1e-6):
    """1 - soft-Dice between the rasterized predicted polygon and the GT mask.

    pred_points : [B, N, 2] in [-1, 1].  gt_masks : [B, 1, H, W] in {0, 1}.
    Computed in fp32 (atan2 is numerically touchy under autocast).
    """
    soft = soft_rasterize(pred_points.float(), size)                       # [B,size,size]
    gt = F.interpolate(gt_masks.float(), size=(size, size), mode="area")
    gt = (gt.squeeze(1) > 0.5).float()                                     # [B,size,size]
    inter = (soft * gt).sum(dim=(1, 2))
    denom = soft.sum(dim=(1, 2)) + gt.sum(dim=(1, 2))
    dice = (2.0 * inter + eps) / (denom + eps)
    return (1.0 - dice).mean()


def points_to_mask(points, img_size):
    """Fill the ordered polygon defined by `points` ([-1,1]) into a binary mask.

    points  : array/tensor [N, 2] in [-1, 1], (x, y) order.
    img_size: (H, W).
    returns : uint8 array [H, W] in {0, 1}.
    """
    if isinstance(points, torch.Tensor):
        points = points.detach().cpu().numpy()
    H, W = img_size
    px = (points[:, 0] + 1.0) / 2.0 * (W - 1)
    py = (points[:, 1] + 1.0) / 2.0 * (H - 1)
    poly = np.stack([px, py], axis=1).round().astype(np.int32)

    mask = np.zeros((H, W), dtype=np.uint8)
    cv2.fillPoly(mask, [poly], 1)
    return mask


def dice_score(pred, gt, eps=1e-6):
    """Dice over two binary {0,1} arrays."""
    pred = pred.astype(bool)
    gt = gt.astype(bool)
    inter = np.logical_and(pred, gt).sum()
    return float((2 * inter + eps) / (pred.sum() + gt.sum() + eps))


def iou_score(pred, gt, eps=1e-6):
    """Intersection-over-union over two binary {0,1} arrays."""
    pred = pred.astype(bool)
    gt = gt.astype(bool)
    inter = np.logical_and(pred, gt).sum()
    union = np.logical_or(pred, gt).sum()
    return float((inter + eps) / (union + eps))
