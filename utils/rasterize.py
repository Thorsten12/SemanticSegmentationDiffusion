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


def soft_dice_loss(pred_points, gt_masks, size=64, eps=1e-6, t=None, T=None, gamma=1.0):
    """1 - soft-Dice between the rasterized predicted polygon and the GT mask.

    pred_points : [B, N, 2] in [-1, 1].  gt_masks : [B, 1, H, W] in {0, 1}.
    Computed in fp32 (atan2 is numerically touchy under autocast).

    Optional time-weighting (both t and T must be given together to enable
    it; otherwise this is a plain per-sample-averaged soft-Dice loss, as
    before -- this is the mode used wherever no timestep is available, e.g.
    the standalone `proposal_dice_loss` in proposal.py):
        t: [B] integer diffusion timesteps.
        T: total diffusion timesteps (`self.timesteps`).
        gamma: exponent controlling how sharply the weight decays.
    Weighting direction is the OPPOSITE of `soft_boundary_iou_loss`: overall
    shape/coverage (soft-Dice's job) matters most when the sample is still
    noisy (high t, close to pure noise, where only the coarse silhouette is
    meaningful) and fades out at low t (clean, where boundary precision --
    `soft_boundary_iou_loss`'s job -- matters more). Weight is
    `w_t = (t / (T-1))**gamma`, i.e. the mirror image of
    `soft_boundary_iou_loss`'s `(1 - t/(T-1))**gamma`.
    """
    soft = soft_rasterize(pred_points.float(), size)                       # [B,size,size]
    gt = F.interpolate(gt_masks.float(), size=(size, size), mode="area")
    gt = (gt.squeeze(1) > 0.5).float()                                     # [B,size,size]
    inter = (soft * gt).sum(dim=(1, 2))
    denom = soft.sum(dim=(1, 2)) + gt.sum(dim=(1, 2))
    dice = (2.0 * inter + eps) / (denom + eps)
    per_sample_loss = 1.0 - dice                                           # [B]

    if t is not None and T is not None:
        w_t = (t.float() / (T - 1)).clamp(min=0.0, max=1.0) ** gamma       # [B]
        return (w_t * per_sample_loss).mean()
    return per_sample_loss.mean()


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

def _boundary_band(mask, dilation=3):
    """Erode+dilate a hard {0,1} mask to extract a thin boundary band.

    mask: [B, H, W] float in {0,1} (or soft values, thresholded internally
        for the erosion/dilation structuring so the band itself stays a
        clean ring regardless of how "soft" the input was).
    dilation: half-width (in pixels, at the mask's own resolution) of the
        band around the boundary. Implemented via max-pooling for the
        dilation and min-pooling (via -maxpool(-x)) for the erosion, so
        this stays fully differentiable end-to-end (grad flows through
        wherever `mask` itself is differentiable, e.g. the soft-rasterized
        polygon) -- unlike a cv2/skimage morphology call.

    Returns: [B, H, W] float band in {0,1}-ish (soft edges possible if the
        input mask itself was soft) that is 1 near the boundary of `mask`
        and 0 both deep inside and deep outside.
    """
    k = 2 * dilation + 1
    m = mask.unsqueeze(1)  # [B,1,H,W]
    dilated = F.max_pool2d(m, kernel_size=k, stride=1, padding=dilation)
    eroded = -F.max_pool2d(-m, kernel_size=k, stride=1, padding=dilation)
    band = (dilated - eroded).clamp(0.0, 1.0)
    return band.squeeze(1)  # [B,H,W]


def soft_boundary_iou_loss(pred_points, gt_masks, size=64, eps=1e-6,
                           dilation=3, t=None, T=None, gamma=1.0):
    """1 - Boundary IoU between the rasterized predicted polygon and GT mask.

    Standard IoU scores the whole filled region, so a model can get a high
    score while still getting the exact boundary location wrong (a big
    shape mostly overlaps regardless of small edge errors). Boundary IoU
    (Cheng et al. 2021 / used here per Sun et al., MICCAI 2023) restricts
    the comparison to a thin band around the boundary of each mask, so it
    penalizes boundary localization error specifically -- exactly the
    signal this project's point-based contour prediction needs on top of
    the coarser `soft_dice_loss`.

    pred_points: [B, N, 2] in [-1, 1], ordered closed contour.
    gt_masks: [B, 1, H, W] in {0, 1}.
    size: rasterization resolution (same convention as `soft_dice_loss`).
    dilation: half-width in pixels (at `size` resolution) of the boundary
        band extracted from both masks before computing IoU.

    Optional time-weighting (both t and T must be given together to enable
    it; if either is None, the loss is returned unweighted, i.e. a plain
    per-sample-averaged Boundary IoU loss -- this is the mode `snapper.py`
    uses, since the snapper's teacher loss isn't part of the diffusion
    timestep chain):
        t: [B] integer diffusion timesteps.
        T: total diffusion timesteps (`self.timesteps`).
        gamma: exponent controlling how sharply the weight decays with t.
    Weighting direction is the OPPOSITE of `soft_dice_loss`'s time
    weighting: boundary precision matters most when the sample is nearly
    clean (low t, close to t=0) and fades out at high t (pure noise, where
    only the coarse shape -- soft-Dice's job -- is meaningful). Weight is
    `w_t = (1 - t/(T-1))**gamma`, matching `nearest_point_loss`'s schedule
    in diffusion.py.
    """
    soft = soft_rasterize(pred_points.float(), size)                       # [B,size,size]
    gt = F.interpolate(gt_masks.float(), size=(size, size), mode="area")
    gt = (gt.squeeze(1) > 0.5).float()                                     # [B,size,size]

    pred_band = _boundary_band(soft, dilation=dilation)
    gt_band = _boundary_band(gt, dilation=dilation)

    inter = (pred_band * gt_band).sum(dim=(1, 2))
    union = (pred_band + gt_band - pred_band * gt_band).sum(dim=(1, 2))
    biou = (inter + eps) / (union + eps)                                   # [B]
    per_sample_loss = 1.0 - biou                                           # [B]

    if t is not None and T is not None:
        w_t = (1.0 - t.float() / (T - 1)).clamp(min=0.0, max=1.0) ** gamma  # [B]
        return (w_t * per_sample_loss).mean()
    return per_sample_loss.mean()