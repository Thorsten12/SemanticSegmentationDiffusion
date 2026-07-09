import torch
import torch.nn.functional as F
import math

def calc_curvature(points):
    """
    Computes normalized load curvature along an ordered set of points. 

    Returns for every Point its curvature.

    Args:
        points (torch: Tensor): Contour points [B, N, 2]

    Returns:
        torch.Tensor :Curvature per points [B, N, 1]
    """

    prev_x = torch.roll(points, shifts=1, dims=1)
    next_x = torch.roll(points, shifts=1, dims=1)


    # Curvature := Difference bettween Vektor( Previuos Point to Current Point ) to Vektor (Current Point to Next Point)
    curvature = torch.norm(next_x + prev_x - 2 * points, dim=-1, keepdim=True) # [B,N,1]
    max_curve = curvature.max(dim=1, keepdim=True)[0].clamp(min=1e-8)

    return curvature / max_curve

def calc_boundary_att(points, t, T:int = 1000, gamma=1.5):
    """
    Computes the special Attention on boundary using the curvature
    """

    spatial_att = calc_curvature(points)

    t_scaled = t.float().view(-1, 1, 1) / T
    time_weight = (1.0 - t_scaled).clamp(min=0.0) ** gamma


    boundary_att = 1.0 + spatial_att * time_weight

    return boundary_att


def soft_rasterize(points, size=64, eps=1e-7):
    """
    Differentiable polygon fill via the winding number.
    """
    B, N, _ = points.shape
    device, dtype = points.device, points.dtype
    
    # Gitter in der reduzierten Auflösung erstellen
    ys = torch.linspace(-1.0, 1.0, size, device=device, dtype=dtype)
    xs = torch.linspace(-1.0, 1.0, size, device=device, dtype=dtype)
    gy, gx = torch.meshgrid(ys, xs, indexing="ij")
    grid = torch.stack([gx, gy], dim=-1).reshape(1, size * size, 1, 2)  # [1, P, 1, 2]

    v = points.unsqueeze(1)                       # [B, 1, N, 2]
    vn = torch.roll(v, shifts=-1, dims=2)         # [B, 1, N, 2]

    a = v - grid                                  # [B, P, N, 2]
    b = vn - grid                                 # [B, P, N, 2]

    # Kreuz- und Skalarprodukt für den Winkel
    cross = a[..., 0] * b[..., 1] - a[..., 1] * b[..., 0]
    dot = a[..., 0] * b[..., 0] + a[..., 1] * b[..., 1] + eps
    
    ang = torch.atan2(cross, dot)                 # [B, P, N]
    
    # Winding Number: Summe der Winkel geteilt durch 2*pi
    winding = ang.sum(dim=-1) / (2.0 * math.pi)   # [B, P]
    
    return winding.abs().clamp(0.0, 1.0).reshape(B, size, size)


def soft_dice_loss(pred_points, gt_masks, size=64, eps=1e-6):
    """
    Berechnet den Soft-Dice-Loss auf einer reduzierten Auflösung für maximale Performance.
    """
    # 1. Punkte in FP32 konvertieren für numerische Stabilität bei atan2
    soft_mask = soft_rasterize(pred_points.float(), size=size)  # [B, size, size]
    
    # 2. GT-Maske auf die gleiche reduzierte Auflösung skalieren
    if gt_masks.dim() == 3:
        gt_masks = gt_masks.unsqueeze(1)  # [B, 1, H, W]
        
    gt_resized = F.interpolate(gt_masks.float(), size=(size, size), mode="area")
    gt_resized = (gt_resized.squeeze(1) > 0.5).float()          # [B, size, size]
    
    # 3. Standard Dice-Berechnung
    intersection = (soft_mask * gt_resized).sum(dim=(1, 2))
    union = soft_mask.sum(dim=(1, 2)) + gt_resized.sum(dim=(1, 2))
    
    dice = (2.0 * intersection + eps) / (union + eps)
    
    return (1.0 - dice).mean()
