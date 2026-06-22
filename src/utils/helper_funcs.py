import torch
import torchvision.transforms.functional as F_vision

def enhance_frequencies(
    images: torch.Tensor,
    mid_gain: float = 1.5,
    edge_gain: float = 1.2,
) -> torch.Tensor:
    """Boost mid-range textures and high-freq edges independently.

    Mid-band:  textures, color gradients, pigment network
    High-band: lesion border sharpness (ABCD border criterion)
    """
    smooth_fine   = F_vision.gaussian_blur(images, kernel_size=11, sigma=2.5)
    smooth_coarse = F_vision.gaussian_blur(images, kernel_size=21, sigma=5.0)

    mid_band  = smooth_fine - smooth_coarse   # mid frequencies
    high_band = images - smooth_fine          # high frequencies (edges)

    enhanced = images \
        + (mid_gain  - 1.0) * mid_band \
        + (edge_gain - 1.0) * high_band

    # Soft rescale per image
    B = enhanced.shape[0]
    flat = enhanced.view(B, -1)
    lo = flat.min(dim=1).values.view(B, 1, 1, 1)
    hi = flat.max(dim=1).values.view(B, 1, 1, 1)
    return 2.0 * (enhanced - lo) / (hi - lo + 1e-8) - 1.0

import torch

def calc_boundary_att(x, t, T, gamma=1.5):
    """
    Computes time-gated boundary attention weights for an ordered set of contour points.
    
    Args:
        x (torch.Tensor): Ground-truth contour coordinates of shape [B, N, 2].
        t (torch.Tensor): Current diffusion timesteps of shape [B].
        T (int): Total number of diffusion timesteps (e.g., 1000).
        gamma (float): Controls the steepness of the temporal decay.
        
    Returns:
        torch.Tensor: Attention weights of shape [B, N, 1], matching the point dimensions.
    """
    # 1. Spatial Component: Compute local curvature along the closed polygon loop
    # Shift points forward and backward to get immediate neighbors
    prev_x = torch.roll(x, shifts=1, dims=1)
    next_x = torch.roll(x, shifts=-1, dims=1)
    
    # The discrete second derivative (Laplacian) represents the local curvature vector
    # Sharp corners/curves yield a high magnitude, straight lines yield near zero
    curvature = torch.norm(next_x + prev_x - 2 * x, dim=-1, keepdim=True)  # [B, N, 1]
    
    # Normalize curvature per batch sample to keep the gradient scale stable
    max_curve = curvature.max(dim=1, keepdim=True)[0].clamp(min=1e-8)
    spatial_att = curvature / max_curve
    
    # 2. Temporal Component: Gate the attention based on the remaining diffusion time
    # Reshape t from [B] to [B, 1, 1] for element-wise broadcasting
    t_scaled = t.float().view(-1, 1, 1) / T
    
    # As t approaches 0 (late sampling/fine-tuning), time_weight approaches 1.0
    time_weight = (1.0 - t_scaled).clamp(min=0.0) ** gamma
    
    # 3. Combine components
    # We add a baseline of 1.0 so that flat segments still receive a basic gradient,
    # while geometric keypoints are heavily penalized in the final training phases.
    boundary_att = 1.0 + spatial_att * time_weight
    
    return boundary_att