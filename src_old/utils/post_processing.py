import torch

def remove_contour_outliers(points, threshold_sigma=3.0):
    """Detects and interpolates outliers in a closed contour [B, N, 2].
    
    Exploits the uniformity constraint: if a point is an outlier, its distance
    to BOTH its left and right neighbor will be statistically anomalous.
    """
    B, N, _ = points.shape
    # Clone to avoid modifying the original tensor in-place
    cleaned_points = points.clone()
    
    # 1. Get immediate neighbors (circular padding for closed loop)
    prev_pts = torch.roll(points, shifts=1, dims=1)
    next_pts = torch.roll(points, shifts=-1, dims=1)
    
    # 2. Compute distances to left and right neighbors
    dist_left = torch.norm(points - prev_pts, dim=-1)   # [B, N]
    dist_right = torch.norm(next_pts - points, dim=-1)  # [B, N]
    
    # A point is an anomaly if it is far away from BOTH neighbors
    # We take the minimum of both jumps to ensure the point itself is the issue
    point_anomaly_metric = torch.minimum(dist_left, dist_right) # [B, N]
    
    # 3. Calculate mean and std per sample in the batch
    mean_dist = point_anomaly_metric.mean(dim=1, keepdim=True)  # [B, 1]
    std_dist = point_anomaly_metric.std(dim=1, keepdim=True)    # [B, 1]
    
    # 4. Define the outlier mask (where distance exceeds mean + 3 * std)
    threshold = mean_dist + threshold_sigma * std_dist
    outlier_mask = point_anomaly_metric > threshold # [B, N] boolean tensor
    
    # 5. Linear interpolation/Spline-approximation for the outliers:
    # Instead of the broken x_i, we place it exactly between x_{i-1} and x_{i+1}
    interpolated_points = (prev_pts + next_pts) * 0.5
    
    # Apply the correction where mask is True
    # outlier_mask.unsqueeze(-1) expands [B, N] to [B, N, 1] for element-wise broadcasting
    cleaned_points = torch.where(outlier_mask.unsqueeze(-1), interpolated_points, cleaned_points)
    
    return cleaned_points

import torch
import torch.nn.functional as F

def smooth_closed_contour(points: torch.Tensor, kernel_size: int = 5, sigma: float = 1.0) -> torch.Tensor:
    """Smoothes a closed 2D contour tensor of shape [B, N, 2] using a 1D Gaussian kernel
    with circular padding to preserve the closed-loop topology.
    """
    if kernel_size <= 1:
        return points

    device = points.device
    B, N, C = points.shape

    # 1. Create a 1D Gaussian kernel
    x = torch.arange(-(kernel_size // 2), kernel_size // 2 + 1, device=device).float()
    kernel = torch.exp(-x**2 / (2 * sigma**2))
    kernel = kernel / kernel.sum()  # Normalize
    
    # Reshape kernel for Conv1d groups: [channels, in_channels_per_group, kernel_width] -> [2, 1, K]
    kernel = kernel.view(1, 1, -1).repeat(2, 1, 1)
    
    # 2. Reshape points from [B, N, 2] to [B, 2, N] to match Conv1d expectations
    x_transposed = points.transpose(1, 2)
    
    # 3. Apply circular padding to handle the closed polygon boundary seamlessly
    pad_len = kernel_size // 2
    x_padded = F.pad(x_transposed, (pad_len, pad_len), mode="circular")
    
    # 4. Perform depthwise 1D convolution (groups=2 ensures X and Y are smoothed independently)
    smoothed = F.conv1d(x_padded, kernel, groups=2)
    
    # 5. Transpose back to original shape [B, N, 2]
    return smoothed.transpose(1, 2)

import torch

def taubin_smooth_closed_contour(points: torch.Tensor, iterations: int = 10, 
                                 lamb: float = 0.5, mu: float = -0.53) -> torch.Tensor:
    """Smoothes a closed contour [B, N, 2] without shrinking its volume/area
    using Taubin's alternating lambda-mu laplacian smoothing.
    """
    p = points.clone()
    
    for _ in range(iterations):
        # --- Step 1: Positive Smoothing (Shrinkage) ---
        prev_p = torch.roll(p, shifts=1, dims=1)
        next_p = torch.roll(p, shifts=-1, dims=1)
        # Discretized Laplacian operator on the polygon chain
        laplacian = 0.5 * (prev_p + next_p) - p
        p = p + lamb * laplacian
        
        # --- Step 2: Negative Smoothing (Inflation) ---
        prev_p = torch.roll(p, shifts=1, dims=1)
        next_p = torch.roll(p, shifts=-1, dims=1)
        laplacian = 0.5 * (prev_p + next_p) - p
        p = p + mu * laplacian
        
    return p