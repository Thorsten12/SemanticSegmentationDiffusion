"""Gaussian diffusion on point coordinates (x0-prediction parameterization).

The denoiser predicts the clean points x0 directly. Training minimizes an
x0 MSE (down-weighted at high noise levels via alpha_bar) plus a uniformity
regularizer that keeps the predicted points evenly spaced around the closed
contour. Sampling uses deterministic DDIM with classifier-free guidance.
"""

import torch
import torch.nn.functional as F

from ..utils.rasterize import soft_dice_loss
# Added import for boundary attention
from ..utils.helper_funcs import calc_boundary_att


def _extract(a: torch.Tensor, t: torch.Tensor, x_shape) -> torch.Tensor:
    """Gather schedule values at timesteps t and broadcast to x_shape."""
    out = a.gather(-1, t)
    return out.reshape(t.shape[0], *((1,) * (len(x_shape) - 1)))


class GaussianDiffusion:
    def __init__(self, timesteps=1000, beta_start=1e-4, beta_end=2e-2, device="cuda"):
        self.timesteps = timesteps
        self.device = device

        betas = torch.linspace(beta_start, beta_end, timesteps, device=device)
        alphas = 1.0 - betas
        self.betas = betas
        self.alphas_cumprod = torch.cumprod(alphas, dim=0)
        self.sqrt_alphas_cumprod = torch.sqrt(self.alphas_cumprod)
        self.sqrt_one_minus_alphas_cumprod = torch.sqrt(1.0 - self.alphas_cumprod)

    # --- forward process -----------------------------------------------------
    def q_sample(self, x0, t, noise=None):
        """Sample x_t ~ q(x_t | x0)."""
        if noise is None:
            noise = torch.randn_like(x0)
            noise = torch.clamp(noise, -3.0, 3.0)  # clamp noise to stabilize training
        return (
            _extract(self.sqrt_alphas_cumprod, t, x0.shape) * x0
            + _extract(self.sqrt_one_minus_alphas_cumprod, t, x0.shape) * noise
        ), noise

    def get_x0_from_noise(self, noise_pred, t, x_t):
        """Compute predicted x0 from noise prediction."""
        sqrt_a_bar = _extract(self.sqrt_alphas_cumprod, t, x_t.shape)
        sqrt_one_minus_a_bar = _extract(self.sqrt_one_minus_alphas_cumprod, t, x_t.shape)
        return (x_t - sqrt_one_minus_a_bar * noise_pred) / sqrt_a_bar.clamp(min=1e-8)
    
    # --- training loss -------------------------------------------------------

    def training_losses(self, predicted_noise, noise, x_t, x0, t, masks=None,
                    lambda_uniformity=0.1, lambda_dice=1.0, lambda_boundary=1.0,
                    adaptive_uniformity=True):
        """Noise MSE scaled by 100^2 + boundary loss + uniformity + soft-Dice via reconstructed x0.
        
        Args:
            adaptive_uniformity (bool): If True, uses curvature-weighted relaxation in indents.
                                       If False, enforces strict equidistant sampling globally (Baseline).
        """
        
        # 1. Element-wise MSE for noise scaled by 100^2
        loss_noise = F.mse_loss(predicted_noise, noise) * (100 ** 2)
    
        # 2. Boundary-weighted noise loss scaled by 100^2
        boundary_att = calc_boundary_att(x0, t, T=self.timesteps, gamma=1.5)
        loss_boundary = (boundary_att * (predicted_noise - noise) ** 2).mean() * (100 ** 2)
    
        # 3. Reconstruct predicted_x0 using the tutor's helper function
        predicted_x0 = self.get_x0_from_noise(predicted_noise, t, x_t)
    
        # Compute distances between predicted adjacent points (needed for both modes)
        nxt = torch.roll(predicted_x0, shifts=-1, dims=1)
        dists = torch.norm(predicted_x0 - nxt, dim=-1)        # [B, N]
    
        # 4. Conditional Uniformity Loss Calculation
        if adaptive_uniformity:
            # --- Curvature-adaptive uniformity loss ---
            # Calculate curvature on the clean ground-truth points (x0)
            prev_gt = torch.roll(x0, shifts=1, dims=1)
            next_gt = torch.roll(x0, shifts=-1, dims=1)
            gt_curvature = torch.norm(next_gt + prev_gt - 2 * x0, dim=-1)  # [B, N]
    
            # Map high curvature to low uniformity weights using negative exponential
            uniformity_weights = torch.exp(-2.0 * gt_curvature)  # [B, N]
    
            # Calculate squared deviation from the mean distance per batch item
            mean_dist = dists.mean(dim=1, keepdim=True)           # [B, 1]
            squared_diffs = (dists - mean_dist) ** 2               # [B, N]
    
            # Apply weights: penalize variance only on flat segments, relax in indents
            loss_uniformity = (squared_diffs * uniformity_weights).mean()
        else:
            # --- Classic strict uniform loss (Baseline) ---
            # Calculates standard deviation of distances globally per sample
            loss_uniformity = dists.std(dim=1).mean()
    
        # 5. Differentiable mask-level term (rasterize polygon -> soft-Dice).
        if masks is not None and lambda_dice > 0:
            loss_dice = soft_dice_loss(predicted_x0, masks)
        else:
            loss_dice = torch.zeros((), device=x0.device)
    
        total = loss_noise + lambda_boundary * loss_boundary + lambda_uniformity * loss_uniformity + lambda_dice * loss_dice
        return total, {"loss_noise": loss_noise.detach(),
                       "loss_boundary": loss_boundary.detach(),
                       "loss_uniformity": loss_uniformity.detach(),
                       "loss_dice": loss_dice.detach()}

        # --- sampling ------------------------------------------------------------

    @torch.no_grad()
    def ddim_sample(self, denoise_fn, cond_fn, shape, ddim_steps=50,
                    guidance_scale=5.0, clamp=1.0):
        """Deterministic DDIM sampling with classifier-free guidance.

        denoise_fn(x, t, cond_maps) -> predicted x0.
        cond_fn(t) -> list of condition maps for timestep t (may be time-dependent);
        the unconditional pass uses zeroed maps.
        """
        device = self.device
        x = torch.randn(shape, device=device)

        step = self.timesteps // ddim_steps
        timesteps = list(reversed(range(0, self.timesteps, step)))

        for i, t in enumerate(timesteps):
            t_b = torch.full((shape[0],), t, device=device, dtype=torch.long)

            cond_maps = cond_fn(t_b)
            pred_noise = denoise_fn(x, t_b, cond_maps)
            x0_cond = self.get_x0_from_noise(pred_noise, t_b, x)
            if guidance_scale != 1.0:
                null_maps = [torch.zeros_like(m) for m in cond_maps]
                pred_noise_uncond = denoise_fn(x, t_b, null_maps)
                x0_uncond = self.get_x0_from_noise(pred_noise_uncond, t_b, x)
                x0 = x0_uncond + guidance_scale * (x0_cond - x0_uncond)
            else:
                x0 = x0_cond
            x0 = torch.clamp(x0, -clamp, clamp)

            a_bar = self.alphas_cumprod[t]
            t_prev = timesteps[i + 1] if i + 1 < len(timesteps) else -1
            a_bar_prev = self.alphas_cumprod[t_prev] if t_prev >= 0 else torch.tensor(1.0, device=device)

            direction = (x - torch.sqrt(a_bar) * x0) / torch.sqrt(1.0 - a_bar)
            x = torch.sqrt(a_bar_prev) * x0 + torch.sqrt(1.0 - a_bar_prev) * direction

        return x