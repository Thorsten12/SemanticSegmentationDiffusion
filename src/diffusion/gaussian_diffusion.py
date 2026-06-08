"""Gaussian diffusion on point coordinates (x0-prediction parameterization).

The denoiser predicts the clean points x0 directly. Training minimizes an
x0 MSE (down-weighted at high noise levels via alpha_bar) plus a uniformity
regularizer that keeps the predicted points evenly spaced around the closed
contour. Sampling uses deterministic DDIM with classifier-free guidance.
"""

import torch
import torch.nn.functional as F

from ..utils.rasterize import soft_dice_loss


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
        return (
            _extract(self.sqrt_alphas_cumprod, t, x0.shape) * x0
            + _extract(self.sqrt_one_minus_alphas_cumprod, t, x0.shape) * noise
        )

    # --- training loss -------------------------------------------------------

    def training_losses(self, predicted_x0, x0, t, masks=None,
                        lambda_uniformity=0.1, lambda_dice=1.0, snr_gamma=5.0):
        """Per-sample min-SNR-weighted x0 MSE + uniformity + soft-Dice.

        masks (optional, [B,1,H,W] in {0,1}) enables the differentiable soft-Dice
        term, which gives the boundary points a geometry-aware (mask-level) signal
        instead of relying solely on order-dependent coordinate MSE.
        """
        # Per-sample min-SNR-gamma weighting (x0 parameterization). SNR = a_bar /
        # (1 - a_bar); clamping at gamma stops low-noise steps from dominating, and
        # -- unlike the old a_bar weighting -- high-noise steps keep real gradient.
        a_bar = _extract(self.alphas_cumprod, t, x0.shape)            # [B,1,1]
        snr = a_bar / (1.0 - a_bar).clamp(min=1e-8)
        w = snr.clamp(max=snr_gamma)
        w = w / w.mean().clamp(min=1e-8)                             # keep scale ~1
        se = ((predicted_x0 - x0) ** 2).mean(dim=tuple(range(1, x0.ndim)),
                                             keepdim=True)            # [B,1,1]
        loss_x0 = (w * se).mean()

        # Closed-contour uniformity: penalize variance of neighbor distances.
        nxt = torch.roll(predicted_x0, shifts=-1, dims=1)
        dists = torch.norm(predicted_x0 - nxt, dim=-1)        # [B, N]
        loss_uniformity = dists.std(dim=1).mean()

        # Differentiable mask-level term (rasterize polygon -> soft-Dice).
        if masks is not None and lambda_dice > 0:
            loss_dice = soft_dice_loss(predicted_x0, masks)
        else:
            loss_dice = torch.zeros((), device=x0.device)

        total = loss_x0 + lambda_uniformity * loss_uniformity + lambda_dice * loss_dice
        return total, {"loss_x0": loss_x0.detach(),
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
            x0_cond = denoise_fn(x, t_b, cond_maps)
            if guidance_scale != 1.0:
                null_maps = [torch.zeros_like(m) for m in cond_maps]
                x0_uncond = denoise_fn(x, t_b, null_maps)
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
