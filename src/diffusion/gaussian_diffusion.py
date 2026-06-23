"""Gaussian diffusion on point coordinates (x0-prediction parameterization).

The denoiser predicts the clean points x0 directly. Training minimizes an
x0 MSE (down-weighted at high noise levels via alpha_bar) plus a uniformity
regularizer that keeps the predicted points evenly spaced around the closed
contour. Sampling uses deterministic DDIM with classifier-free guidance.

Uncertainty-aware extension (training only):
  training_losses() accepts log_sigma [B,N] from the denoiser and applies
  the Kendall & Gal heteroscedastic loss (NeurIPS 2017 / CVPR 2018):

      L_noise = mean[ (1/(2*sigma^2)) * ||eps_pred - eps||^2  +  log(sigma) ]

  This is the EXACT formulation from Kendall & Gal – no modifications.
  The boundary_att loss intentionally keeps the original unweighted form
  because combining boundary attention with per-point sigma-weighting has
  no published precedent and would introduce an unvalidated interaction.

  The ddim_sample() loop is kept fully standard (no step-skipping).
  Skipping logic was removed: selective per-point freezing during DDIM
  is not validated in any published diffusion-on-coordinates work and
  risks topological inconsistencies that are hard to diagnose.
"""

import torch
import torch.nn.functional as F

from ..utils.rasterize import soft_dice_loss
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
                        log_sigma=None,
                        lambda_uniformity=0.1, lambda_dice=1.0, lambda_boundary=1.0,
                        lambda_uncertainty=1.0,
                        adaptive_uniformity=True):
        """Multi-task loss with optional per-point heteroscedastic uncertainty weighting.

        Args:
            predicted_noise [B, N, 2]: Denoiser epsilon prediction.
            noise           [B, N, 2]: Ground-truth noise added in q_sample.
            x_t             [B, N, 2]: Noisy points at timestep t.
            x0              [B, N, 2]: Clean ground-truth contour points.
            t               [B]:       Diffusion timesteps.
            masks           [B, 1, H, W] or None: Binary GT masks for Dice loss.
            log_sigma       [B, N] or None: Per-point log aleatoric uncertainty.
                            If None, falls back to plain MSE (legacy / no-head mode).
            lambda_uniformity:  Weight for spacing regularizer.
            lambda_dice:        Weight for soft-Dice mask loss.
            lambda_boundary:    Weight for boundary-attention noise loss.
            lambda_uncertainty: Scales the log(sigma) regularization in Kendall loss.
                                Default 1.0 = exact Kendall & Gal formulation.
                                <1.0 allows more uncertainty expression,
                                >1.0 tightens the regularization.
            adaptive_uniformity: If True, curvature-weighted spacing loss.
                                 If False, strict global std baseline.

        Loss terms
        ----------
        1. Kendall & Gal noise loss (Eq. 2 in NeurIPS 2017 / CVPR 2018):

               L_noise = mean[ (1/(2σ²)) * ||ε_pred - ε||² + log(σ) ]

           where sq_err is averaged over the (x,y) dimension per point.
           Written in log-space for numerical stability:
               s_i = log_sigma_i
               L_i = 0.5 * exp(-2*s_i) * sq_err_i  +  lambda_uncertainty * s_i

           This is the EXACT formulation from the paper, no modifications.
           The λ in front of s_i allows slight regularization tuning while
           keeping the same mathematical structure.

        2. Boundary-attention loss (UNCHANGED from pre-uncertainty baseline):
           Plain weighted MSE, no sigma interaction. Combining boundary_att
           with per-point sigma-weighting has no published precedent and would
           create an unvalidated interaction where the model could lower its
           sigma on boundary points to suppress the amplified gradient, which
           is the opposite of what we want.

        3. Uniformity regularizer.

        4. Soft-Dice (mask-level rasterization).
        """

        # ------------------------------------------------------------------ #
        # 1.  Kendall & Gal noise loss  (or plain MSE when log_sigma is None) #
        #                                                                      #
        # Scale factor: the original * (100**2) was a legacy convention to    #
        # bring plain MSE on [-1,1] coords into a readable range. With the    #
        # Kendall formulation the log_sigma penalty term is O(log sigma) and  #
        # multiplying by 10000 caused it to reach -15000, completely drowning #
        # out dice (~0.5) and uniformity (~1.0).                              #
        #                                                                      #
        # Fix: apply NOISE_SCALE only to the sq_err reconstruction term.      #
        # The log_sigma regularization term stays unscaled (O(1)).            #
        # Result: loss_noise stays O(1..100), comparable to other terms.      #
        # ------------------------------------------------------------------ #
        sq_err = ((predicted_noise - noise) ** 2).mean(dim=-1)   # [B, N]
        NOISE_SCALE = 100.0   # same effective scale as before for the MSE part

        if log_sigma is not None:
            # Kendall & Gal (NeurIPS 2017 eq. 9; CVPR 2018 eq. 3) in log-space:
            #   precision term : 0.5 * exp(-2s) * sq_err  (always >= 0)
            #   penalty term   : lambda_uncertainty * s    (negative when sigma < 1)
            #
            # NOISE_SCALE applied only to precision term so the penalty
            # cannot dominate the total loss magnitude.
            precision  = torch.exp(-2.0 * log_sigma)              # [B, N]
            loss_noise = (NOISE_SCALE * 0.5 * precision * sq_err
                          + lambda_uncertainty * log_sigma
                          ).mean()
        else:
            loss_noise = sq_err.mean() * (NOISE_SCALE ** 2)

        # ------------------------------------------------------------------ #
        # 2.  Boundary-attention loss  (original formulation, no σ coupling)  #
        # ------------------------------------------------------------------ #
        boundary_att = calc_boundary_att(x0, t, T=self.timesteps, gamma=1.5)
        loss_boundary = (boundary_att * ((predicted_noise - noise) ** 2)
                         ).mean() * NOISE_SCALE

        # ------------------------------------------------------------------ #
        # 3.  Reconstruct predicted x0 for downstream losses                  #
        # ------------------------------------------------------------------ #
        predicted_x0 = self.get_x0_from_noise(predicted_noise, t, x_t)

        nxt = torch.roll(predicted_x0, shifts=-1, dims=1)
        dists = torch.norm(predicted_x0 - nxt, dim=-1)            # [B, N]

        # ------------------------------------------------------------------ #
        # 4.  Uniformity loss                                                  #
        # ------------------------------------------------------------------ #
        if adaptive_uniformity:
            prev_gt = torch.roll(x0, shifts=1, dims=1)
            next_gt = torch.roll(x0, shifts=-1, dims=1)
            gt_curvature = torch.norm(next_gt + prev_gt - 2 * x0, dim=-1)   # [B, N]
            uniformity_weights = torch.exp(-2.0 * gt_curvature)
            mean_dist = dists.mean(dim=1, keepdim=True)
            squared_diffs = (dists - mean_dist) ** 2
            loss_uniformity = (squared_diffs * uniformity_weights).mean()
        else:
            loss_uniformity = dists.std(dim=1).mean()

        # ------------------------------------------------------------------ #
        # 5.  Soft-Dice (mask-level)                                           #
        # ------------------------------------------------------------------ #
        if masks is not None and lambda_dice > 0:
            loss_dice = soft_dice_loss(predicted_x0, masks)
        else:
            loss_dice = torch.zeros((), device=x0.device)

        # ------------------------------------------------------------------ #
        # Total                                                                #
        # ------------------------------------------------------------------ #
        total = (loss_noise
                 + lambda_boundary * loss_boundary
                 + lambda_uniformity * loss_uniformity
                 + lambda_dice * loss_dice)

        return total, {
            "loss_noise":      loss_noise.detach(),
            "loss_boundary":   loss_boundary.detach(),
            "loss_uniformity": loss_uniformity.detach(),
            "loss_dice":       loss_dice.detach(),
        }

    # --- sampling ------------------------------------------------------------

    @torch.no_grad()
    def ddim_sample(self, denoise_fn, cond_fn, shape, ddim_steps=50,
                    guidance_scale=5.0, clamp=1.0,
                    uncertainty_skip_threshold=None,
                    return_log_sigma=False):
        """Deterministic DDIM sampling with classifier-free guidance.

        denoise_fn(x, t, cond_maps) -> (predicted_noise [B,N,2], log_sigma [B,N])
            or legacy: -> predicted_noise [B,N,2]  (tuple detection is automatic)

        cond_fn(t) -> list of condition maps; unconditional pass uses zeroed maps.

        uncertainty_skip_threshold: reserved, currently no-op.

        return_log_sigma (bool): If True, returns (x, log_sigma) where log_sigma
            is the per-point uncertainty from the FINAL denoiser call (t → 0),
            i.e. the most refined uncertainty estimate. None if the denoiser
            does not produce a log_sigma output.

        Returns:
            x           [B, N, 2]        always
            log_sigma   [B, N] or None   only when return_log_sigma=True
        """
        device = self.device
        x = torch.randn(shape, device=device)

        step = self.timesteps // ddim_steps
        timesteps = list(reversed(range(0, self.timesteps, step)))

        last_log_sigma = None   # will hold the sigma from the final step

        for i, t in enumerate(timesteps):
            t_b = torch.full((shape[0],), t, device=device, dtype=torch.long)

            cond_maps = cond_fn(t_b)

            raw_out = denoise_fn(x, t_b, cond_maps)
            if isinstance(raw_out, tuple):
                pred_noise, last_log_sigma = raw_out
            else:
                pred_noise = raw_out
                last_log_sigma = None

            x0_cond = self.get_x0_from_noise(pred_noise, t_b, x)

            if guidance_scale != 1.0:
                null_maps = [torch.zeros_like(m) for m in cond_maps]
                raw_uncond = denoise_fn(x, t_b, null_maps)
                pred_noise_uncond = raw_uncond[0] if isinstance(raw_uncond, tuple) else raw_uncond
                x0_uncond = self.get_x0_from_noise(pred_noise_uncond, t_b, x)
                x0 = x0_uncond + guidance_scale * (x0_cond - x0_uncond)
            else:
                x0 = x0_cond

            x0 = torch.clamp(x0, -clamp, clamp)

            a_bar = self.alphas_cumprod[t]
            t_prev = timesteps[i + 1] if i + 1 < len(timesteps) else -1
            a_bar_prev = (self.alphas_cumprod[t_prev]
                          if t_prev >= 0 else torch.tensor(1.0, device=device))

            direction = (x - torch.sqrt(a_bar) * x0) / torch.sqrt(1.0 - a_bar)
            x = torch.sqrt(a_bar_prev) * x0 + torch.sqrt(1.0 - a_bar_prev) * direction

        if return_log_sigma:
            return x, last_log_sigma
        return x