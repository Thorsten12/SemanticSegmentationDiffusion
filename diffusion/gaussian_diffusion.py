"""Gaussian diffusion on contour point coordinates (x0-prediction).

The denoiser predicts the clean points x0 directly. Training minimizes a
min-SNR-weighted x0 MSE together with auxiliary geometric terms (soft-Dice,
Boundary-IoU, nearest-point, boundary attention, uniformity, curvature).
Sampling uses deterministic DDIM with classifier-free guidance (CFG).

Optional components
-------------------
Coarse-contour proposal:
    A deterministic coarse contour (see `proposal.py`) provides global
    position, size and coarse shape; the denoiser then predicts only the
    residual on top of it (`proposal_target="residual"`). The proposal is
    combined with the reconstructed x0 identically in training (by the caller)
    and in `ddim_sample`. This is distinct from the points-vs-noise residual
    inside the denoiser itself (`ContourDenoiser.predict_residual`).

In-loop boundary snapping:
    `ddim_sample` can apply a `BoundarySnapper` to the x0 estimate inside the
    sampling loop, restricted to low-noise steps. This is an alternative to,
    and can be combined with, the standalone post-DDIM refinement described
    in `snapper.py`.
"""

from ..utils.helper_funcs import calc_boundary_att
import torch
import torch.nn.functional as F

from ..models import apply_proposal_target
from ..utils.rasterize import soft_dice_loss, soft_boundary_iou_loss


def _extract(a: torch.Tensor, t: torch.Tensor, x_shape) -> torch.Tensor:
    """Gather schedule values at timesteps t and broadcast to x_shape."""
    out = a.gather(-1, t)
    return out.reshape(t.shape[0], *((1,) * (len(x_shape) - 1)))


def nearest_point_loss(predicted_x0, x0, t, T, gamma=1.0):
    """Asymmetric nearest-neighbor loss (prediction -> ground truth).

    For each predicted point, the distance to the closest ground-truth contour
    point is computed. Unlike the x0 MSE, no one-to-one vertex correspondence
    is enforced. The loss is weighted by (1 - t / (T - 1))^gamma, i.e. maximal
    at t = 0 and vanishing at t = T - 1.

    Args:
        predicted_x0, x0: [B, N, 2].
        t: [B] long, timestep indices.
    """
    dists = torch.cdist(predicted_x0, x0, p=2)          # [B, N_pred, N_gt]
    min_dists, _ = dists.min(dim=-1)                    # [B, N_pred]
    per_sample = min_dists.mean(dim=-1)                 # [B]

    w_t = (1.0 - t.float() / (T - 1)).clamp(min=0.0, max=1.0) ** gamma  # [B]
    loss = (w_t * per_sample).mean()
    return loss


def curvature_penalty_loss(predicted_x0, t, T, gamma=1.0):
    """Smoothness prior on the predicted contour (second-difference penalty).

    Penalizes the squared norm of the second difference along the closed
    contour. The term is a pure shape prior on the prediction and involves no
    comparison to the ground truth.

    Weighted by (t / (T - 1))^gamma, i.e. strong at high noise and fading
    towards t = 0. At high noise, only a coarse, low-frequency contour can be
    inferred from the low-resolution early-scale features, so smoothness is
    enforced there; the constraint is relaxed late in the chain, where finer
    scales provide local information. This complements `nearest_point_loss`
    and the Boundary-IoU term, which demand accuracy at low t.

    Args:
        predicted_x0: [B, N, 2].
        t: [B] long, timestep indices.
    """
    prev_p = torch.roll(predicted_x0, shifts=1, dims=1)
    next_p = torch.roll(predicted_x0, shifts=-1, dims=1)
    curvature = torch.norm(next_p + prev_p - 2 * predicted_x0, dim=-1)  # [B, N]
    per_sample = (curvature ** 2).mean(dim=-1)                          # [B]

    w_t = (t.float() / (T - 1)).clamp(min=0.0, max=1.0) ** gamma
    return (w_t * per_sample).mean()


class GaussianDiffusion:
    def __init__(self, timesteps=1000, beta_start=1e-4, beta_end=2e-2, device="cuda"):
        self.timesteps = timesteps
        self.device = device

        # Linear beta schedule and derived cumulative quantities.
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
                        lambda_mse=1.0, lambda_uniformity=0.0, lambda_dice=1.0, lambda_boundary=1.0, snr_gamma=5.0,
                        lambda_biou=1.0, biou_gamma=1.0,
                        lambda_nearest=1.0, nearest_gamma=1.0,
                        lambda_curvature=0.0, curvature_gamma=1.0,
                        soft_dice_size=64):
        """Weighted sum of coordinate-level and mask-level training losses.

        Terms:
            loss_x0:         min-SNR-weighted x0 MSE.
            loss_boundary:   boundary-attention-weighted squared error.
            loss_uniformity: std of neighbor distances along the closed contour
                             (keeps points evenly spaced).
            loss_dice:       soft-Dice between the rasterized polygon and the
                             GT mask; time-weighted, strong at high t (coarse
                             shape).
            loss_biou:       differentiable Boundary IoU (Sun et al., MICCAI
                             2023) between the rasterized polygon and the GT
                             mask, focused on the boundary band. Time-weighted
                             opposite to loss_dice: strongest at low t (fine
                             boundary detail), fading to 0 at high t.
            loss_nearest:    asymmetric pred -> GT nearest-neighbor distance;
                             time-weighted like loss_biou.
            loss_curvature:  second-difference smoothness prior on the
                             prediction (no GT); time-weighted like loss_dice.
                             See `curvature_penalty_loss`.

        Mask-level terms (dice, biou) require `masks` ([B, 1, H, W] in {0, 1})
        and give the points a geometry-aware signal instead of relying solely
        on order-dependent coordinate MSE.

        Every term is computed only if its lambda is > 0, which saves compute
        and ensures the logged value reflects what contributes to `total`.

        `predicted_x0` must already be the final x0 candidate: when a proposal
        is used, the caller has combined the denoiser output with the proposal
        via `apply_proposal_target`. This function is proposal-agnostic.

        Args:
            predicted_x0, x0: [B, N, 2].
            t: [B] long, timestep indices.
            snr_gamma: Clamp value of the min-SNR weight.
        Returns:
            (total_loss, dict of detached per-term losses).
        """
        # Min-SNR-gamma weighting (x0 parameterization): SNR = a_bar / (1 - a_bar),
        # clamped at gamma so low-noise steps do not dominate while high-noise
        # steps retain meaningful gradient. Normalized to keep the scale ~1.
        if lambda_mse > 0:
            a_bar = _extract(self.alphas_cumprod, t, x0.shape)            # [B,1,1]
            snr = a_bar / (1.0 - a_bar).clamp(min=1e-8)
            w = snr.clamp(max=snr_gamma)
            w = w / w.mean().clamp(min=1e-8)
            se = ((predicted_x0 - x0) ** 2).mean(dim=tuple(range(1, x0.ndim)),
                                                 keepdim=True)            # [B,1,1]
            loss_x0 = (w * se).mean()
        else:
            loss_x0 = torch.zeros((), device=x0.device)

        if lambda_boundary > 0:
            boundary_att = calc_boundary_att(x0, t, T=self.timesteps, gamma=1.5)
            loss_boundary = (boundary_att * ((predicted_x0 - x0) ** 2)).mean()
        else:
            loss_boundary = torch.zeros((), device=x0.device)

        # Uniformity: penalize the spread of neighbor distances on the closed contour.
        if lambda_uniformity > 0:
            nxt = torch.roll(predicted_x0, shifts=-1, dims=1)
            dists = torch.norm(predicted_x0 - nxt, dim=-1)        # [B, N]
            loss_uniformity = dists.std(dim=1).mean()
        else:
            loss_uniformity = torch.zeros((), device=x0.device)

        # Mask-level terms: rasterize the polygon differentiably, then compare to GT.
        if masks is not None and lambda_dice > 0:
            loss_dice = soft_dice_loss(predicted_x0, masks, size=soft_dice_size,
                                       t=t, T=self.timesteps)
        else:
            loss_dice = torch.zeros((), device=x0.device)

        if masks is not None and lambda_biou > 0:
            loss_biou = soft_boundary_iou_loss(predicted_x0, masks, size=soft_dice_size,
                                               t=t, T=self.timesteps, gamma=biou_gamma)
        else:
            loss_biou = torch.zeros((), device=x0.device)

        if lambda_nearest > 0:
            loss_nearest = nearest_point_loss(predicted_x0, x0, t, T=self.timesteps,
                                              gamma=nearest_gamma)
        else:
            loss_nearest = torch.zeros((), device=x0.device)

        if lambda_curvature > 0:
            loss_curvature = curvature_penalty_loss(predicted_x0, t, T=self.timesteps,
                                                    gamma=curvature_gamma)
        else:
            loss_curvature = torch.zeros((), device=x0.device)

        total = (lambda_mse * loss_x0 + lambda_uniformity * loss_uniformity + lambda_dice * loss_dice
                + lambda_boundary * loss_boundary + lambda_biou * loss_biou
                + lambda_nearest * loss_nearest + lambda_curvature * loss_curvature)

        return total, {"loss_x0": loss_x0.detach(),
                       "loss_uniformity": loss_uniformity.detach(),
                       "loss_dice": loss_dice.detach(),
                       "loss_boundary": loss_boundary.detach(),
                       "loss_biou": loss_biou.detach(),
                       "loss_nearest": loss_nearest.detach(),
                       "loss_curvature": loss_curvature.detach()}

    # --- sampling ------------------------------------------------------------

    @torch.no_grad()
    def ddim_sample(self, denoise_fn, cond_fn, shape, ddim_steps=50, eta=0.2,
                    proposal=None, proposal_target="absolute",
                    guidance_scale=5.0, clamp=1.0,
                    snapper=None, snap_maps=None, snap_image=None,
                    snap_t_threshold=None, snap_every=1,
                    collect_every=0):
        """Deterministic DDIM sampling with classifier-free guidance.

        At each step, the x0 estimate is formed as follows: denoiser output
        (conditional and, if CFG is active, unconditional) -> CFG interpolation
        -> proposal combination -> clamp -> optional snapping. The DDIM update
        then uses this final x0.

        Args:
            denoise_fn: (x, t, cond_maps) -> predicted x0, or the residual on
                the proposal, depending on `proposal_target`. The denoiser is
                unaware of the proposal; it is combined here.
            cond_fn: t -> list of condition maps (may be time-dependent). The
                unconditional CFG pass uses zeroed maps.
            shape: Shape of the sampled points, [B, N, 2].
            guidance_scale: CFG scale; x0 = x0_uncond + s * (x0_cond - x0_uncond).
                A value of 1.0 disables the unconditional pass.
            clamp: Symmetric clamp applied to x0 in normalized coordinates.
            proposal: [B, N, 2] deterministic coarse-contour proposal (see
                `proposal.py`) or None. It does not depend on t and is
                computed once by the caller.
            proposal_target: "absolute" (proposal ignored) or "residual"
                (x0 = proposal + denoiser output at every step). The proposal
                is added once per step AFTER the CFG interpolation, matching
                training, where the conditional prediction is combined with
                the proposal exactly once.

            In-loop snapping:
            snapper: Optional `BoundarySnapper`, applied to the per-step x0
                estimate, restricted to low-noise steps.
            snap_maps: Raw backbone feature maps (not the fused `cond_maps`
                from `cond_fn`), as `BoundarySnapper` samples raw multi-scale
                features directly. Required if `snapper` is given; computed
                once by the caller.
            snap_image: Optional raw image for the snapper's RGB path. If None,
                the snapper skips its RGB term.
            snap_t_threshold: Snap only at timesteps t <= snap_t_threshold
                (the tail of sampling). None disables in-loop snapping even if
                a snapper is given. A typical value is a small fraction of
                `self.timesteps`, e.g. int(0.15 * timesteps).
            snap_every: Within the eligible window, snap only every
                `snap_every`-th DDIM step to trade correction frequency for
                compute. 1 snaps at every eligible step.

            collect_every: If > 0, the x0 estimate at every `collect_every`-th
                DDIM step (and always the last) is stored and returned as
                `(x, intermediates)`, where `intermediates` is a list of
                detached [B, N, 2] tensors in sampling order (coarse -> clean).
                If <= 0 (default), only `x` is returned.

            eta: Unused; sampling is fully deterministic.
        """
        device = self.device
        x = torch.randn(shape, device=device)

        step = self.timesteps // ddim_steps
        timesteps = list(reversed(range(0, self.timesteps, step)))

        do_snap = snapper is not None and snap_maps is not None and snap_t_threshold is not None
        snap_call_idx = 0

        do_collect = collect_every and collect_every > 0
        intermediates = [] if do_collect else None

        for i, t in enumerate(timesteps):
            t_b = torch.full((shape[0],), t, device=device, dtype=torch.long)

            # Classifier-free guidance.
            cond_maps = cond_fn(t_b)
            x0_cond = denoise_fn(x, t_b, cond_maps)
            if guidance_scale != 1.0:
                null_maps = [torch.zeros_like(m) for m in cond_maps]
                x0_uncond = denoise_fn(x, t_b, null_maps)
                x0 = x0_uncond + guidance_scale * (x0_cond - x0_uncond)
            else:
                x0 = x0_cond
            x0 = apply_proposal_target(x0, proposal, proposal_target)
            x0 = torch.clamp(x0, -clamp, clamp)

            # Optional in-loop snapping (low-noise steps only).
            if do_snap and t <= snap_t_threshold:
                if snap_call_idx % max(1, int(snap_every)) == 0:
                    x0 = snapper(x0, snap_maps, image=snap_image, hard=True)
                    x0 = torch.clamp(x0, -clamp, clamp)
                snap_call_idx += 1

            # Record the x0 this step commits to (after proposal, clamp and
            # snapping), i.e. the same quantity used in the DDIM update below.
            if do_collect:
                is_last = (i == len(timesteps) - 1)
                if is_last or (i % collect_every == 0):
                    intermediates.append(x0.detach().clone())

            # DDIM update (eta = 0): x_{t_prev} = sqrt(a_prev) * x0 + sqrt(1 - a_prev) * eps_hat.
            a_bar = self.alphas_cumprod[t]
            t_prev = timesteps[i + 1] if i + 1 < len(timesteps) else -1
            a_bar_prev = self.alphas_cumprod[t_prev] if t_prev >= 0 else torch.tensor(1.0, device=device)

            direction = (x - torch.sqrt(a_bar) * x0) / torch.sqrt(1.0 - a_bar)
            x = torch.sqrt(a_bar_prev) * x0 + torch.sqrt(1.0 - a_bar_prev) * direction

        if do_collect:
            return x, intermediates
        return x