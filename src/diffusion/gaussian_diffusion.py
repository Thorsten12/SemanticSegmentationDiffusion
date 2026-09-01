"""Gaussian diffusion in an unbounded latent contour coordinate space.

Ground-truth image coordinates p in [-1,1] are mapped to
``z = atanh(p)``.  Standard DDPM/DDIM algebra is performed on z, while the
network uses ``tanh(z_t)`` whenever it needs to query the 2-D image.  This keeps
all feature sampling geometrically valid even at the noisiest timesteps.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F

from ..utils.rasterize import soft_dice_loss, soft_mask_boundary_losses


def _extract(a: torch.Tensor, t: torch.Tensor, x_shape) -> torch.Tensor:
    out = a.gather(-1, t)
    return out.reshape(t.shape[0], *((1,) * (len(x_shape) - 1)))


def points_to_latent(points: torch.Tensor, eps: float = 1e-4) -> torch.Tensor:
    p = points.clamp(-1.0 + eps, 1.0 - eps)
    return 0.5 * (torch.log1p(p) - torch.log1p(-p))


def latent_to_points(latent: torch.Tensor) -> torch.Tensor:
    return torch.tanh(latent)


def _cyclic_candidates(target: torch.Tensor, allow_reverse: bool = True) -> torch.Tensor:
    """Return [B,C,N,2] cyclic (and optionally reversed) target contours."""
    n = target.shape[1]
    direct = torch.stack([torch.roll(target, shifts=-s, dims=1) for s in range(n)], dim=1)
    if not allow_reverse:
        return direct
    rev = torch.flip(target, dims=[1])
    reverse = torch.stack([torch.roll(rev, shifts=-s, dims=1) for s in range(n)], dim=1)
    return torch.cat([direct, reverse], dim=1)


def best_cyclic_alignment(pred_points: torch.Tensor, target: torch.Tensor,
                          allow_reverse: bool = True) -> torch.Tensor:
    """Align target phase/direction to prediction without changing polygon shape.

    The discrete best shift is selected without gradient; gradients then flow
    through the loss against that aligned target.  N is only ~100, so exhaustive
    cyclic matching is cheap and much more stable than relying on one top-most
    vertex as a perfect canonical start.
    """
    candidates = _cyclic_candidates(target, allow_reverse=allow_reverse)
    with torch.no_grad():
        cost = F.smooth_l1_loss(
            pred_points[:, None].expand_as(candidates), candidates,
            reduction="none",
        ).mean(dim=(-1, -2))
        best = cost.argmin(dim=1)
    batch = torch.arange(target.shape[0], device=target.device)
    return candidates[batch, best]


class GaussianDiffusion:
    def __init__(self, timesteps=1000, beta_start=1e-4, beta_end=2e-2, device="cuda"):
        self.timesteps = int(timesteps)
        self.device = device

        betas = torch.linspace(beta_start, beta_end, self.timesteps, device=device)
        alphas = 1.0 - betas
        self.betas = betas
        self.alphas_cumprod = torch.cumprod(alphas, dim=0)
        self.sqrt_alphas_cumprod = torch.sqrt(self.alphas_cumprod)
        self.sqrt_one_minus_alphas_cumprod = torch.sqrt(1.0 - self.alphas_cumprod)

    # --- coordinate transforms ---------------------------------------------

    @staticmethod
    def points_to_latent(points):
        return points_to_latent(points)

    @staticmethod
    def latent_to_points(latent):
        return latent_to_points(latent)

    # --- forward process ----------------------------------------------------

    def q_sample(self, points, t, noise=None):
        """Sample z_t from a clean contour given in image coordinates."""
        z0 = points_to_latent(points)
        return self.q_sample_state(z0, t, noise=noise)

    def q_sample_state(self, clean_state, t, noise=None):
        """Sample a Gaussian state (used directly by residual diffusion)."""
        z0 = clean_state
        if noise is None:
            noise = torch.randn_like(z0)
        return (
            _extract(self.sqrt_alphas_cumprod, t, z0.shape) * z0
            + _extract(self.sqrt_one_minus_alphas_cumprod, t, z0.shape) * noise
        )

    # --- training loss ------------------------------------------------------

    def training_losses(
        self,
        predicted_z0,
        target_points,
        t,
        masks=None,
        lambda_uniformity=0.0,
        lambda_dice=1.0,
        lambda_chamfer=0.25,
        lambda_edge=0.10,
        snr_gamma=5.0,
        loss_weighting="uniform",
        soft_dice_size=64,
        cyclic_reverse=True,
        geometry_points=None,
        lambda_boundary_band=0.0,
        boundary_band_width=2,
        lambda_hd=0.0,
        hd_fraction=0.20,
        lambda_curvature=0.0,
        lambda_hard_boundary=0.0,
        hard_boundary_fraction=0.15,
        target_state=None,
        predicted_points=None,
    ):
        """Geometry-aware loss for a predicted clean latent contour.

        * index-consistent latent Smooth-L1 (stable diffusion target)
        * symmetric Chamfer boundary loss (correspondence free)
        * neighbour edge-length matching (prevents collapse / self-shortening)
        * differentiable polygon soft-Dice (mask-level supervision)
        """
        del cyclic_reverse  # retained as a backwards-compatible argument
        pred_points = (latent_to_points(predicted_z0)
                       if predicted_points is None else predicted_points)
        geom_points = pred_points if geometry_points is None else geometry_points

        # IMPORTANT: the diffusion regression target stays index-consistent with
        # the forward process z_t[i] = a*z_0[i] + noise[i].  Cyclic/reversal
        # matching is useful for shape diagnostics, but using a shifted z_0 in
        # the DDPM regression would make the DDIM epsilon estimate inconsistent.
        target_z0 = (points_to_latent(target_points)
                     if target_state is None else target_state)
        per_sample = F.smooth_l1_loss(
            predicted_z0, target_z0, reduction="none", beta=0.1,
        ).mean(dim=(1, 2))

        if loss_weighting == "min_snr":
            a_bar = self.alphas_cumprod.gather(0, t)
            snr = a_bar / (1.0 - a_bar).clamp(min=1e-8)
            weights = snr.clamp(max=float(snr_gamma))
            loss_x0 = (weights * per_sample).mean()
        elif loss_weighting == "uniform":
            loss_x0 = per_sample.mean()
        else:
            raise ValueError(f"loss_weighting must be uniform|min_snr, got {loss_weighting!r}")

        # Symmetric contour Chamfer.  fp32 avoids half-precision cdist issues.
        d = torch.cdist(geom_points.float(), target_points.float(), p=2)
        pred_near = d.min(dim=2).values
        gt_near = d.min(dim=1).values
        loss_chamfer = 0.5 * (pred_near.mean() + gt_near.mean())

        # A small worst-segment term prevents a high average Dice from hiding a
        # visibly wrong local boundary (the main V2 failure).  Mean top-k is much
        # stabler than optimizing the single Hausdorff maximum.
        if lambda_hd > 0:
            frac = min(max(float(hd_fraction), 1.0 / max(geom_points.shape[1], 1)), 1.0)
            k_pred = max(1, int(round(pred_near.shape[1] * frac)))
            k_gt = max(1, int(round(gt_near.shape[1] * frac)))
            loss_hd = 0.5 * (
                pred_near.topk(k_pred, dim=1).values.mean()
                + gt_near.topk(k_gt, dim=1).values.mean()
            )
        else:
            loss_hd = predicted_z0.new_zeros(())

        # V3.1: focus a small extra penalty only on the worst *predicted*
        # vertices. These are the outliers that form long polygon spikes while
        # average Dice/Chamfer can still look good. Keep the weight small.
        if lambda_hard_boundary > 0:
            frac_h = min(max(float(hard_boundary_fraction),
                             1.0 / max(pred_near.shape[1], 1)), 1.0)
            k_h = max(1, int(round(pred_near.shape[1] * frac_h)))
            loss_hard_boundary = pred_near.topk(k_h, dim=1).values.mean()
        else:
            loss_hard_boundary = predicted_z0.new_zeros(())

        # Match local turning/curvature after phase-invariant cyclic alignment.
        # Unlike a smoothness penalty this encourages the *GT irregularities*
        # (concavities and corners) instead of rounding them away.
        if lambda_curvature > 0:
            aligned_gt = best_cyclic_alignment(geom_points.float(), target_points.float(), allow_reverse=True)
            pred_curv = (
                torch.roll(geom_points.float(), 1, dims=1)
                - 2.0 * geom_points.float()
                + torch.roll(geom_points.float(), -1, dims=1)
            )
            gt_curv = (
                torch.roll(aligned_gt, 1, dims=1)
                - 2.0 * aligned_gt
                + torch.roll(aligned_gt, -1, dims=1)
            )
            loss_curvature = F.smooth_l1_loss(pred_curv, gt_curv, beta=0.01)
        else:
            loss_curvature = predicted_z0.new_zeros(())

        # Match adjacent edge lengths against the phase-aligned GT.  Since GT is
        # arc-length resampled, this is a strong anti-collapse / anti-clumping term.
        pred_edges = torch.norm(geom_points - torch.roll(geom_points, -1, dims=1), dim=-1)
        gt_edges = torch.norm(
            target_points - torch.roll(target_points, -1, dims=1), dim=-1
        )
        # GT points are arc-length resampled, so matching the target mean step is
        # phase invariant while still preventing clumping/collapse.
        target_step = gt_edges.mean(dim=1, keepdim=True)
        loss_edge = F.smooth_l1_loss(pred_edges, target_step.expand_as(pred_edges))

        # Optional legacy uniformity term, normalized by target spacing so its
        # scale is comparable across lesion sizes.  Edge matching is the primary
        # regularizer and should normally be kept on.
        if lambda_uniformity > 0:
            target_step_u = gt_edges.mean(dim=1, keepdim=True).detach().clamp(min=1e-4)
            loss_uniformity = ((pred_edges - pred_edges.mean(dim=1, keepdim=True))
                               / target_step_u).pow(2).mean()
        else:
            loss_uniformity = predicted_z0.new_zeros(())

        if masks is not None and (lambda_dice > 0 or lambda_boundary_band > 0):
            loss_dice, loss_boundary_band = soft_mask_boundary_losses(
                geom_points, masks, size=soft_dice_size,
                band_width=boundary_band_width,
            )
        else:
            loss_dice = predicted_z0.new_zeros(())
            loss_boundary_band = predicted_z0.new_zeros(())

        total = (
            loss_x0
            + float(lambda_chamfer) * loss_chamfer
            + float(lambda_edge) * loss_edge
            + float(lambda_uniformity) * loss_uniformity
            + float(lambda_dice) * loss_dice
            + float(lambda_boundary_band) * loss_boundary_band
            + float(lambda_hd) * loss_hd
            + float(lambda_curvature) * loss_curvature
            + float(lambda_hard_boundary) * loss_hard_boundary
        )
        return total, {
            "loss_x0": loss_x0.detach(),
            "loss_chamfer": loss_chamfer.detach(),
            "loss_edge": loss_edge.detach(),
            "loss_uniformity": loss_uniformity.detach(),
            "soft_dice_loss": loss_dice.detach(),
            "boundary_band_loss": loss_boundary_band.detach(),
            "loss_hd": loss_hd.detach(),
            "loss_curvature": loss_curvature.detach(),
            "loss_hard_boundary": loss_hard_boundary.detach(),
        }

    # --- sampling -----------------------------------------------------------

    @torch.no_grad()
    def ddim_sample(
        self,
        denoise_fn,
        cond_fn,
        shape,
        ddim_steps=50,
        guidance_scale=1.0,
        latent_clamp=4.0,
        clamp=None,  # backwards-compatible alias
        proposal_points=None,
        residual_scale=1.0,
        generator=None,
    ):
        """Deterministic DDIM in latent contour space; returns bounded points."""
        if clamp is not None:
            latent_clamp = clamp
        device = self.device
        residual_scale = float(residual_scale)
        if residual_scale <= 0:
            raise ValueError("residual_scale must be positive")
        z = torch.randn(shape, device=device, generator=generator)

        ddim_steps = max(1, min(int(ddim_steps), self.timesteps))
        # linspace guarantees exactly ddim_steps distinct indices for common T.
        ts = torch.linspace(0, self.timesteps - 1, ddim_steps, device=device)
        timesteps = torch.unique(ts.round().long(), sorted=True).flip(0).tolist()

        for i, t in enumerate(timesteps):
            t_b = torch.full((shape[0],), int(t), device=device, dtype=torch.long)
            cond_maps = cond_fn(t_b)
            z0_cond = denoise_fn(z, t_b, cond_maps)

            if guidance_scale != 1.0:
                if isinstance(cond_maps, dict):
                    null_maps = {
                        k: (v if (proposal_points is not None and k == "proposal_points")
                            else ([torch.zeros_like(x) for x in v] if isinstance(v, list)
                                  else torch.zeros_like(v)))
                        for k, v in cond_maps.items()
                    }
                else:
                    null_maps = [torch.zeros_like(m) for m in cond_maps]
                z0_uncond = denoise_fn(z, t_b, null_maps)
                z0 = z0_uncond + float(guidance_scale) * (z0_cond - z0_uncond)
            else:
                z0 = z0_cond
            z0 = z0.clamp(-float(latent_clamp), float(latent_clamp))

            a_bar = self.alphas_cumprod[int(t)]
            t_prev = timesteps[i + 1] if i + 1 < len(timesteps) else -1
            a_bar_prev = (
                self.alphas_cumprod[int(t_prev)]
                if t_prev >= 0 else torch.tensor(1.0, device=device)
            )
            eps = (z - torch.sqrt(a_bar) * z0) / torch.sqrt(1.0 - a_bar).clamp(min=1e-8)
            z = torch.sqrt(a_bar_prev) * z0 + torch.sqrt(1.0 - a_bar_prev) * eps

        if proposal_points is not None:
            return (proposal_points + residual_scale * z).clamp(-0.999, 0.999)
        return latent_to_points(z)
