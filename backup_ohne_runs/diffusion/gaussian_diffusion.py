"""Gaussian diffusion on point coordinates (x0-prediction parameterization).

The denoiser predicts the clean points x0 directly. Training minimizes an
x0 MSE (down-weighted at high noise levels via alpha_bar) plus a uniformity
regularizer that keeps the predicted points evenly spaced around the closed
contour. Sampling uses deterministic DDIM with classifier-free guidance.

V5: optionally, a deterministic coarse-contour proposal (see `proposal.py`)
supplies global position/size/coarse-shape, and the denoiser only predicts
the residual on top of it (`proposal_target="residual"`). This is a
*different* residual from `ContourDenoiser.predict_residual` (which is a
points-vs-noise residual inside the denoiser itself) -- the proposal
combination happens one level up, applied identically to the reconstructed
x0 in both `training_losses` (via the caller, see train.py) and here in
`ddim_sample`.

V6: `ddim_sample` optionally applies the `BoundarySnapper` INSIDE the
sampling loop instead of (or in addition to) as a one-shot post-hoc step
after DDIM finishes. This matches the V5.2 README's description of the
snapper as a "low-t only" correction that lives inside low-noise diffusion,
rather than the standalone post-DDIM refiner in `snapper.py`'s own
docstring. See `ddim_sample`'s `snapper` / `snap_t_threshold` /
`snap_every` args below for the exact gating rule.
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
    """Asymmetrischer Nearest-Neighbor-Loss (pred -> GT).
    Für jeden vorhergesagten Punkt wird die Distanz zum nächstgelegenen
    Ground-Truth-Konturpunkt bestimmt (kein 1:1-Korrespondenz-Zwang, im
    Gegensatz zur normalen x0-MSE). Zeitgewichtung: bei t=0 maximal (=1),
    fällt zu t=T-1 auf ~0 ab.
    predicted_x0, x0: [B, N, 2]
    t: [B] (long, timestep indices)
    """
    dists = torch.cdist(predicted_x0, x0, p=2)          # [B, N_pred, N_gt]
    min_dists, _ = dists.min(dim=-1)                    # [B, N_pred]
    per_sample = min_dists.mean(dim=-1)                 # [B]

    w_t = (1.0 - t.float() / (T - 1)).clamp(min=0.0, max=1.0) ** gamma  # [B]
    loss = (w_t * per_sample).mean()
    return loss
    

def curvature_penalty_loss(predicted_x0, t, T, gamma=1.0):
    """Bestraft hohe lokale Krümmung (zweite Differenz entlang der
    geschlossenen Kontur) der VORHERGESAGTEN Punkte selbst -- kein
    Vergleich gegen GT, reine Formvorgabe an das Modell.

    Zeitgewichtet wie loss_dice (stark bei hohem t, schwach bei niedrigem
    t): zwingt das Modell bei starkem Rauschen zu einer glatten,
    niederfrequenten Kontur -- das ist alles, was aus einer nur grob
    aufgeloesten frühen Scale (z.B. 7x7) ueberhaupt ableitbar ist -- und
    laesst diesen Zwang zum Ende der Sampling-Kette hin nach, wo feinere
    Scales tatsaechlich lokale Information liefern koennen.

    Komplementaer zu nearest_point_loss/loss_biou (die GENAUIGKEIT ggü.
    GT bei niedrigem t einfordern): dieser Term fordert reine GLATTHEIT
    bei hohem t, unabhaengig davon, wo GT tatsaechlich liegt.

    predicted_x0: [B, N, 2]
    t: [B] (long, timestep indices)
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
                        lambda_curvature=0.0, curvature_gamma=1.0,   # NEW
                        soft_dice_size=64):
        """Per-sample min-SNR-weighted x0 MSE + uniformity + soft-Dice +
        Boundary-IoU + nearest-point term.

        `predicted_x0` here is expected to already be the FINAL x0 candidate
        -- i.e. if you're using the V5 proposal, the caller has already
        combined the denoiser's raw output with the proposal via
        `apply_proposal_target` before calling this. This function itself
        stays proposal-agnostic; it just compares whatever x0 it's given
        against GT.

        masks (optional, [B,1,H,W] in {0,1}) enables the differentiable soft-Dice
        and Boundary-IoU terms, which give the boundary points a geometry-aware
        (mask-level) signal instead of relying solely on order-dependent
        coordinate MSE.

        lambda_biou (default 1.0): weight for the differentiable Boundary IoU
        loss (Sun et al., MICCAI 2023) between the rasterized predicted polygon
        and the GT mask, focused on the boundary band rather than the whole
        shape. Requires `masks`, like loss_dice. Time-weighted the OPPOSITE
        direction from loss_dice: strongest at low t (clean, fine boundary
        detail matters most), fades to 0 at high t (pure noise, only coarse
        shape matters -- that's loss_dice's job).

        lambda_nearest (default 1.0): weight for the asymmetric nearest-neighbor
        term (pred -> GT). Time-weighted like loss_biou (strongest at low t).

        Every auxiliary term (uniformity, boundary, dice, biou, nearest) is only
        computed when its lambda is > 0 -- both to save compute and so the
        logged/printed value reflects what's actually contributing to `total`
        (a lambda of 0.0 now always shows 0.0000 instead of a nonzero number
        that has no effect on the gradient).

        lambda_curvature (default 0.0, aus, altes Verhalten): Gewicht für
        einen reinen Selbst-Constraint auf die Kontur-Glattheit der
        Vorhersage (kein GT-Vergleich). Zeitgewichtet wie loss_dice
        (stark bei hohem t, schwach bei niedrigem t) -- siehe
        curvature_penalty_loss's Docstring.
        """
        # Per-sample min-SNR-gamma weighting (x0 parameterization). SNR = a_bar /
        # (1 - a_bar); clamping at gamma stops low-noise steps from dominating, and
        # -- unlike the old a_bar weighting -- high-noise steps keep real gradient.
        if lambda_mse > 0:
            a_bar = _extract(self.alphas_cumprod, t, x0.shape)            # [B,1,1]
            snr = a_bar / (1.0 - a_bar).clamp(min=1e-8)
            w = snr.clamp(max=snr_gamma)
            w = w / w.mean().clamp(min=1e-8)                             # keep scale ~1
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

        # Closed-contour uniformity: penalize variance of neighbor distances.
        if lambda_uniformity > 0:
            nxt = torch.roll(predicted_x0, shifts=-1, dims=1)
            dists = torch.norm(predicted_x0 - nxt, dim=-1)        # [B, N]
            loss_uniformity = dists.std(dim=1).mean()
        else:
            loss_uniformity = torch.zeros((), device=x0.device)

        # Differentiable mask-level term (rasterize polygon -> soft-Dice).
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

        denoise_fn(x, t, cond_maps) -> predicted x0 (residual-on-proposal or
        full x0, depending on `proposal_target` -- this function only
        combines them; the denoiser itself is unaware of the proposal).
        cond_fn(t) -> list of condition maps for timestep t (may be time-dependent);
        the unconditional pass uses zeroed maps.

        proposal: [B,N,2] deterministic coarse-contour proposal (V5, see
        `proposal.py`), or None. Computed once by the caller (it doesn't
        depend on t), applied at every sampling step here.
        proposal_target: "absolute" (proposal ignored, pre-V5 behavior) or
        "residual" (x0 = proposal + denoiser_out at every step).

        Note: the proposal is added ONCE per step, AFTER the CFG
        interpolation between x0_cond and x0_uncond -- not inside each
        branch separately. This matches training, where `pred_x0` (the
        analogue of x0_cond there, after the CFG-dropout masking) is
        combined with the proposal exactly once.

        --- V6: optional in-loop boundary snapping -----------------------
        snapper: optional `BoundarySnapper` instance. When given (together
            with `snap_maps`), applied to the per-step x0 estimate DURING
            sampling, gated to low-noise steps only -- matching the V5.2
            README's description of the exact-boundary correction as a
            "low-t only" branch that lives inside the diffusion process,
            as opposed to `snapper.py`'s own standalone post-DDIM usage
            (still available separately; this is an alternative, not a
            replacement -- see `evaluate()` in sample.py for how the two
            compose).
        snap_maps: the RAW backbone feature maps (same list passed to the
            denoiser's conditioning path via `encoder.extract`, NOT the
            fused `cond_maps` from `cond_fn`) -- required if `snapper` is
            given, since `BoundarySnapper` samples raw multi-scale features
            directly (see snapper.py's docstring). Computed once by the
            caller, like `proposal`.
        snap_image: optional raw image tensor for the snapper's RGB sampling
            path (`BoundarySnapper(use_rgb=True)`); pass None to let the
            snapper skip its RGB term (harmless -- it's additive).
        snap_t_threshold: apply the snapper only at timesteps `t <=
            snap_t_threshold` (i.e. the tail of sampling, closest to clean).
            `None` disables in-loop snapping entirely regardless of whether
            `snapper` is provided, so passing a snapper without a threshold
            is a deliberate, explicit no-op rather than an implicit
            always-on behavior. A typical value is a small fraction of
            `self.timesteps`, e.g. `int(0.15 * timesteps)`, mirroring the
            V5.2 README's "low-t only" framing.
        snap_every: only run the snapper every `snap_every`-th eligible
            DDIM step (within the low-t window) rather than on all of them,
            trading correction frequency for compute -- since each snap call
            costs the snapper's own forward pass at every one of `n_points`
            per sample. `1` (default) snaps every eligible step.

        collect_every: if > 0, the x0 estimate at every `collect_every`-th DDIM
        step (plus always the very last step) is stored and returned as a
        second value: `(x, intermediates)` where `intermediates` is a
        Python list of tensors [B,N,2] (detached, still on `device`), in
        sampling order (noisy/coarse -> clean). If `collect_every <= 0`
        (default), behavior is UNCHANGED: only `x` is returned, so all
        existing call sites keep working without modification.
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

            if do_snap and t <= snap_t_threshold:
                if snap_call_idx % max(1, int(snap_every)) == 0:
                    x0 = snapper(x0, snap_maps, image=snap_image, hard=True)
                    x0 = torch.clamp(x0, -clamp, clamp)
                snap_call_idx += 1

            # --- NEW: stash this step's x0 estimate for the GIF -------------
            # Stored AFTER proposal/clamp/snap, i.e. exactly the x0 this step
            # actually commits to -- the same quantity the DDIM update below
            # uses, so the GIF shows the real denoising trajectory rather than
            # a pre-correction intermediate.
            if do_collect:
                is_last = (i == len(timesteps) - 1)
                if is_last or (i % collect_every == 0):
                    intermediates.append(x0.detach().clone())

            a_bar = self.alphas_cumprod[t]
            t_prev = timesteps[i + 1] if i + 1 < len(timesteps) else -1
            a_bar_prev = self.alphas_cumprod[t_prev] if t_prev >= 0 else torch.tensor(1.0, device=device)

            direction = (x - torch.sqrt(a_bar) * x0) / torch.sqrt(1.0 - a_bar)
            x = torch.sqrt(a_bar_prev) * x0 + torch.sqrt(1.0 - a_bar_prev) * direction

        if do_collect:
            return x, intermediates
        return x