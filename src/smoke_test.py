"""CPU smoke test for P2SDiff V5.2; no dataset/timm required."""

import math
import torch

from .diffusion import GaussianDiffusion, best_cyclic_alignment
from .models import ContourDenoiser, UNetConditioner
from .models.boundary_refiner import perturb_along_normals
from .utils.rasterize import soft_rasterize


def main():
    torch.manual_seed(0)
    b, n, size = 2, 32, 64
    encoder = UNetConditioner(in_channels=3, start_dim=16, dim_mults=(1, 2, 4), groups=8)
    denoiser = ContourDenoiser(
        n_points=n, hidden_dim=64, num_layers=2, num_heads=4,
        scale_channels=encoder.feature_channels, timesteps=100,
        deformable_samples=5, global_levels=3, global_grid=8,
        proposal_type="fourier", diffusion_target="residual", fourier_harmonics=4,
        exact_boundary_enabled=True, exact_boundary_levels=2,
        exact_boundary_samples=9, exact_boundary_radius=0.10,
        exact_boundary_profile_dim=16, exact_boundary_hidden=32,
        exact_boundary_low_t_fraction=0.35,
    )
    diffusion = GaussianDiffusion(timesteps=100, device="cpu")

    images = torch.randn(b, 3, size, size)
    phi = -0.5 * math.pi + 2.0 * math.pi * torch.arange(n) / n
    gt = torch.stack([0.45 * phi.cos(), 0.30 * phi.sin()], dim=-1)[None].repeat(b, 1, 1)
    gt[:, :, 0] += 0.05 * torch.sin(3 * phi)[None]
    masks = (soft_rasterize(gt, size=size) > 0.5).float().unsqueeze(1)
    t = torch.tensor([5, 20])  # both exercise the low-t exact branch

    maps = encoder.extract(images)
    cond = denoiser.prepare_condition(maps, image=images)
    proposal = denoiser.proposal_points(cond)
    aligned_gt = best_cyclic_alignment(proposal.detach(), gt, allow_reverse=True)
    target_state = aligned_gt - proposal.detach()
    zt = diffusion.q_sample_state(target_state, t)
    pred_z0 = denoiser(zt, t, cond)
    pred_points = (proposal + pred_z0).clamp(-0.999, 0.999)

    loss_diff, parts = diffusion.training_losses(
        pred_z0, gt, t, masks=masks, geometry_points=pred_points,
        lambda_dice=0.2, lambda_boundary_band=0.2,
        lambda_hd=0.1, hd_fraction=0.25, lambda_curvature=0.05,
        lambda_hard_boundary=0.08, hard_boundary_fraction=0.20,
        soft_dice_size=48, target_state=target_state, predicted_points=pred_points,
    )
    teacher = perturb_along_normals(gt, max_offset=0.06, smooth_passes=2)
    exact_off, exact_conf, exact_parts = denoiser.exact_boundary_loss(
        gt, cond, teacher_points=teacher,
    )
    loss = loss_diff + exact_off + 0.5 * exact_conf
    loss.backward()

    # Critical gradient checks.
    assert denoiser.exact_corrector.offset_head[-1].weight.grad is not None
    assert denoiser.exact_corrector.conf_head[-1].weight.grad is not None
    assert denoiser.exact_corrector.sample_proj[0][0].weight.grad is not None

    with torch.no_grad():
        encoder.eval(); denoiser.eval()
        cond_eval = denoiser.prepare_condition(encoder.extract(images), image=images)
        proposal_eval = denoiser.proposal_points(cond_eval)
        pred = diffusion.ddim_sample(
            denoiser, lambda _t: cond_eval, (b, n, 2), ddim_steps=5,
            guidance_scale=1.0, proposal_points=proposal_eval, residual_scale=1.0,
        )

    print("P2SDiff V5.2 smoke test: OK")
    print("feature pyramid:", [tuple(m.shape) for m in maps])
    print("dense boundary head active: False")
    print("post snapper active: False")
    print("train loss:", float(loss.detach()))
    print("sample shape/range:", tuple(pred.shape), float(pred.min()), float(pred.max()))
    print("exact parts:", {k: float(v) for k, v in exact_parts.items()})
    print("stats:", {k: float(v) for k, v in denoiser.last_stats.items() if torch.is_tensor(v)})


if __name__ == "__main__":
    main()
