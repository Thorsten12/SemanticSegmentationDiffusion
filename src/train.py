"""Train P2SDiff V5.2 confidence-gated exact-boundary contour diffusion.

Examples are in the repository README for from-scratch U-Net-encoder training
and full fine-tuning of ImageNet-pretrained PVT-v2.
"""

import argparse
import json
import math
import os
import traceback

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from .config import Config
from .data import DATASET_NAMES, build_contour_dataset, split_counts
from .diffusion import GaussianDiffusion, best_cyclic_alignment
from .models import ContourDenoiser, build_conditioner
from .models.contour_decoder import stats_to_float
from .models.boundary_refiner import perturb_along_normals
from .utils import EMA
from .utils.rasterize import soft_dice_loss


def str2bool(v):
    if isinstance(v, bool):
        return v
    v = str(v).lower()
    if v in ("1", "true", "t", "yes", "y"):
        return True
    if v in ("0", "false", "f", "no", "n"):
        return False
    raise argparse.ArgumentTypeError(f"Boolean expected, got {v!r}")


def set_boundary_teacher_loss(pred, target, hard_fraction=0.20, hard_weight=0.50):
    """Correspondence-free teacher loss for the snapper.

    Unlike V3's index-wise Smooth-L1, this does not punish a vertex merely for
    landing on the correct boundary at a different contour phase.
    """
    d = torch.cdist(pred.float(), target.float(), p=2)
    pred_near = d.min(dim=2).values
    gt_near = d.min(dim=1).values
    chamfer = 0.5 * (pred_near.mean() + gt_near.mean())
    frac = min(max(float(hard_fraction), 1.0 / max(pred_near.shape[1], 1)), 1.0)
    k = max(1, int(round(pred_near.shape[1] * frac)))
    hard = pred_near.topk(k, dim=1).values.mean()
    return chamfer + float(hard_weight) * hard, chamfer.detach(), hard.detach()


def geometry_snap_alpha(epoch_idx, warmup_epochs, ramp_epochs):
    """0 -> coarse-only geometry; 1 -> V3 snapped geometry."""
    e = int(epoch_idx) + 1
    warm = max(0, int(warmup_epochs))
    ramp = max(0, int(ramp_epochs))
    if e <= warm:
        return 0.0
    if ramp <= 0:
        return 1.0
    return min(1.0, max(0.0, (e - warm) / float(ramp)))


def coarse_proposal_losses(proposal, target, masks, dice_size=96):
    """Direct low-frequency geometry supervision for the spatial proposal."""
    d = torch.cdist(proposal.float(), target.float(), p=2)
    chamfer = 0.5 * (d.min(dim=2).values.mean() + d.min(dim=1).values.mean())
    dice = soft_dice_loss(proposal, masks, size=dice_size)
    return chamfer, dice


def build_models(cfg: Config, device):
    encoder = build_conditioner(cfg).to(device)
    denoiser = ContourDenoiser(
        n_points=cfg.n_points,
        hidden_dim=cfg.hidden_dim,
        num_layers=cfg.n_transformer_layers,
        num_heads=cfg.n_heads,
        scale_channels=encoder.feature_channels,
        coord_fourier_bands=cfg.coord_fourier_bands,
        tag_maps=cfg.tag_maps,
        tag_mode=getattr(cfg, "tag_mode", "concat"),
        timesteps=cfg.timesteps,
        deformable_samples=cfg.deformable_samples,
        local_radius_min=cfg.local_radius_min,
        local_radius_max=cfg.local_radius_max,
        global_levels=cfg.global_levels,
        global_grid=cfg.global_grid,
        contour_phase_bands=cfg.contour_phase_bands,
        boundary_dim=cfg.boundary_dim,
        boundary_levels=cfg.boundary_levels,
        snap_candidates=cfg.snap_candidates,
        snap_iterations=cfg.snap_iterations,
        snap_radius_max=cfg.snap_radius_max,
        snap_radius_min=cfg.snap_radius_min,
        snap_temperature=cfg.snap_temperature,
        snap_profile_radius=cfg.snap_profile_radius,
        proposal_type=cfg.proposal_type,
        diffusion_target=cfg.diffusion_target,
        fourier_harmonics=cfg.fourier_harmonics,
        proposal_arclength_resample=cfg.proposal_arclength_resample,
        residual_scale=cfg.residual_scale,
        exact_boundary_enabled=cfg.exact_boundary_enabled,
        exact_boundary_levels=cfg.exact_boundary_levels,
        exact_boundary_samples=cfg.exact_boundary_samples,
        exact_boundary_radius=cfg.exact_boundary_radius,
        exact_boundary_profile_dim=cfg.exact_boundary_profile_dim,
        exact_boundary_hidden=cfg.exact_boundary_hidden,
        exact_boundary_ring_bands=cfg.exact_boundary_ring_bands,
        exact_boundary_relative_bias=cfg.exact_boundary_relative_bias,
        exact_boundary_confidence_power=cfg.exact_boundary_confidence_power,
        exact_boundary_low_t_fraction=cfg.exact_boundary_low_t_fraction,
        exact_confidence_radius=cfg.exact_confidence_radius,
        exact_tangent_tolerance=cfg.exact_tangent_tolerance,
        exact_use_rgb=cfg.exact_use_rgb,
    ).to(device)
    return encoder, denoiser


def _load_init_checkpoint(path, encoder, denoiser, device):
    """Load matching EMA weights for transfer; skip mismatched keys."""
    ckpt = torch.load(path, map_location=device, weights_only=False)
    for name, module in (("encoder", encoder), ("denoiser", denoiser)):
        if name not in ckpt:
            print(f"  warn: no '{name}' in {path}")
            continue
        missing, unexpected = module.load_state_dict(ckpt[name], strict=False)
        print(f"  loaded {name}: missing={len(missing)} unexpected={len(unexpected)}")


def _source_state_dict(ckpt):
    if isinstance(ckpt.get("encoder"), dict):
        return ckpt["encoder"], "conditioner"
    if isinstance(ckpt.get("model"), dict):
        return ckpt["model"], "pixel"
    return ckpt, "raw"


def _candidate_keys(src_key):
    """Map pixel-baseline / conditioner keys onto `build_conditioner` names."""
    keys = [src_key]
    if src_key.startswith("backbone.encoder."):
        rest = src_key[len("backbone.encoder."):]
        keys.append("unet." + rest)
    if src_key.startswith("unet.encoder."):
        keys.append("unet." + src_key[len("unet.encoder."):])
    return keys


def _load_init_backbone(path, encoder, device):
    """Copy a trained pyramid into the conditioner; leave DP2Seg / stem random.

    Accepts a P2SDiff encoder checkpoint or a pixel baseline (`train_pixel_baseline.py`)
    for PVT (`backbone.*`) and U-Net (`backbone.encoder.*`).
    """
    ckpt = torch.load(path, map_location=device, weights_only=False)
    src, kind = _source_state_dict(ckpt)
    dest = encoder.state_dict()
    mapped = {}
    for k, v in src.items():
        for nk in _candidate_keys(k):
            if nk in dest and dest[nk].shape == v.shape:
                mapped[nk] = v
                break
    missing, unexpected = encoder.load_state_dict(mapped, strict=False)
    n_bb = sum(1 for k in mapped if k.startswith("backbone.") or k.startswith("unet."))
    print(f"  init_backbone ({kind}): {len(mapped)} tensors ({n_bb} backbone) "
          f"missing={len(missing)} unexpected={len(unexpected)}")


class WarmupCosineScheduler:
    """Linear warmup then cosine decay to min_lr (per-epoch steps)."""

    def __init__(self, optimizer, warmup_epochs, total_epochs, min_lr):
        self.optimizer = optimizer
        self.warmup_epochs = max(0, int(warmup_epochs))
        self.total_epochs = max(1, int(total_epochs))
        self.min_lr = float(min_lr)
        self.base_lrs = [g["lr"] for g in optimizer.param_groups]
        self.last_epoch = 0

    def step(self, epoch=None):
        if epoch is None:
            epoch = self.last_epoch + 1
        self.last_epoch = epoch
        for group, base in zip(self.optimizer.param_groups, self.base_lrs):
            if epoch <= self.warmup_epochs and self.warmup_epochs > 0:
                lr = base * epoch / self.warmup_epochs
            else:
                t = epoch - self.warmup_epochs
                t_max = max(1, self.total_epochs - self.warmup_epochs)
                cos = 0.5 * (1.0 + math.cos(math.pi * min(t, t_max) / t_max))
                lr = self.min_lr + (base - self.min_lr) * cos
            group["lr"] = lr

    def get_last_lr(self):
        return [g["lr"] for g in self.optimizer.param_groups]


def main():
    parser = argparse.ArgumentParser(description="Train P2SDiff")
    parser.add_argument("--dataset", choices=list(DATASET_NAMES))
    parser.add_argument("--skin_root", type=str)
    parser.add_argument("--data_root", type=str)
    parser.add_argument("--polyp_test", type=str)
    parser.add_argument("--out_dir", type=str)
    parser.add_argument("--epochs", type=int)
    parser.add_argument("--batch_size", type=int)
    parser.add_argument("--lr", type=float)
    parser.add_argument("--weight_decay", type=float)
    parser.add_argument("--n_points", type=int)
    parser.add_argument("--encoder", choices=["convnext", "pvt", "unet", "timm"])
    parser.add_argument("--backbone", type=str)
    parser.add_argument("--pretrained", type=str2bool, default=None)
    parser.add_argument("--pvt_variant", type=str)
    parser.add_argument("--pvt_pretrained_path", type=str)
    parser.add_argument("--unet_start_dim", type=int)
    parser.add_argument("--freeze_backbone", type=str2bool, default=None)
    parser.add_argument("--backbone_lr", type=float)
    parser.add_argument("--stem_dim", type=int)
    parser.add_argument("--tag_maps", type=str2bool, default=None)
    parser.add_argument("--tag_mode", choices=["concat", "add", "none"])
    parser.add_argument("--hidden_dim", type=int)
    parser.add_argument("--n_transformer_layers", type=int)
    parser.add_argument("--n_heads", type=int)
    parser.add_argument("--coord_fourier_bands", type=int)
    parser.add_argument("--contour_phase_bands", type=int)
    parser.add_argument("--deformable_samples", type=int)
    parser.add_argument("--local_radius_min", type=float)
    parser.add_argument("--local_radius_max", type=float)
    parser.add_argument("--global_levels", type=int)
    parser.add_argument("--global_grid", type=int)
    parser.add_argument("--proposal_type", choices=["ellipse", "fourier"])
    parser.add_argument("--diffusion_target", choices=["absolute", "residual"])
    parser.add_argument("--fourier_harmonics", type=int)
    parser.add_argument("--proposal_arclength_resample", type=str2bool, default=None)
    parser.add_argument("--residual_scale", type=float)
    parser.add_argument("--exact_boundary_enabled", type=str2bool, default=None)
    parser.add_argument("--exact_boundary_levels", type=int)
    parser.add_argument("--exact_boundary_samples", type=int)
    parser.add_argument("--exact_boundary_radius", type=float)
    parser.add_argument("--exact_boundary_profile_dim", type=int)
    parser.add_argument("--exact_boundary_hidden", type=int)
    parser.add_argument("--exact_boundary_ring_bands", type=int)
    parser.add_argument("--exact_boundary_relative_bias", type=float)
    parser.add_argument("--exact_boundary_confidence_power", type=float)
    parser.add_argument("--exact_boundary_low_t_fraction", type=float)
    parser.add_argument("--exact_confidence_radius", type=float)
    parser.add_argument("--exact_tangent_tolerance", type=float)
    parser.add_argument("--exact_use_rgb", type=str2bool, default=None)
    parser.add_argument("--exact_teacher_offset", type=float)
    parser.add_argument("--exact_teacher_smooth", type=int)
    parser.add_argument("--lambda_exact_offset", type=float)
    parser.add_argument("--lambda_exact_confidence", type=float)
    parser.add_argument("--boundary_dim", type=int)
    parser.add_argument("--boundary_levels", type=int)
    parser.add_argument("--boundary_thickness", type=int)
    parser.add_argument("--snap_candidates", type=int)
    parser.add_argument("--snap_iterations", type=int)
    parser.add_argument("--snap_radius_max", type=float)
    parser.add_argument("--snap_radius_min", type=float)
    parser.add_argument("--snap_temperature", type=float)
    parser.add_argument("--snap_hard_inference", type=str2bool, default=None)
    parser.add_argument("--snap_teacher_offset", type=float)
    parser.add_argument("--snap_teacher_smooth", type=int)
    parser.add_argument("--snap_profile_radius", type=float)
    parser.add_argument("--low_t_fraction", type=float)
    parser.add_argument("--low_t_max_fraction", type=float)
    parser.add_argument("--device", type=str)
    parser.add_argument("--num_workers", type=int)
    parser.add_argument("--eval_every", type=int)
    parser.add_argument("--aug_level", choices=["none", "light", "strong"])
    parser.add_argument("--guidance_scale", type=float)
    parser.add_argument("--cfg_dropout", type=float)
    parser.add_argument("--ddim_steps", type=int)
    parser.add_argument("--timesteps", type=int)
    parser.add_argument("--beta_start", type=float)
    parser.add_argument("--beta_end", type=float)
    parser.add_argument("--latent_clamp", type=float)
    parser.add_argument("--lambda_dice", type=float)
    parser.add_argument("--lambda_chamfer", type=float)
    parser.add_argument("--lambda_edge", type=float)
    parser.add_argument("--lambda_uniformity", type=float)
    parser.add_argument("--lambda_boundary_band", type=float)
    parser.add_argument("--lambda_boundary_head", type=float)
    parser.add_argument("--lambda_snap_teacher", type=float)
    parser.add_argument("--lambda_hd", type=float)
    parser.add_argument("--hd_fraction", type=float)
    parser.add_argument("--lambda_curvature", type=float)
    parser.add_argument("--lambda_hard_boundary", type=float)
    parser.add_argument("--hard_boundary_fraction", type=float)
    parser.add_argument("--snap_teacher_hard_weight", type=float)
    parser.add_argument("--snap_teacher_hard_fraction", type=float)
    parser.add_argument("--lambda_proposal_chamfer", type=float)
    parser.add_argument("--lambda_proposal_dice", type=float)
    parser.add_argument("--proposal_dice_size", type=int)
    parser.add_argument("--boundary_band_width", type=int)
    parser.add_argument("--soft_dice_size", type=int)
    parser.add_argument("--snr_gamma", type=float)
    parser.add_argument("--loss_weighting", choices=["uniform", "min_snr"])
    parser.add_argument("--cyclic_reverse", type=str2bool, default=None)
    parser.add_argument("--scheduler", choices=["none", "cosine"])
    parser.add_argument("--warmup_epochs", type=int)
    parser.add_argument("--snap_geometry_warmup_epochs", type=int)
    parser.add_argument("--snap_geometry_ramp_epochs", type=int)
    parser.add_argument("--min_lr", type=float)
    parser.add_argument("--init_checkpoint", type=str)
    parser.add_argument("--init_backbone", type=str)
    parser.add_argument("--seed", type=int)
    parser.add_argument("--eval_seed", type=int)
    parser.add_argument("--no_amp", action="store_true")
    args = parser.parse_args()

    cfg = Config.from_args(args)
    if args.no_amp:
        cfg.amp = False

    torch.manual_seed(cfg.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(cfg.seed)
    device = torch.device(cfg.device if torch.cuda.is_available() else "cpu")
    os.makedirs(cfg.out_dir, exist_ok=True)
    with open(os.path.join(cfg.out_dir, "config.json"), "w") as f:
        json.dump({k: getattr(cfg, k) for k in cfg.__dataclass_fields__}, f, indent=2, default=str)

    # ----- data -----
    counts = split_counts(
        cfg.skin_root, cfg.dataset, cfg.npy_size,
        data_root=cfg.data_root, polyp_test=cfg.polyp_test,
    )
    print(f"Dataset {cfg.dataset} | split -> train {counts['tr']} | "
          f"val {counts['vl']} | test {counts['te']}")
    ds_kwargs = dict(
        n_points=cfg.n_points, img_size=cfg.img_size, npy_size=cfg.npy_size,
        data_root=cfg.data_root, polyp_test=cfg.polyp_test,
    )
    train_ds = build_contour_dataset(
        cfg.skin_root, cfg.dataset, "tr", augment=cfg.augment,
        aug_level=cfg.aug_level, **ds_kwargs,
    )
    val_ds = build_contour_dataset(
        cfg.skin_root, cfg.dataset, "vl", augment=False, **ds_kwargs,
    )
    persist = cfg.num_workers > 0
    train_loader = DataLoader(train_ds, batch_size=cfg.batch_size, shuffle=True,
                              num_workers=cfg.num_workers, drop_last=True, pin_memory=True,
                              persistent_workers=persist)
    val_loader = DataLoader(val_ds, batch_size=cfg.batch_size, shuffle=False,
                            num_workers=cfg.num_workers, persistent_workers=persist)

    # ----- models / diffusion / optim -----
    encoder, denoiser = build_models(cfg, device)
    if cfg.init_backbone:
        print(f"Init backbone from {cfg.init_backbone}")
        _load_init_backbone(cfg.init_backbone, encoder, device)
    if cfg.init_checkpoint:
        print(f"Warm-start from {cfg.init_checkpoint}")
        _load_init_checkpoint(cfg.init_checkpoint, encoder, denoiser, device)

    diffusion = GaussianDiffusion(cfg.timesteps, cfg.beta_start, cfg.beta_end, device=device)
    ema = EMA([encoder, denoiser], decay=cfg.ema_decay)

    # Discriminative LR: a low LR for the pretrained backbone (when fine-tuning),
    # the normal LR for the freshly-initialized fusion + denoiser.
    backbone = getattr(encoder, "backbone", None)
    backbone_ids = {id(p) for p in backbone.parameters()} if backbone is not None else set()
    backbone_params, head_params = [], []
    for p in encoder.parameters():
        if not p.requires_grad:
            continue
        (backbone_params if id(p) in backbone_ids else head_params).append(p)
    head_params += [p for p in denoiser.parameters() if p.requires_grad]

    param_groups = [{"params": head_params, "lr": cfg.lr}]
    if backbone_params:
        param_groups.append({"params": backbone_params, "lr": cfg.backbone_lr})
    optimizer = torch.optim.AdamW(param_groups, lr=cfg.lr, weight_decay=cfg.weight_decay)
    scheduler = None
    if cfg.scheduler == "cosine":
        scheduler = WarmupCosineScheduler(
            optimizer, warmup_epochs=cfg.warmup_epochs,
            total_epochs=cfg.epochs, min_lr=cfg.min_lr,
        )
    trainable = head_params + backbone_params
    use_amp = cfg.amp and device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

    n_train = sum(p.numel() for p in trainable)
    n_total = sum(p.numel() for p in encoder.parameters()) + sum(p.numel() for p in denoiser.parameters())
    bb_state = "frozen" if not backbone_params else f"fine-tune @ lr {cfg.backbone_lr:g}"
    enc_name = f"{cfg.encoder}/{cfg.pvt_variant}" if cfg.encoder == "pvt" else f"{cfg.encoder}/{cfg.backbone}"
    print(f"Encoder: {enc_name} (backbone {bb_state}) | "
          f"proposal {cfg.proposal_type} | diffusion {cfg.diffusion_target} | "
          f"arc-resample {cfg.proposal_arclength_resample} | residual-scale {cfg.residual_scale:g} | "
          f"V5.2 sparse exact-boundary-in-diffusion | "
          f"trainable {n_train/1e6:.2f}M / total {n_total/1e6:.2f}M | "
          f"device {device} | amp {use_amp} | sched {cfg.scheduler}")

    history = {
        "epoch": [], "loss": [], "loss_x0": [], "loss_chamfer": [], "loss_edge": [],
        "soft_dice_loss": [], "loss_uniformity": [],
        "boundary_band_loss": [], "boundary_head_loss": [], "snap_teacher_loss": [],
        "snap_teacher_chamfer": [], "snap_teacher_hard": [],
        "loss_hd": [], "loss_curvature": [], "loss_hard_boundary": [],
        "proposal_chamfer": [], "proposal_dice_loss": [],
        "exact_loss_offset": [], "exact_loss_conf": [],
        "exact_target_conf_rate": [], "exact_teacher_conf_rate": [],
        "snap_geometry_alpha": [],
        "lr": [],
        "val_epoch": [], "val_dice": [], "val_iou": [],
        "dp2seg": [],
    }
    best_val = -1.0
    stats_path = os.path.join(cfg.out_dir, "dp2seg_stats.jsonl")

    for epoch in range(cfg.epochs):
        if scheduler is not None:
            scheduler.step(epoch + 1)
        cur_lr = optimizer.param_groups[0]["lr"]
        bb_lr = optimizer.param_groups[-1]["lr"] if len(optimizer.param_groups) > 1 else cur_lr

        encoder.train(); denoiser.train()
        running = 0.0
        running_x0 = running_chamfer = running_edge = running_dice = running_unif = 0.0
        running_band = running_bhead = running_snap = 0.0
        running_proposal_ch = running_proposal_dice = 0.0
        running_hd = running_curv = running_hardb = 0.0
        running_teacher_ch = running_teacher_hard = 0.0
        running_exact_off = running_exact_conf = 0.0
        running_exact_rate = running_exact_teacher_rate = 0.0
        snap_alpha = geometry_snap_alpha(
            epoch, cfg.snap_geometry_warmup_epochs, cfg.snap_geometry_ramp_epochs
        )
        epoch_dp = []
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{cfg.epochs}")
        for step, (images, points, masks) in enumerate(pbar):
            images = images.to(device, non_blocking=True)
            points = points.to(device, non_blocking=True)
            masks = masks.to(device, non_blocking=True)
            b = images.shape[0]

            optimizer.zero_grad(set_to_none=True)
            # Mixed timestep curriculum: preserve the full diffusion range, but
            # deliberately show the denoiser more low-noise examples where exact
            # boundary alignment is learned. This changes training frequency only;
            # the forward process and DDIM equations remain unchanged.
            t = torch.randint(0, cfg.timesteps, (b,), device=device).long()
            if cfg.low_t_fraction > 0:
                use_low = torch.rand(b, device=device) < float(cfg.low_t_fraction)
                low_max = max(1, int(cfg.timesteps * float(cfg.low_t_max_fraction)))
                n_low = int(use_low.sum().item())
                if n_low:
                    t[use_low] = torch.randint(0, low_max, (n_low,), device=device)

            with torch.amp.autocast("cuda", enabled=use_amp):
                raw_maps = encoder.extract(images)        # backbone once
                full_cond = denoiser.prepare_condition(raw_maps, image=images)  # sparse condition cache
                proposal = denoiser.proposal_points(full_cond)
                if cfg.diffusion_target == "residual":
                    # Keep the stochastic target stable; proposal has its own
                    # direct loss and still receives geometry gradients later.
                    # Fourier parameterization is not arc-length/cyclic-index
                    # invariant, so align GT phase before defining pointwise
                    # residuals. Polygon geometry remains unchanged.
                    aligned_points = best_cyclic_alignment(
                        proposal.detach(), points, allow_reverse=cfg.cyclic_reverse,
                    )
                    target_state = (
                        aligned_points - proposal.detach()
                    ) / float(cfg.residual_scale)
                    noisy = diffusion.q_sample_state(target_state, t)
                else:
                    target_state = None
                    noisy = diffusion.q_sample(points, t)

                loss_bhead = proposal.new_zeros(())
                bhead_parts = {}

                if cfg.cfg_dropout > 0:
                    keep = (torch.rand(b, device=device) >= cfg.cfg_dropout)
                    keep = keep.to(raw_maps[0].dtype).view(b, 1, 1, 1)
                    cond = denoiser.drop_condition(full_cond, keep)
                else:
                    cond = full_cond

                pred_z0 = denoiser(noisy, t, cond)
                pred_z0 = torch.clamp(pred_z0, -cfg.latent_clamp, cfg.latent_clamp)
                if cfg.diffusion_target == "residual":
                    coarse_points = (
                        proposal + float(cfg.residual_scale) * pred_z0
                    ).clamp(-0.999, 0.999)
                else:
                    coarse_points = torch.tanh(pred_z0)

                # V5.2 has no post-snapper.  `pred_z0` already includes the
                # confidence-gated exact local correction at low timesteps.
                geometry_points = coarse_points
                loss_diff, parts = diffusion.training_losses(
                    pred_z0, points, t, masks=masks,
                    lambda_uniformity=cfg.lambda_uniformity,
                    lambda_dice=cfg.lambda_dice,
                    lambda_chamfer=cfg.lambda_chamfer,
                    lambda_edge=cfg.lambda_edge,
                    snr_gamma=cfg.snr_gamma,
                    loss_weighting=cfg.loss_weighting,
                    soft_dice_size=cfg.soft_dice_size,
                    cyclic_reverse=cfg.cyclic_reverse,
                    geometry_points=geometry_points,
                    lambda_boundary_band=cfg.lambda_boundary_band,
                    boundary_band_width=cfg.boundary_band_width,
                    lambda_hd=cfg.lambda_hd,
                    hd_fraction=cfg.hd_fraction,
                    lambda_curvature=cfg.lambda_curvature,
                    lambda_hard_boundary=cfg.lambda_hard_boundary,
                    hard_boundary_fraction=cfg.hard_boundary_fraction,
                    target_state=target_state,
                    predicted_points=coarse_points,
                )

                if cfg.lambda_proposal_chamfer > 0 or cfg.lambda_proposal_dice > 0:
                    proposal_chamfer, proposal_dice = coarse_proposal_losses(
                        proposal, points, masks, dice_size=cfg.proposal_dice_size,
                    )
                else:
                    proposal_chamfer = pred_z0.new_zeros(())
                    proposal_dice = pred_z0.new_zeros(())

                # Confidence cannot collapse to "uncertain everywhere": a close
                # teacher contour supplies abundant positive exact-boundary examples,
                # while the cached on-policy branch supervises real diffusion errors.
                teacher_in = perturb_along_normals(
                    points, max_offset=cfg.exact_teacher_offset,
                    smooth_passes=cfg.exact_teacher_smooth,
                ) if cfg.exact_boundary_enabled else None
                exact_off_loss, exact_conf_loss, exact_parts = denoiser.exact_boundary_loss(
                    points, full_cond, teacher_points=teacher_in,
                )
                loss_snap_teacher = pred_z0.new_zeros(())
                teacher_chamfer = pred_z0.new_zeros(())
                teacher_hard = pred_z0.new_zeros(())

                loss = (
                    loss_diff
                    + cfg.lambda_proposal_chamfer * proposal_chamfer
                    + cfg.lambda_proposal_dice * proposal_dice
                    + cfg.lambda_exact_offset * exact_off_loss
                    + cfg.lambda_exact_confidence * exact_conf_loss
                )

            scaler.scale(loss).backward()
            if cfg.grad_clip:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(trainable, cfg.grad_clip)
            scaler.step(optimizer)
            scaler.update()
            ema.update([encoder, denoiser])

            loss_x0 = float(parts["loss_x0"].item())
            loss_chamfer = float(parts["loss_chamfer"].item())
            loss_edge = float(parts["loss_edge"].item())
            soft_dice_loss = float(parts["soft_dice_loss"].item())
            loss_unif = float(parts["loss_uniformity"].item())
            loss_band = float(parts["boundary_band_loss"].item())
            loss_bhead_v = float(loss_bhead.detach().item())
            loss_snap_v = float(loss_snap_teacher.detach().item())
            loss_hd_v = float(parts["loss_hd"].item())
            loss_curv_v = float(parts["loss_curvature"].item())
            loss_hardb_v = float(parts["loss_hard_boundary"].item())
            teacher_ch_v = float(teacher_chamfer.item())
            teacher_hard_v = float(teacher_hard.item())
            proposal_ch_v = float(proposal_chamfer.detach().item())
            proposal_dice_v = float(proposal_dice.detach().item())
            exact_off_v = float(exact_parts["exact_loss_offset"].item())
            exact_conf_v = float(exact_parts["exact_loss_conf"].item())
            exact_rate_v = float(exact_parts["exact_target_conf_rate"].item())
            exact_teacher_rate_v = float(exact_parts["exact_teacher_conf_rate"].item())
            running += loss.item()
            running_x0 += loss_x0
            running_chamfer += loss_chamfer
            running_edge += loss_edge
            running_dice += soft_dice_loss
            running_unif += loss_unif
            running_band += loss_band
            running_bhead += loss_bhead_v
            running_snap += loss_snap_v
            running_hd += loss_hd_v
            running_curv += loss_curv_v
            running_hardb += loss_hardb_v
            running_teacher_ch += teacher_ch_v
            running_teacher_hard += teacher_hard_v
            running_proposal_ch += proposal_ch_v
            running_proposal_dice += proposal_dice_v
            running_exact_off += exact_off_v
            running_exact_conf += exact_conf_v
            running_exact_rate += exact_rate_v
            running_exact_teacher_rate += exact_teacher_rate_v
            dp = stats_to_float(getattr(denoiser, "last_stats", {}) or {})
            record = {
                "epoch": epoch + 1, "step": step,
                "loss": float(loss.item()),
                "loss_x0": loss_x0,
                "loss_chamfer": loss_chamfer,
                "loss_edge": loss_edge,
                "soft_dice_loss": soft_dice_loss,
                "loss_uniformity": loss_unif,
                "boundary_band_loss": loss_band,
                "boundary_head_loss": loss_bhead_v,
                "snap_teacher_loss": loss_snap_v,
                "snap_teacher_chamfer": teacher_ch_v,
                "snap_teacher_hard": teacher_hard_v,
                "loss_hd": loss_hd_v,
                "loss_curvature": loss_curv_v,
                "loss_hard_boundary": loss_hardb_v,
                "proposal_chamfer": proposal_ch_v,
                "proposal_dice_loss": proposal_dice_v,
                "exact_loss_offset": exact_off_v,
                "exact_loss_conf": exact_conf_v,
                "exact_target_conf_rate": exact_rate_v,
                "exact_teacher_conf_rate": exact_teacher_rate_v,
                "snap_geometry_alpha": 0.0,
                "boundary_bce": 0.0,
                "boundary_dice_loss": 0.0,
                "t_batch": float(t.float().mean().item()) / max(cfg.timesteps - 1, 1),
            }
            if dp:
                record.update(dp)
                epoch_dp.append(record)
            with open(stats_path, "a") as f:
                f.write(json.dumps(record) + "\n")
            postfix = {"loss": f"{loss.item():.4f}",
                       "x0": f"{loss_x0:.4f}",
                       "ch": f"{loss_chamfer:.3f}",
                       "edge": f"{loss_edge:.3f}",
                       "dice_l": f"{soft_dice_loss:.4f}",
                       "band": f"{loss_band:.3f}",
                       "exOff": f"{exact_off_v:.3f}",
                       "exCf": f"{exact_conf_v:.3f}",
                       "hd": f"{loss_hd_v:.3f}",
                       "lr": f"{cur_lr:.2e}"}
            if dp:
                postfix["gG"] = f"{dp.get('global_gate', 0):.2f}"
                postfix["gL"] = f"{dp.get('local_gate', 0):.2f}"
                postfix["off"] = f"{dp.get('offset_abs', 0):.3f}"
                postfix["exd"] = f"{dp.get('exact_applied_abs', 0):.3f}"
                postfix["exc"] = f"{dp.get('exact_conf_mean', 0):.2f}"
            pbar.set_postfix(postfix)

        n_steps = max(1, len(train_loader))
        avg = running / n_steps
        avg_x0 = running_x0 / n_steps
        avg_chamfer = running_chamfer / n_steps
        avg_edge = running_edge / n_steps
        avg_dice = running_dice / n_steps
        avg_unif = running_unif / n_steps
        avg_band = running_band / n_steps
        avg_bhead = running_bhead / n_steps
        avg_snap = running_snap / n_steps
        avg_hd = running_hd / n_steps
        avg_curv = running_curv / n_steps
        avg_hardb = running_hardb / n_steps
        avg_teacher_ch = running_teacher_ch / n_steps
        avg_teacher_hard = running_teacher_hard / n_steps
        avg_proposal_ch = running_proposal_ch / n_steps
        avg_proposal_dice = running_proposal_dice / n_steps
        avg_exact_off = running_exact_off / n_steps
        avg_exact_conf = running_exact_conf / n_steps
        avg_exact_rate = running_exact_rate / n_steps
        avg_exact_teacher_rate = running_exact_teacher_rate / n_steps
        history["epoch"].append(epoch + 1)
        history["loss"].append(avg)
        history["loss_x0"].append(avg_x0)
        history["loss_chamfer"].append(avg_chamfer)
        history["loss_edge"].append(avg_edge)
        history["soft_dice_loss"].append(avg_dice)
        history["loss_uniformity"].append(avg_unif)
        history["boundary_band_loss"].append(avg_band)
        history["boundary_head_loss"].append(avg_bhead)
        history["snap_teacher_loss"].append(avg_snap)
        history["loss_hd"].append(avg_hd)
        history["loss_curvature"].append(avg_curv)
        history["loss_hard_boundary"].append(avg_hardb)
        history["snap_teacher_chamfer"].append(avg_teacher_ch)
        history["snap_teacher_hard"].append(avg_teacher_hard)
        history["snap_geometry_alpha"].append(float(snap_alpha))
        history["proposal_chamfer"].append(avg_proposal_ch)
        history["proposal_dice_loss"].append(avg_proposal_dice)
        history["exact_loss_offset"].append(avg_exact_off)
        history["exact_loss_conf"].append(avg_exact_conf)
        history["exact_target_conf_rate"].append(avg_exact_rate)
        history["exact_teacher_conf_rate"].append(avg_exact_teacher_rate)
        history["lr"].append(cur_lr)
        loss_msg = (
            f"Epoch {epoch+1} | train loss {avg:.4f} | "
            f"x0 {avg_x0:.4f} | chamfer {avg_chamfer:.4f} | edge {avg_edge:.4f} | "
            f"soft_dice_loss {avg_dice:.4f} | band {avg_band:.4f} | hd {avg_hd:.4f} | "
            f"curv {avg_curv:.4f} | hardb {avg_hardb:.4f} | unif {avg_unif:.4f} | "
            f"proposal_ch {avg_proposal_ch:.4f} | proposal_dice {avg_proposal_dice:.4f} | "
            f"exact_off {avg_exact_off:.4f} | exact_conf {avg_exact_conf:.4f} | "
            f"conf_rate {avg_exact_rate:.3f}/{avg_exact_teacher_rate:.3f} | lr {cur_lr:.2e}"
        )
        if epoch_dp:
            keys = [k for k in epoch_dp[0] if k not in ("epoch", "step")]
            mean_dp = {k: float(sum(d[k] for d in epoch_dp) / len(epoch_dp)) for k in keys}
            history["dp2seg"].append({"epoch": epoch + 1, **mean_dp})
            print(
                f"{loss_msg} bb_lr {bb_lr:.2e} | V5.2 t={mean_dp.get('t_mean', 0):.2f} "
                f"global={mean_dp.get('global_gate', 0):.2f} local={mean_dp.get('local_gate', 0):.2f} "
                f"fine={mean_dp.get('scale_fine', 0):.2f} coarse={mean_dp.get('scale_coarse', 0):.2f} "
                f"offset={mean_dp.get('offset_abs', 0):.3f} "
                f"exact={mean_dp.get('exact_applied_abs', 0):.4f} "
                f"conf={mean_dp.get('exact_conf_mean', 0):.2f} "
                f"residual={mean_dp.get('residual_abs', 0):.3f}"
            )
        else:
            print(loss_msg)

        # ----- periodic validation with EMA weights -----
        do_eval = (epoch + 1) % cfg.eval_every == 0 or (epoch + 1) == cfg.epochs
        if do_eval:
            from .sample import evaluate  # local import avoids a cycle at module load
            ema_encoder, ema_denoiser = ema.modules
            dice, iou = evaluate(ema_encoder, ema_denoiser, diffusion, val_loader, cfg, device,
                                 viz_path=os.path.join(cfg.out_dir, f"val_epoch{epoch+1}.png"),
                                 metrics_path=os.path.join(
                                     cfg.out_dir, f"val_epoch{epoch+1}_case_metrics.csv"
                                 ))
            history["val_epoch"].append(epoch + 1)
            history["val_dice"].append(dice)
            history["val_iou"].append(iou)
            print(f"Epoch {epoch+1} | val Dice {dice:.4f} | val IoU {iou:.4f}")

            if dice > best_val:
                best_val = dice
                torch.save({
                    "encoder": ema_encoder.state_dict(),
                    "denoiser": ema_denoiser.state_dict(),
                    "epoch": epoch + 1,
                    "val_dice": dice, "val_iou": iou,
                    "config": {k: getattr(cfg, k) for k in cfg.__dataclass_fields__},
                }, os.path.join(cfg.out_dir, "best.pth"))
                print(f"  -> new best (Dice {dice:.4f}) -> best.pth")

        # Always keep the latest EMA checkpoint.
        ema_encoder, ema_denoiser = ema.modules
        torch.save({"encoder": ema_encoder.state_dict(), "denoiser": ema_denoiser.state_dict(),
                    "epoch": epoch + 1,
                    "config": {k: getattr(cfg, k) for k in cfg.__dataclass_fields__},
                    }, os.path.join(cfg.out_dir, "last.pth"))

        with open(os.path.join(cfg.out_dir, "history.json"), "w") as f:
            json.dump(history, f, indent=2)

    _plot_history(history, os.path.join(cfg.out_dir, "loss_curve.png"))
    # Final test eval on best checkpoint if present.
    best_path = os.path.join(cfg.out_dir, "best.pth")
    test_metrics = {}
    if os.path.isfile(best_path):
        try:
            from .sample import evaluate, load_checkpoint
            test_ds = build_contour_dataset(
                cfg.skin_root, cfg.dataset, "te", augment=False, **ds_kwargs,
            )
            test_loader = DataLoader(test_ds, batch_size=cfg.batch_size, shuffle=False,
                                     num_workers=cfg.num_workers)
            cfg_eval = Config()
            ckpt = torch.load(best_path, map_location=device, weights_only=False)
            for k, v in (ckpt.get("config") or {}).items():
                if hasattr(cfg_eval, k):
                    setattr(cfg_eval, k, v)
            enc_t, den_t = load_checkpoint(best_path, cfg_eval, device)
            test_dice, test_iou = evaluate(
                enc_t, den_t, diffusion, test_loader, cfg_eval, device,
                viz_path=os.path.join(cfg.out_dir, "test_grid.png"),
                metrics_path=os.path.join(cfg.out_dir, "test_case_metrics.csv"),
            )
            test_metrics = {"test_dice": test_dice, "test_iou": test_iou}
            with open(os.path.join(cfg.out_dir, "test_metrics.json"), "w") as f:
                json.dump(test_metrics, f, indent=2)
            print(f"Test Dice {test_dice:.4f} | Test IoU {test_iou:.4f}")
        except Exception:
            print("Test eval failed:")
            traceback.print_exc()

    summary = {
        "out_dir": cfg.out_dir,
        "dataset": cfg.dataset,
        "best_val_dice": best_val,
        **test_metrics,
        "epochs": cfg.epochs,
        "lr": cfg.lr,
        "weight_decay": cfg.weight_decay,
        "scheduler": cfg.scheduler,
        "encoder": cfg.encoder,
        "backbone": cfg.pvt_variant if cfg.encoder == "pvt" else cfg.backbone,
        "decoder": f"{cfg.proposal_type}_{cfg.diffusion_target}_v5_2_exact_boundary",
        "proposal_type": cfg.proposal_type,
        "diffusion_target": cfg.diffusion_target,
        "fourier_harmonics": cfg.fourier_harmonics,
        "proposal_arclength_resample": cfg.proposal_arclength_resample,
        "residual_scale": cfg.residual_scale,
        "lambda_dice": cfg.lambda_dice,
        "lambda_chamfer": cfg.lambda_chamfer,
        "lambda_edge": cfg.lambda_edge,
        "lambda_boundary_band": cfg.lambda_boundary_band,
        "lambda_boundary_head": cfg.lambda_boundary_head,
        "lambda_snap_teacher": cfg.lambda_snap_teacher,
        "lambda_hd": cfg.lambda_hd,
        "lambda_curvature": cfg.lambda_curvature,
        "lambda_hard_boundary": cfg.lambda_hard_boundary,
        "hard_boundary_fraction": cfg.hard_boundary_fraction,
        "snap_teacher_hard_weight": cfg.snap_teacher_hard_weight,
        "snap_geometry_warmup_epochs": cfg.snap_geometry_warmup_epochs,
        "snap_geometry_ramp_epochs": cfg.snap_geometry_ramp_epochs,
        "lambda_proposal_chamfer": cfg.lambda_proposal_chamfer,
        "lambda_proposal_dice": cfg.lambda_proposal_dice,
        "exact_boundary_enabled": cfg.exact_boundary_enabled,
        "exact_boundary_radius": cfg.exact_boundary_radius,
        "exact_confidence_radius": cfg.exact_confidence_radius,
        "lambda_exact_offset": cfg.lambda_exact_offset,
        "lambda_exact_confidence": cfg.lambda_exact_confidence,
        "low_t_fraction": cfg.low_t_fraction,
        "loss_weighting": cfg.loss_weighting,
        "guidance_scale": cfg.guidance_scale,
        "aug_level": cfg.aug_level,
        "freeze_backbone": cfg.freeze_backbone,
    }
    with open(os.path.join(cfg.out_dir, "summary.json"), "w") as f:
        json.dump(summary, f, indent=2)
    print(f"Done. Best val Dice {best_val:.4f}. Artifacts in {cfg.out_dir}")


def _plot_history(history, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax1 = plt.subplots(figsize=(10, 5))
    ax1.plot(history["epoch"], history["loss"], "k-", label="train loss")
    if history.get("soft_dice_loss"):
        ax1.plot(history["epoch"], history["soft_dice_loss"], "C1--",
                 label="soft-Dice loss", alpha=0.85)
    if history.get("proposal_dice_loss") and any(history["proposal_dice_loss"]):
        ax1.plot(history["epoch"], history["proposal_dice_loss"], "C0--",
                 label="proposal Dice loss", alpha=0.85)
    ax1.set_xlabel("epoch"); ax1.set_ylabel("loss"); ax1.grid(alpha=0.3)
    handles, labels = ax1.get_legend_handles_labels()
    if history["val_dice"]:
        ax2 = ax1.twinx()
        ax2.plot(history["val_epoch"], history["val_dice"], "g.-", label="val Dice")
        ax2.set_ylabel("val Dice")
        h2, l2 = ax2.get_legend_handles_labels()
        handles += h2
        labels += l2
    ax1.legend(handles, labels, loc="best")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


if __name__ == "__main__":
    main()
