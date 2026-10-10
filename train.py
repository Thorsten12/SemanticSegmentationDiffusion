"""Train the P2SDiff boundary-point diffusion model.

Run from the repository root:
    python -m src.train --dataset isic2017 --epochs 100 --batch_size 16

Training jointly optimizes the encoder, the contour denoiser, the boundary
snapper and (optionally) the coarse-contour proposal head. EMA checkpoints, a
loss curve, scale-gate statistics and periodic validation visualizations are
saved under `cfg.out_dir`. After training, the best checkpoint is evaluated on
the test split and per-component complexity statistics are recorded.
"""

import argparse
import json
import math
import os
import traceback

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
import torch.nn as nn
from tqdm import tqdm


from .config import Config
from .data import build_contour_dataset, split_counts
from .models import (
    ContourDenoiser, build_conditioner, BoundarySnapper, ContourProposalHead,
    apply_proposal_target, proposal_dice_loss,
)
from .diffusion import GaussianDiffusion
from .utils import EMA

def str2bool(v):
    """Parse a boolean CLI argument."""
    if isinstance(v, bool):
        return v
    v = str(v).lower()
    if v in ("1", "true", "t", "yes", "y"):
        return True
    if v in ("0", "false", "f", "no", "n"):
        return False
    raise argparse.ArgumentTypeError(f"Boolean expected, got {v!r}")

def _current_onpolicy_weight(cfg, epoch_idx: int) -> float:
    """Weight of the on-policy snapper loss at a given epoch.

    The weight is 0 during the warmup epochs, then ramps linearly to
    `lambda_snap_onpolicy` over `snap_onpolicy_ramp_epochs` (immediately if the
    ramp is 0). `epoch_idx` is 0-based.
    """
    if cfg.lambda_snap_onpolicy <= 0:
        return 0.0
    warmup = max(int(cfg.snap_onpolicy_warmup_epochs), 0)
    if epoch_idx < warmup:
        return 0.0
    ramp = max(int(cfg.snap_onpolicy_ramp_epochs), 0)
    if ramp == 0:
        return cfg.lambda_snap_onpolicy
    progress = min(1.0, (epoch_idx - warmup + 1) / ramp)
    return cfg.lambda_snap_onpolicy * progress


def _resolve_residual_target(cfg: Config) -> str:
    """Decide what the denoiser's zero-initialized head output represents.

    "zero":  a proposal supplies the initial guess; the denoiser predicts the
             local correction on top of it.
    "input": the denoiser predicts a residual on its noisy input (no proposal).
    "none":  direct prediction, no residual.
    """
    if cfg.proposal_target == "residual":
        return "zero"
    return "input" if cfg.predict_residual else "none"


def build_models(cfg: Config, device):
    """Instantiate encoder, denoiser, snapper and (optionally) the proposal head."""
    encoder = build_conditioner(cfg).to(device)
    residual_target = _resolve_residual_target(cfg)
    print("n_points", cfg.n_points, "hidden_dim", cfg.hidden_dim,
          "n_transformer_layers", cfg.n_transformer_layers, "n_heads", cfg.n_heads,
          "feature_channels", encoder.feature_channels, "cond_channels", cfg.cond_channels,
          "coord_fourier_bands", cfg.coord_fourier_bands,
          "sampler_type", cfg.sampler_type,
          "use_deformable_sampling", cfg.use_deformable_sampling,
          "deform_n_samples", cfg.deform_n_samples,
          "deform_min_scale_res", cfg.deform_min_scale_res,
          "deform_radius_cells", cfg.deform_radius_cells,
          "attn_sampler_window", cfg.attn_sampler_window,
          "attn_sampler_heads", cfg.attn_sampler_heads, 
          "top_k_scales", cfg.top_k_scales,
          "use_global_attn", cfg.use_global_attn,
          "global_attn_levels", cfg.global_attn_levels,
          "global_attn_grid", cfg.global_attn_grid,
          "global_attn_position", cfg.global_attn_position,
          "global_attn_gate_schedule", cfg.global_attn_gate_schedule,
          "local_drop_coarsest", cfg.local_drop_coarsest)

    denoiser = ContourDenoiser(
        pos_scale=cfg.pos_scale,
        n_points=cfg.n_points,
        hidden_dim=cfg.hidden_dim,
        num_layers=cfg.n_transformer_layers,
        num_heads=cfg.n_heads,
        attn_window=cfg.attn_mask,
        scale_channels=encoder.feature_channels,
        proj_dim=cfg.cond_channels,
        coord_fourier_bands=cfg.coord_fourier_bands,
        timesteps=cfg.timesteps,
        predict_residual=cfg.predict_residual,
        residual_target=residual_target,
        guidance_time_scale=cfg.guidance_time_scale,
        sampler_type=cfg.sampler_type,
        use_deformable_sampling=cfg.use_deformable_sampling,
        deform_n_samples=cfg.deform_n_samples,
        deform_min_scale_res=cfg.deform_min_scale_res,
        deform_radius_cells=cfg.deform_radius_cells,
        attn_sampler_window=cfg.attn_sampler_window,
        attn_sampler_heads=cfg.attn_sampler_heads,
        top_k_scales=cfg.top_k_scales,
        use_global_attn=cfg.use_global_attn,
        global_attn_levels=cfg.global_attn_levels,
        global_attn_grid=cfg.global_attn_grid,
        global_attn_position=cfg.global_attn_position,
        global_attn_gate_schedule=cfg.global_attn_gate_schedule,
        global_attn_gate_min=cfg.global_attn_gate_min,
        global_attn_gate_max=cfg.global_attn_gate_max,
        local_drop_coarsest=cfg.local_drop_coarsest,
    ).to(device)
    snapper = BoundarySnapper(
        scale_channels=encoder.feature_channels,
        n_points=cfg.n_points,
        levels=cfg.snap_levels,
        n_samples=cfg.snap_n_samples,
        radius=cfg.snap_radius,
        profile_dim=cfg.snap_profile_dim,
        hidden_dim=cfg.snap_hidden_dim,
        ring_bands=cfg.snap_ring_bands,
        num_heads=cfg.snap_num_heads,
        relative_bias_strength=cfg.snap_relative_bias,
        confidence_power=cfg.snap_confidence_power,
        use_rgb=cfg.snap_use_rgb,
    ).to(device)

    # The proposal head operates on the coarsest backbone features.
    proposal_head = None
    if cfg.proposal_target != "absolute":
        coarsest_channels = encoder.feature_channels[-1]
        proposal_head = ContourProposalHead(
            coarsest_channels=coarsest_channels,
            n_points=cfg.n_points,
            hidden_dim=cfg.proposal_hidden_dim,
            num_heads=cfg.proposal_num_heads,
            harmonics=cfg.proposal_harmonics,
            proposal_type=cfg.proposal_type,
            n_shapes=cfg.proposal_n_shapes,
            union_raster_size=cfg.proposal_union_raster_size,
        ).to(device)
        print(f"ContourProposalHead: type={cfg.proposal_type!r} n_shapes={cfg.proposal_n_shapes} "
              f"union_raster_size={cfg.proposal_union_raster_size}"
              + (" (n_shapes==1 -> union machinery unused, identical to single-shape V5)"
                 if cfg.proposal_n_shapes == 1 else ""))

    return encoder, denoiser, snapper, proposal_head


def _load_init_checkpoint(path, encoder, denoiser, snapper, device, proposal_head=None):
    """Warm-start from a checkpoint (EMA weights); mismatched keys are skipped."""
    ckpt = torch.load(path, map_location=device, weights_only=False)
    for name, module in (("encoder", encoder), ("denoiser", denoiser), ("snapper", snapper)):
        if name not in ckpt:
            print(f"  warn: no '{name}' in {path}")
            continue
        missing, unexpected = module.load_state_dict(ckpt[name], strict=False)
        print(f"  loaded {name}: missing={len(missing)} unexpected={len(unexpected)}")
    if proposal_head is not None:
        if "proposal_head" in ckpt:
            missing, unexpected = proposal_head.load_state_dict(ckpt["proposal_head"], strict=False)
            print(f"  loaded proposal_head: missing={len(missing)} unexpected={len(unexpected)}")
        else:
            print(f"  warn: no 'proposal_head' in {path} -- using freshly-initialized proposal head")


# --------------------------------------------------------------------------
# Model complexity profiling
# --------------------------------------------------------------------------
# The wrappers below expose each component as a plain nn.Module with
# positional tensor inputs, so that hook-based FLOP counters (fvcore) see the
# real submodule tree.

class _EncoderExtractWrapper(nn.Module):
    """Wraps `encoder.extract` as a module."""
    def __init__(self, encoder):
        super().__init__()
        self.encoder = encoder

    def forward(self, image):
        return self.encoder.extract(image)


class _DenoiserCallWrapper(nn.Module):
    """Wraps the denoiser call; condition maps are passed as *args."""
    def __init__(self, denoiser):
        super().__init__()
        self.denoiser = denoiser

    def forward(self, points, t, *cond_maps):
        # Each map is a separate positional tensor so fvcore's tracer sees it
        # as its own input.
        return self.denoiser(points, t, list(cond_maps), return_scale_weights=False)


class _SnapperCallWrapper(nn.Module):
    """Wraps the snapper call (raw maps as *args, no RGB input)."""
    def __init__(self, snapper):
        super().__init__()
        self.snapper = snapper

    def forward(self, points, *raw_maps):
        return self.snapper(points, list(raw_maps), image=None, hard=True)


def compute_model_stats(module, device, n_warmup=5, n_timed=20, dummy_inputs=None):
    """Profile parameters, FLOPs, latency and peak GPU memory of a module.

    FLOPs are counted with fvcore if available, otherwise with thop (MACs * 2).
    Latency is the mean over `n_timed` forward passes after `n_warmup` warmup
    passes. Peak memory is measured on CUDA only. Failures in individual
    measurements are recorded in the returned dict instead of raised.
    """
    stats = {}
    n_total = sum(p.numel() for p in module.parameters())
    n_trainable = sum(p.numel() for p in module.parameters() if p.requires_grad)
    stats["total_params"] = n_total
    stats["trainable_params"] = n_trainable
    stats["total_params_M"] = round(n_total / 1e6, 3)
    stats["trainable_params_M"] = round(n_trainable / 1e6, 3)

    module.eval()

    # FLOPs.
    stats["flops"] = None
    stats["flops_G"] = None
    stats["flops_source"] = None
    if dummy_inputs is not None:
        try:
            from fvcore.nn import FlopCountAnalysis
            with torch.no_grad():
                flop_counter = FlopCountAnalysis(module, dummy_inputs)
                flop_counter.unsupported_ops_warnings(False)
                flops = flop_counter.total()
            stats["flops"] = int(flops)
            stats["flops_G"] = round(flops / 1e9, 3)
            stats["flops_source"] = "fvcore"
        except ImportError:
            try:
                from thop import profile
                with torch.no_grad():
                    macs, _ = profile(module, inputs=dummy_inputs, verbose=False)
                stats["flops"] = int(macs * 2)
                stats["flops_G"] = round(macs * 2 / 1e9, 3)
                stats["flops_source"] = "thop (MACs*2)"
            except ImportError:
                stats["flops_source"] = "unavailable (pip install fvcore or thop for this)"
        except Exception as e:
            stats["flops_source"] = f"error: {e}"
    else:
        stats["flops_source"] = "skipped (no dummy_inputs provided)"

    # Latency.
    stats["latency_ms_per_call"] = None
    if dummy_inputs is not None:
        try:
            import time
            with torch.no_grad():
                for _ in range(n_warmup):
                    module(*dummy_inputs)
                if device.type == "cuda":
                    torch.cuda.synchronize()
                start = time.perf_counter()
                for _ in range(n_timed):
                    module(*dummy_inputs)
                if device.type == "cuda":
                    torch.cuda.synchronize()
                elapsed = time.perf_counter() - start
            stats["latency_ms_per_call"] = round((elapsed / n_timed) * 1000, 3)
        except Exception as e:
            stats["latency_error"] = str(e)

    # Peak memory.
    stats["peak_memory_MB"] = None
    if device.type == "cuda" and dummy_inputs is not None:
        try:
            torch.cuda.reset_peak_memory_stats(device)
            with torch.no_grad():
                module(*dummy_inputs)
            torch.cuda.synchronize()
            stats["peak_memory_MB"] = round(torch.cuda.max_memory_allocated(device) / 1e6, 2)
        except Exception as e:
            stats["peak_memory_error"] = str(e)

    module.train()
    return stats


def compute_p2sdiff_model_stats(encoder, denoiser, snapper, cfg, device):
    """Per-component complexity statistics (encoder, denoiser, snapper) and totals.

    Measured on a single dummy sample at t = 0 with the configured image size.
    Totals sum parameters, FLOPs and latency across components (None if any
    component lacks the value) and take the maximum for peak memory. The output
    format matches `baseline_seg.compute_model_stats`.
    """
    b = 1
    img_size = cfg.img_size
    if isinstance(img_size, (tuple, list)):
        h, w = img_size
    else:
        h = w = img_size
    img = torch.randn(b, 3, h, w, device=device)
    t = torch.zeros(b, dtype=torch.long, device=device)
    points = torch.zeros(b, cfg.n_points, 2, device=device)

    with torch.no_grad():
        raw = encoder.extract(img)
        cond_maps = encoder.fuse(raw, t)

    encoder_wrapper = _EncoderExtractWrapper(encoder).to(device)
    encoder_stats = compute_model_stats(encoder_wrapper, device, dummy_inputs=(img,))

    denoiser_wrapper = _DenoiserCallWrapper(denoiser).to(device)
    denoiser_stats = compute_model_stats(
        denoiser_wrapper, device, dummy_inputs=(points, t, *cond_maps)
    )

    snapper_wrapper = _SnapperCallWrapper(snapper).to(device)
    snapper_stats = compute_model_stats(
        snapper_wrapper, device, dummy_inputs=(points, *raw)
    )

    def _sum_field(field):
        vals = [d.get(field) for d in (encoder_stats, denoiser_stats, snapper_stats)]
        if any(v is None for v in vals):
            return None
        return sum(vals)

    total_params = _sum_field("total_params")
    total_trainable = _sum_field("trainable_params")
    total_stats = {
        "total_params": total_params,
        "trainable_params": total_trainable,
        "total_params_M": round(total_params / 1e6, 3) if total_params is not None else None,
        "trainable_params_M": round(total_trainable / 1e6, 3) if total_trainable is not None else None,
        "flops_G": _sum_field("flops_G"),
        "latency_ms_per_call": _sum_field("latency_ms_per_call"),
        "peak_memory_MB": max(
            (v for v in (d.get("peak_memory_MB") for d in
                         (encoder_stats, denoiser_stats, snapper_stats)) if v is not None),
            default=None,
        ),
    }

    return {
        "encoder": encoder_stats,
        "denoiser": denoiser_stats,
        "snapper": snapper_stats,
        "total": total_stats,
    }


class WarmupCosineScheduler:
    """Per-epoch learning-rate schedule: linear warmup, then cosine decay to `min_lr`."""

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


# Scale-gate statistics are aggregated over this many equally sized timestep bins.
N_GATE_BINS = 3


def _bin_labels(n_bins):
    if n_bins == 3:
        return ["early", "mid", "late"]
    return [f"bin{i}" for i in range(n_bins)]


def main():
    parser = argparse.ArgumentParser(description="Train P2SDiff")

    # ----- data and optimization -----
    parser.add_argument("--dataset", choices=[  "ph2", "isic2017", "isic2018", "ham10000", "busi", "tn3k",
                                                "polyp_clinicdb", "polyp_kvasir", "polyp_colondb", "polyp_etis", "polyp_cvc300"], required=True)
    parser.add_argument("--skin_root", type=str)
    parser.add_argument("--out_dir", type=str)
    parser.add_argument("--img_size", type=lambda x: tuple(map(int, x.split(","))), default=(224, 224))
    parser.add_argument("--epochs", type=int)
    parser.add_argument("--batch_size", type=int)
    parser.add_argument("--lr", type=float)
    parser.add_argument("--weight_decay", type=float)
    parser.add_argument("--n_points", type=int)
    parser.add_argument("--encoder", choices=["convnext", "pvt", "resnet", "swin", "vmamba"], default="convnext")
    parser.add_argument("--backbone", type=str)
    parser.add_argument("--freeze_backbone", type=str2bool, default=None)
    parser.add_argument("--backbone_lr", type=float)
    parser.add_argument("--stem_dim", type=int)
    parser.add_argument("--coord_fourier_bands", type=int)
    parser.add_argument("--pos_grid_bands", type=int)
    parser.add_argument("--device", type=str)
    parser.add_argument("--num_workers", type=int)
    parser.add_argument("--eval_every", type=int)
    parser.add_argument("--aug_level", choices=["none", "light", "strong"])
    parser.add_argument("--guidance_scale", type=float)
    parser.add_argument("--ddim_steps", type=int)
    parser.add_argument("--lambda_dice", type=float)
    parser.add_argument("--lambda_uniformity", type=float)
    parser.add_argument("--lambda_boundary", type=float)
    parser.add_argument("--soft_dice_size", type=int)
    parser.add_argument("--snr_gamma", type=float)
    parser.add_argument("--scheduler", choices=["none", "cosine"])
    parser.add_argument("--warmup_epochs", type=int)
    parser.add_argument("--min_lr", type=float)
    parser.add_argument("--init_checkpoint", type=str)
    parser.add_argument("--seed", type=int)
    parser.add_argument("--no_amp", action="store_true")
    parser.add_argument("--cond_channels", type=int, default=64)
    parser.add_argument("--hidden_dim", type=int, default=128)
    parser.add_argument("--lora_rank", type=int, default=8)
    parser.add_argument("--use_lora", type=str2bool, default=False)
    parser.add_argument("--pos_scale", type=float, default=0.15)
    parser.add_argument("--attn_mask", type=int, default=7)
    parser.add_argument("--predict_residual", type=str2bool, default=True)
    parser.add_argument("--lambda_mse", type=float, default=1.0)
    parser.add_argument("--lambda_biou", type=float, default=1.0)
    parser.add_argument("--biou_gamma", type=float,default=1.0)
    parser.add_argument("--n_heads", type=int, default=4)
    parser.add_argument("--lambda_nearest", type=float, default=1.0)
    parser.add_argument("--nearest_gamma", type=float, default=1.0)

    # ----- boundary snapper -----
    parser.add_argument("--lambda_snap", type=float, default=1.0)
    parser.add_argument("--snap_n_samples", type=int, default=11)
    parser.add_argument("--snap_radius", type=float, default=0.10)
    parser.add_argument("--snap_levels", type=int, default=2)
    parser.add_argument("--snap_profile_dim", type=int, default=20)
    parser.add_argument("--snap_hidden_dim", type=int, default=64)
    parser.add_argument("--snap_ring_bands", type=int, default=4)
    parser.add_argument("--snap_num_heads", type=int, default=4)
    parser.add_argument("--snap_relative_bias", type=float, default=0.12)
    parser.add_argument("--snap_confidence_power", type=float, default=2.0)
    parser.add_argument("--snap_use_rgb", type=str2bool, default=True)
    parser.add_argument("--snap_perturb_offset", type=float, default=0.08)
    parser.add_argument("--snap_perturb_smooth", type=int, default=2)
    parser.add_argument("--snap_confidence_radius", type=float, default=0.060)
    parser.add_argument("--snap_tangent_tolerance", type=float, default=0.040)
    parser.add_argument("--lambda_snap_biou", type=float, default=0.0,
                        help="Weight for an additional BoundaryIoU loss on the snapper's "
                             "own gated correction. 0.0 (default) keeps the old behavior.")
    parser.add_argument("--snap_biou_size", type=int, default=64)

    # Snapper usage at inference: post-hoc ("post"), inside the DDIM loop
    # ("loop"), both, or disabled.
    parser.add_argument("--snap_mode", choices=["post", "loop", "both", "none"], default="post")
    parser.add_argument("--snap_t_threshold_frac", type=float, default=0.15)
    parser.add_argument("--snap_every", type=int, default=1)

    # ----- coarse-contour proposal -----
    parser.add_argument("--proposal_target", choices=["absolute", "residual"], default="residual")
    parser.add_argument("--proposal_type", choices=["fourier", "ellipse"], default="fourier")
    parser.add_argument("--proposal_n_shapes", type=int, default=1)
    parser.add_argument("--proposal_union_raster_size", type=int, default=64)
    parser.add_argument("--proposal_harmonics", type=int, default=4)
    parser.add_argument("--proposal_hidden_dim", type=int, default=128)
    parser.add_argument("--proposal_num_heads", type=int, default=4)
    parser.add_argument("--lambda_proposal_dice", type=float, default=0.0)

    parser.add_argument("--gif", action="store_true")
    parser.add_argument("--gif_steps", type=int, default=10)

    # ----- timestep-dependent damping of the multi-scale guidance -----
    parser.add_argument("--guidance_time_scale", type=float, default=0.0)

    # ----- sensitivity regularization -----
    parser.add_argument("--lambda_sensitivity", type=float, default=0.0)
    parser.add_argument("--sensitivity_min_ratio", type=float, default=0.15)

    # ----- multi-scale sampling strategy (mutually exclusive; see denoiser.py) -----
    parser.add_argument("--sampler_type", choices=["input", "deformable", "attention"], default=None)
    parser.add_argument("--use_deformable_sampling", type=str2bool, default=None)
    parser.add_argument("--deform_n_samples", type=int, default=None)
    parser.add_argument("--deform_min_scale_res", type=int, default=None)
    parser.add_argument("--deform_radius_cells", type=float, default=None)
    parser.add_argument("--attn_sampler_window", type=int, default=None)
    parser.add_argument("--attn_sampler_heads", type=int, default=None)

    parser.add_argument("--lambda_curvature", type=float, default=0.0)
    parser.add_argument("--curvature_gamma", type=float, default=1.0)
    parser.add_argument("--top_k_scales", type=int, default=None)

    # ----- on-policy snapper loss (trained on the denoiser's own predictions) -----
    parser.add_argument("--lambda_snap_onpolicy", type=float, default=0.0)
    parser.add_argument("--snap_onpolicy_warmup_epochs", type=int, default=10)
    parser.add_argument("--snap_onpolicy_ramp_epochs", type=int, default=10)

    # ----- global attention -----
    parser.add_argument("--use_global_attn", type=str2bool, default=False)
    parser.add_argument("--global_attn_levels", type=int, default=2)
    parser.add_argument("--global_attn_grid", type=int, default=14)

    parser.add_argument("--global_attn_position", choices=["pre", "post", "both"], default=None)
    parser.add_argument("--global_attn_gate_schedule",
                        choices=["linear", "quadratic", "cosine", "constant"], default=None)
    parser.add_argument("--global_attn_gate_min", type=float, default=None)
    parser.add_argument("--global_attn_gate_max", type=float, default=None)

    parser.add_argument("--local_drop_coarsest", type=int, default=None)

    args = parser.parse_args()

    cfg = Config.from_args(args)
    if args.no_amp:
        cfg.amp = False

    # Legacy flag: map the boolean deformable switch onto `sampler_type`.
    if args.sampler_type is None and args.use_deformable_sampling is not None:
        cfg.sampler_type = "deformable" if args.use_deformable_sampling else "input"

    if cfg.proposal_n_shapes > 1 and cfg.lambda_proposal_dice <= 0:
        print("WARNING: --proposal_n_shapes > 1 but --lambda_proposal_dice is 0 -- the "
              "individual shape parameters will receive NO gradient at all. Set "
              "--lambda_proposal_dice > 0 unless this is intentional.")

    torch.manual_seed(cfg.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(cfg.seed)
    device = torch.device(cfg.device if torch.cuda.is_available() else "cpu")
    os.makedirs(cfg.out_dir, exist_ok=True)
    with open(os.path.join(cfg.out_dir, "config.json"), "w") as f:
        json.dump({k: getattr(cfg, k) for k in cfg.__dataclass_fields__}, f, indent=2, default=str)

    # ----- data -----
    counts = split_counts(cfg.skin_root, cfg.dataset, cfg.npy_size)
    print(f"Dataset {cfg.dataset} | split -> train {counts['tr']} | "
          f"val {counts['vl']} | test {counts['te']}")
    train_ds = build_contour_dataset(cfg.skin_root, cfg.dataset, "tr", cfg.n_points,
                                     cfg.img_size, augment=cfg.augment,
                                     aug_level=cfg.aug_level, npy_size=cfg.npy_size)
    val_ds = build_contour_dataset(cfg.skin_root, cfg.dataset, "vl", cfg.n_points,
                                   cfg.img_size, augment=False, npy_size=cfg.npy_size)
    persist = cfg.num_workers > 0
    train_loader = DataLoader(train_ds, batch_size=cfg.batch_size, shuffle=True,
                              num_workers=cfg.num_workers, drop_last=True, pin_memory=True,
                              persistent_workers=persist)
    val_loader = DataLoader(val_ds, batch_size=cfg.batch_size, shuffle=False,
                            num_workers=cfg.num_workers, persistent_workers=persist)

    # ----- models, EMA, optimizer -----
    encoder, denoiser, snapper, proposal_head = build_models(cfg, device)
    if cfg.init_checkpoint:
        print(f"Warm-start from {cfg.init_checkpoint}")
        _load_init_checkpoint(cfg.init_checkpoint, encoder, denoiser, snapper, device,
                              proposal_head=proposal_head)

    diffusion = GaussianDiffusion(cfg.timesteps, cfg.beta_start, cfg.beta_end, device=device)
    ema_modules = [encoder, denoiser, snapper]
    if proposal_head is not None:
        ema_modules.append(proposal_head)
    ema = EMA(ema_modules, decay=cfg.ema_decay)

    # Separate parameter groups: the backbone (if unfrozen) uses its own learning rate.
    backbone = getattr(encoder, "backbone", None)
    backbone_ids = {id(p) for p in backbone.parameters()} if backbone is not None else set()
    backbone_params, head_params = [], []
    for p in encoder.parameters():
        if not p.requires_grad:
            continue
        (backbone_params if id(p) in backbone_ids else head_params).append(p)
    head_params += list(denoiser.parameters())
    head_params += list(snapper.parameters())
    if proposal_head is not None:
        head_params += list(proposal_head.parameters())

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
    scaler = torch.cuda.amp.GradScaler(enabled=use_amp)

    n_train = sum(p.numel() for p in trainable)
    n_total = (sum(p.numel() for p in encoder.parameters())
               + sum(p.numel() for p in denoiser.parameters())
               + sum(p.numel() for p in snapper.parameters())
               + (sum(p.numel() for p in proposal_head.parameters()) if proposal_head is not None else 0))
    bb_state = "frozen" if not backbone_params else f"fine-tune @ lr {cfg.backbone_lr:g}"
    print(f"Encoder: {cfg.encoder}/{cfg.backbone} (backbone {bb_state}) | "
          f"trainable {n_train/1e6:.2f}M / total {n_total/1e6:.2f}M | "
          f"device {device} | amp {use_amp} | sched {cfg.scheduler} | "
          f"proposal_target {cfg.proposal_target} | snap_mode {cfg.snap_mode} | "
          f"guidance_time_scale {cfg.guidance_time_scale} | lambda_sensitivity {cfg.lambda_sensitivity} | "
          f"sampler_type {cfg.sampler_type}")

    n_scales = denoiser.sampler.n_scales
    bin_labels = _bin_labels(N_GATE_BINS)
    print(f"Scale-gate logging: {n_scales} scales x {N_GATE_BINS} timestep bins ({'/'.join(bin_labels)})")

    history = {"epoch": [], "loss": [], "lr": [], "val_epoch": [], "val_dice": [], "val_iou": [],
               "scale_gates": [], "snap_offset_loss": [], "snap_conf_loss": [], "snap_conf_rate": [],
               "snap_biou_loss": [], "proposal_dice_loss": [], "loss_sensitivity": [],
               "snap_onpolicy_offset_loss": [], "snap_onpolicy_conf_loss": [],
                "snap_onpolicy_weight": []}
    best_val = -1.0

    # ----- training loop -----
    for epoch in range(cfg.epochs):
        print(f"Positional scale: {denoiser.pos_scale.item()}")

        if scheduler is not None:
            scheduler.step(epoch + 1)
        cur_lr = optimizer.param_groups[0]["lr"]

        encoder.train(); denoiser.train(); snapper.train()
        if proposal_head is not None:
            proposal_head.train()
        running = 0.0
        snap_off_running = 0.0
        snap_conf_running = 0.0
        snap_conf_rate_running = 0.0
        snap_biou_running = 0.0
        proposal_dice_running = 0.0
        sensitivity_running = 0.0
        snap_onpolicy_off_running = 0.0
        snap_onpolicy_conf_running = 0.0

        # Accumulators for the mean scale-gate value per timestep bin.
        gate_sum = torch.zeros(N_GATE_BINS, n_scales, device=device)
        gate_cnt = torch.zeros(N_GATE_BINS, device=device)

        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{cfg.epochs}")
        onpolicy_weight = _current_onpolicy_weight(cfg, epoch)
        onpolicy_active = onpolicy_weight > 0 and cfg.snap_mode in ("loop", "both") and cfg.lambda_snap > 0
        for images, points, masks in pbar:
            images = images.to(device, non_blocking=True)
            points = points.to(device, non_blocking=True)
            masks = masks.to(device, non_blocking=True)
            b = images.shape[0]

            optimizer.zero_grad(set_to_none=True)
            t = torch.randint(0, cfg.timesteps, (b,), device=device).long()

            with torch.amp.autocast("cuda", enabled=use_amp):
                # Forward diffusion, feature extraction and coarse proposal.
                noisy = diffusion.q_sample(points, t)
                raw = encoder.extract(images)
                proposal = proposal_head(raw) if proposal_head is not None else None

                if proposal_head is not None and cfg.lambda_proposal_dice > 0:
                    loss_proposal_dice = proposal_dice_loss(proposal_head, raw, masks, size=cfg.soft_dice_size)
                else:
                    loss_proposal_dice = torch.zeros((), device=device)

                # Classifier-free guidance training: randomly drop the condition maps.
                cond_maps = encoder.fuse(raw, t)
                keep = (torch.rand(b, device=device) >= cfg.cfg_dropout)
                keep = keep.to(cond_maps[0].dtype).view(b, 1, 1, 1)
                cond_maps = [m * keep for m in cond_maps]
                pred_x0, scale_w = denoiser(noisy, t, cond_maps, return_scale_weights=True)
                pred_x0 = torch.clamp(pred_x0, -cfg.x0_clamp, cfg.x0_clamp)
                pred_x0 = apply_proposal_target(pred_x0, proposal, cfg.proposal_target)

                # Sensitivity regularization: a second prediction from the same
                # x0 with independent noise.
                if cfg.lambda_sensitivity > 0:
                    noise_b = torch.randn_like(points)
                    noisy_b = diffusion.q_sample(points, t, noise=noise_b)
                    pred_x0_b, _ = denoiser(noisy_b, t, cond_maps, return_scale_weights=True)
                    pred_x0_b = torch.clamp(pred_x0_b, -cfg.x0_clamp, cfg.x0_clamp)
                    pred_x0_b = apply_proposal_target(pred_x0_b, proposal, cfg.proposal_target)

                    loss_sensitivity = sensitivity_loss(
                        pred_x0, pred_x0_b, noisy, noisy_b,
                        min_ratio=cfg.sensitivity_min_ratio,
                    )
                else:
                    loss_sensitivity = torch.zeros((), device=device)

                # Diffusion losses and auxiliary terms.
                loss, parts = diffusion.training_losses(
                    pred_x0, points, t, masks=masks,
                    lambda_mse=cfg.lambda_mse,
                    lambda_boundary=cfg.lambda_boundary,
                    lambda_uniformity=cfg.lambda_uniformity,
                    lambda_dice=cfg.lambda_dice, snr_gamma=cfg.snr_gamma,
                    lambda_nearest=cfg.lambda_nearest, nearest_gamma=cfg.nearest_gamma,
                    biou_gamma=cfg.biou_gamma, lambda_biou=cfg.lambda_biou,     
                    lambda_curvature=cfg.lambda_curvature, curvature_gamma=cfg.curvature_gamma,
                    soft_dice_size=cfg.soft_dice_size)
                loss = loss + cfg.lambda_proposal_dice * loss_proposal_dice
                loss = loss + cfg.lambda_sensitivity * loss_sensitivity

                # Snapper teacher loss (perturbed ground-truth contours).
                if cfg.lambda_snap > 0:
                    snap_loss, snap_parts = snapper.teacher_loss(
                        points, raw, image=images,
                        perturb_max_offset=cfg.snap_perturb_offset,
                        perturb_smooth_passes=cfg.snap_perturb_smooth,
                        confidence_radius=cfg.snap_confidence_radius,
                        tangent_tolerance=cfg.snap_tangent_tolerance,
                        masks=masks,
                        lambda_biou=cfg.lambda_snap_biou,
                        biou_size=cfg.snap_biou_size,
                    )
                    loss = loss + cfg.lambda_snap * snap_loss
                else:
                    snap_parts = {
                        "snap_teacher_loss_offset": torch.zeros((), device=device),
                        "snap_teacher_loss_conf": torch.zeros((), device=device),
                        "snap_teacher_conf_rate": torch.zeros((), device=device),
                        "snap_teacher_loss_biou": torch.zeros((), device=device),
                    }

                # Snapper on-policy loss: same targets as the teacher loss, but on
                # the denoiser's own (detached) x0 predictions. Active only after
                # warmup, when snapping inside the sampling loop is used.
                if onpolicy_active:
                    onpolicy_points = pred_x0.detach()
                    off_pred_op, conf_logit_op, normal_op = snapper._predict(
                        onpolicy_points, raw, image=images
                    )
                    with torch.no_grad():
                        d_op = torch.cdist(onpolicy_points.float(), points.float(), p=2)
                        min_dist_op, idx_op = d_op.min(dim=2)
                        b_idx_op = torch.arange(onpolicy_points.shape[0], device=device)[:, None]
                        nearest_op = points[b_idx_op, idx_op]
                        delta_op = nearest_op - onpolicy_points
                        tangent_op = torch.stack([normal_op[..., 1], -normal_op[..., 0]], dim=-1)
                        signed_n_op = (delta_op * normal_op).sum(dim=-1)
                        signed_t_op = (delta_op * tangent_op).sum(dim=-1).abs()
                        target_conf_op = (
                            (min_dist_op <= cfg.snap_confidence_radius)
                            & (signed_t_op <= cfg.snap_tangent_tolerance)
                            & (signed_n_op.abs() <= snapper.radius)
                        ).to(onpolicy_points.dtype)
                        target_offset_op = signed_n_op.clamp(-snapper.radius, snapper.radius)

                    pos_op = target_conf_op.sum().clamp(min=1.0)
                    neg_op = (target_conf_op.numel() - target_conf_op.sum()).clamp(min=1.0)
                    pos_weight_op = (neg_op / pos_op).clamp(1.0, 12.0)
                    loss_conf_op = F.binary_cross_entropy_with_logits(
                        conf_logit_op, target_conf_op, pos_weight=pos_weight_op
                    )

                    w_op = target_conf_op
                    if float(w_op.sum().item()) > 0:
                        loss_off_op = F.smooth_l1_loss(
                            off_pred_op / max(snapper.radius, 1e-6),
                            target_offset_op / max(snapper.radius, 1e-6),
                            reduction="none",
                        )
                        loss_off_op = (loss_off_op * w_op).sum() / w_op.sum().clamp(min=1.0)
                    else:
                        loss_off_op = off_pred_op.new_zeros(())

                    loss_onpolicy = loss_off_op + loss_conf_op
                    loss = loss + onpolicy_weight * loss_onpolicy
                else:
                    loss_off_op = torch.zeros((), device=device)
                    loss_conf_op = torch.zeros((), device=device)

            # Optimization step and EMA update.
            scaler.scale(loss).backward()
            if cfg.grad_clip:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(trainable, cfg.grad_clip)
            scaler.step(optimizer)
            scaler.update()
            ema.update([encoder, denoiser, snapper] + ([proposal_head] if proposal_head is not None else []))

            running += loss.item()
            snap_off_running += snap_parts["snap_teacher_loss_offset"].item()
            snap_conf_running += snap_parts["snap_teacher_loss_conf"].item()
            snap_conf_rate_running += snap_parts["snap_teacher_conf_rate"].item()
            snap_biou_running += snap_parts["snap_teacher_loss_biou"].item()
            proposal_dice_running += loss_proposal_dice.item()
            sensitivity_running += loss_sensitivity.item()
            snap_onpolicy_off_running += loss_off_op.item()
            snap_onpolicy_conf_running += loss_conf_op.item()

            # Accumulate scale-gate statistics per timestep bin.
            with torch.no_grad():
                bin_idx = (t.float() / cfg.timesteps * N_GATE_BINS).long().clamp(0, N_GATE_BINS - 1)
                w = scale_w.detach().float()
                for bi in range(N_GATE_BINS):
                    mask = bin_idx == bi
                    if mask.any():
                        gate_sum[bi] += w[mask].sum(dim=0)
                        gate_cnt[bi] += mask.sum()
                batch_mean_w = w.mean(dim=0)

            pbar.set_postfix({"loss": f"{loss.item():.4f}",
                             "x0": f"{parts['loss_x0'].item():.4f}",
                              "dice": f"{parts['loss_dice'].item():.4f}",
                              "unif": f"{parts['loss_uniformity'].item():.4f}",
                              "boundary": f"{parts['loss_boundary'].item():.4f}",
                              "loss_biou": f"{parts['loss_biou'].item():.4f}",
                              "loss_nearest": f"{parts['loss_nearest'].item():.4f}",
                              "loss_curv": f"{parts['loss_curvature'].item():.4f}",
                              "prop_dice": f"{loss_proposal_dice.item():.4f}",
                              "sens": f"{loss_sensitivity.item():.4f}",
                              "snap_off": f"{snap_parts['snap_teacher_loss_offset'].item():.4f}",
                              "snap_conf": f"{snap_parts['snap_teacher_loss_conf'].item():.4f}",
                              "snap_crate": f"{snap_parts['snap_teacher_conf_rate'].item():.3f}",
                              "snap_biou": f"{snap_parts['snap_teacher_loss_biou'].item():.4f}",
                              "gate": "/".join(f"{v:.2f}" for v in batch_mean_w.tolist()),
                              "lr": f"{cur_lr:.2e}"})

        # ----- epoch logging -----
        avg = running / max(1, len(train_loader))
        n_batches = max(1, len(train_loader))
        history["epoch"].append(epoch + 1)
        history["loss"].append(avg)
        history["lr"].append(cur_lr)
        history["snap_offset_loss"].append(snap_off_running / n_batches)
        history["snap_conf_loss"].append(snap_conf_running / n_batches)
        history["snap_conf_rate"].append(snap_conf_rate_running / n_batches)
        history["snap_biou_loss"].append(snap_biou_running / n_batches)
        history["proposal_dice_loss"].append(proposal_dice_running / n_batches)
        history["loss_sensitivity"].append(sensitivity_running / n_batches)
        history["snap_onpolicy_offset_loss"].append(snap_onpolicy_off_running / n_batches)
        history["snap_onpolicy_conf_loss"].append(snap_onpolicy_conf_running / n_batches)
        history["snap_onpolicy_weight"].append(onpolicy_weight)
        print(f"Epoch {epoch+1} | snapper on-policy: offset {history['snap_onpolicy_offset_loss'][-1]:.4f} "
              f"| conf {history['snap_onpolicy_conf_loss'][-1]:.4f} | weight {onpolicy_weight:.3f}")

        gate_avg = (gate_sum / gate_cnt.clamp(min=1).unsqueeze(1)).cpu().tolist()
        history["scale_gates"].append(gate_avg)
        gate_str = " | ".join(
            f"{lbl}: [" + ", ".join(f"s{si}={v:.3f}" for si, v in enumerate(row)) + "]"
            for lbl, row in zip(bin_labels, gate_avg)
        )
        print(f"Epoch {epoch+1} | train loss {avg:.4f} | lr {cur_lr:.2e}")
        print(f"Epoch {epoch+1} | scale-gate importance -> {gate_str}")
        print(f"Epoch {epoch+1} | proposal dice: {history['proposal_dice_loss'][-1]:.4f}")
        print(f"Epoch {epoch+1} | sensitivity loss: {history['loss_sensitivity'][-1]:.4f}")
        print(f"Epoch {epoch+1} | snapper teacher: offset {history['snap_offset_loss'][-1]:.4f} "
              f"| conf {history['snap_conf_loss'][-1]:.4f} "
              f"| conf_rate {history['snap_conf_rate'][-1]:.3f} "
              f"| biou {history['snap_biou_loss'][-1]:.4f}")

        # ----- validation (EMA weights) and best-checkpoint selection -----
        do_eval = (epoch + 1) % cfg.eval_every == 0 or (epoch + 1) == cfg.epochs
        if do_eval:
            from .sample import evaluate
            ema_mods = ema.modules
            ema_encoder, ema_denoiser, ema_snapper = ema_mods[0], ema_mods[1], ema_mods[2]
            ema_proposal_head = ema_mods[3] if proposal_head is not None else None
            snap_t_threshold = int(round(cfg.snap_t_threshold_frac * cfg.timesteps))
            dice, iou = evaluate(ema_encoder, ema_denoiser, diffusion, val_loader, cfg, device,
                                 snapper=ema_snapper, proposal_head=ema_proposal_head,
                                 viz_path=os.path.join(cfg.out_dir, f"val_epoch{epoch+1}.png"),
                                 snap_mode=cfg.snap_mode, snap_t_threshold=snap_t_threshold,
                                 snap_every=cfg.snap_every,
                                 gif=cfg.gif, gif_steps=cfg.gif_steps)
            history["val_epoch"].append(epoch + 1)
            history["val_dice"].append(dice)
            history["val_iou"].append(iou)
            print(f"Epoch {epoch+1} | val Dice {dice:.4f} | val IoU {iou:.4f}")

            if dice > best_val:
                best_val = dice
                ckpt_out = {
                    "encoder": ema_encoder.state_dict(),
                    "denoiser": ema_denoiser.state_dict(),
                    "snapper": ema_snapper.state_dict(),
                    "epoch": epoch + 1,
                    "val_dice": dice, "val_iou": iou,
                    "config": {k: getattr(cfg, k) for k in cfg.__dataclass_fields__},
                }
                if ema_proposal_head is not None:
                    ckpt_out["proposal_head"] = ema_proposal_head.state_dict()
                torch.save(ckpt_out, os.path.join(cfg.out_dir, "best.pth"))
                print(f"  -> new best (Dice {dice:.4f}) -> best.pth")

        # Always save the latest EMA checkpoint and the training history.
        ema_mods = ema.modules
        ema_encoder, ema_denoiser, ema_snapper = ema_mods[0], ema_mods[1], ema_mods[2]
        ema_proposal_head = ema_mods[3] if proposal_head is not None else None
        last_ckpt = {"encoder": ema_encoder.state_dict(), "denoiser": ema_denoiser.state_dict(),
                    "snapper": ema_snapper.state_dict(),
                    "epoch": epoch + 1,
                    "config": {k: getattr(cfg, k) for k in cfg.__dataclass_fields__},
                    }
        if ema_proposal_head is not None:
            last_ckpt["proposal_head"] = ema_proposal_head.state_dict()
        torch.save(last_ckpt, os.path.join(cfg.out_dir, "last.pth"))

        with open(os.path.join(cfg.out_dir, "history.json"), "w") as f:
            json.dump(history, f, indent=2)

    _plot_history(history, os.path.join(cfg.out_dir, "loss_curve.png"))
    _plot_scale_gates(history, bin_labels, os.path.join(cfg.out_dir, "scale_gates.png"))

    # ----- final test evaluation and complexity statistics (best checkpoint) -----
    best_path = os.path.join(cfg.out_dir, "best.pth")
    test_metrics = {}
    model_stats = {}   # stays empty if checkpoint loading or evaluation fails
    if os.path.isfile(best_path):
        try:
            from .sample import evaluate, load_checkpoint

            cfg_eval = Config()
            enc_t, den_t, snap_t, prop_t = load_checkpoint(best_path, cfg_eval, device)


            test_ds = build_contour_dataset(cfg.skin_root, cfg.dataset, "te", cfg.n_points,
                                            cfg.img_size, augment=False, npy_size=cfg.npy_size)
            test_loader = DataLoader(test_ds, batch_size=cfg.batch_size, shuffle=False,
                                     num_workers=cfg.num_workers)

            test_snap_t_threshold = int(round(cfg_eval.snap_t_threshold_frac * cfg_eval.timesteps))
            test_dice, test_iou = evaluate(
                enc_t, den_t, diffusion, test_loader, cfg_eval, device,
                snapper=snap_t, proposal_head=prop_t,
                viz_path=os.path.join(cfg.out_dir, "test_grid.png"),
                snap_mode=cfg_eval.snap_mode, snap_t_threshold=test_snap_t_threshold,
                snap_every=cfg_eval.snap_every,
                gif=cfg_eval.gif, gif_steps=cfg_eval.gif_steps
            )
            test_metrics = {"test_dice": test_dice, "test_iou": test_iou}
            with open(os.path.join(cfg.out_dir, "test_metrics.json"), "w") as f:
                json.dump(test_metrics, f, indent=2)
            print(f"Test Dice {test_dice:.4f} | Test IoU {test_iou:.4f}")


            print("Computing FLOPs / latency / memory stats (encoder/denoiser/snapper)...")
            model_stats = compute_p2sdiff_model_stats(enc_t, den_t, snap_t, cfg_eval, device)
        except Exception as e:

            print("Test eval failed:")
            traceback.print_exc()
            with open(os.path.join(cfg.out_dir, "test_eval_failed.json"), "w") as f:
                json.dump({"error": str(e), "traceback": traceback.format_exc()}, f, indent=2)
    else:

        with open(os.path.join(cfg.out_dir, "test_eval_failed.json"), "w") as f:
            json.dump({"error": "no best.pth was ever written during training"}, f, indent=2)

    summary = {
        "out_dir": cfg.out_dir,
        "dataset": cfg.dataset,
        "best_val_dice": best_val,
        **test_metrics,
        "epochs": cfg.epochs,
        "lr": cfg.lr,
        "weight_decay": cfg.weight_decay,
        "scheduler": cfg.scheduler,
        "backbone": cfg.backbone,
        "guidance_scale": cfg.guidance_scale,
        "aug_level": cfg.aug_level,
        "freeze_backbone": cfg.freeze_backbone,
        "proposal_target": cfg.proposal_target,
        "proposal_type": cfg.proposal_type,
        "proposal_n_shapes": cfg.proposal_n_shapes,
        "snap_mode": cfg.snap_mode,
        "guidance_time_scale": cfg.guidance_time_scale,
        "lambda_sensitivity": cfg.lambda_sensitivity,
        "sampler_type": cfg.sampler_type,
        "top_k_scales": cfg.top_k_scales,   
        "use_global_attn": cfg.use_global_attn,
        "global_attn_levels": cfg.global_attn_levels,
        "global_attn_grid": cfg.global_attn_grid,
        "global_attn_position": cfg.global_attn_position,
        "global_attn_gate_schedule": cfg.global_attn_gate_schedule,
        **(model_stats.get("total", {}) if model_stats else {}),  
        "model_stats": model_stats if model_stats else None,      
    }
    with open(os.path.join(cfg.out_dir, "summary.json"), "w") as f:
        json.dump(summary, f, indent=2)
    print(f"Done. Best val Dice {best_val:.4f}. Artifacts in {cfg.out_dir}")


def _plot_history(history, path):
    """Plot the training loss and (if available) the validation Dice."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax1 = plt.subplots(figsize=(10, 5))
    ax1.plot(history["epoch"], history["loss"], "k-", label="train loss")
    ax1.set_xlabel("epoch"); ax1.set_ylabel("loss"); ax1.grid(alpha=0.3)
    if history["val_dice"]:
        ax2 = ax1.twinx()
        ax2.plot(history["val_epoch"], history["val_dice"], "g.-", label="val Dice")
        ax2.set_ylabel("val Dice")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def _plot_scale_gates(history, bin_labels, path):
    """Plot the mean scale-gate value per scale over epochs, one panel per timestep bin."""
    gates = history.get("scale_gates")
    if not gates:
        return
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    epochs = history["epoch"]
    n_bins = len(bin_labels)
    n_scales = len(gates[0][0])

    fig, axes = plt.subplots(1, n_bins, figsize=(5 * n_bins, 4), sharey=True)
    if n_bins == 1:
        axes = [axes]
    for bi, (ax, lbl) in enumerate(zip(axes, bin_labels)):
        for si in range(n_scales):
            series = [epoch_gates[bi][si] for epoch_gates in gates]
            ax.plot(epochs, series, label=f"scale {si}")
        ax.set_title(f"t-bin: {lbl}")
        ax.set_xlabel("epoch")
        ax.grid(alpha=0.3)
        ax.set_ylim(0, 1)
    axes[0].set_ylabel("mean sigmoid gate")
    axes[-1].legend(loc="best", fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


if __name__ == "__main__":
    main()