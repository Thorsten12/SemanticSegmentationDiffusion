"""All hyperparameters for P2SDiff V5.2 confidence-gated exact-boundary diffusion."""

from dataclasses import dataclass, fields
from typing import Tuple


@dataclass
class Config:
    model_version: str = "v5_2_exact_boundary"

    # ----- data -----
    skin_root: str = "/hdd/datasets/Skin"
    data_root: str = "/hdd/datasets"
    dataset: str = "ph2"
    npy_size: int = 224
    img_size: Tuple[int, int] = (224, 224)
    n_points: int = 100
    augment: bool = True
    aug_level: str = "light"
    polyp_test: str = "kvasir"

    # ----- backbone -----
    in_channels: int = 3
    encoder: str = "unet"
    stem_dim: int = 32
    # Legacy options retained so older commands/configs still parse.  V2 does
    # explicit projected spatial PE inside the global/local decoder.
    tag_maps: bool = True
    tag_mode: str = "concat"
    backbone: str = "convnext_tiny"
    pretrained: bool = False
    freeze_backbone: bool = False
    backbone_lr: float = 1e-5
    pvt_variant: str = "pvt_v2_b2"
    pvt_pretrained_path: str = "pretrained_pth/pvt/pvt_v2_b2.pth"
    unet_start_dim: int = 64
    unet_dim_mults: Tuple[int, ...] = (1, 2, 4)
    unet_groupnorm_groups: int = 16

    # ----- contour denoiser -----
    hidden_dim: int = 128
    n_transformer_layers: int = 4
    n_heads: int = 4
    coord_fourier_bands: int = 6
    contour_phase_bands: int = 4
    deformable_samples: int = 5
    local_radius_min: float = 0.025
    local_radius_max: float = 0.22
    global_levels: int = 3
    global_grid: int = 14
    # Coarse geometry ablations:
    #   ellipse + absolute = V3.1-compatible baseline
    #   fourier + absolute = spatial proposal ablation
    #   fourier + residual = proposed low/high-frequency decomposition
    proposal_type: str = "ellipse"               # "ellipse" | "fourier"
    diffusion_target: str = "absolute"            # "absolute" | "residual"
    fourier_harmonics: int = 4
    # V5.1: put Fourier vertices on a uniform arc-length grid before defining
    # pointwise residuals, then diffuse a normalized residual state u=(GT-P)/s.
    # Defaults preserve every existing V5 checkpoint and command.
    proposal_arclength_resample: bool = False
    residual_scale: float = 1.0

    # ----- legacy V3 snap options kept for checkpoint/CLI compatibility -----
    boundary_dim: int = 32
    boundary_levels: int = 3
    boundary_thickness: int = 1
    snap_candidates: int = 15
    snap_iterations: int = 3
    snap_radius_max: float = 0.16
    snap_radius_min: float = 0.015
    snap_temperature: float = 0.16
    snap_hard_inference: bool = False
    snap_teacher_offset: float = 0.10
    snap_teacher_smooth: int = 2
    snap_profile_radius: float = 0.018

    # Bias part of training toward low-noise timesteps where pixel-accurate
    # boundary refinement is learned, while retaining uniform high-noise samples.
    low_t_fraction: float = 0.35
    low_t_max_fraction: float = 0.25


    # ----- V5.2 sparse exact-boundary branch -----
    exact_boundary_enabled: bool = True
    exact_boundary_levels: int = 2
    exact_boundary_samples: int = 11
    exact_boundary_radius: float = 0.10
    exact_boundary_profile_dim: int = 20
    exact_boundary_hidden: int = 64
    exact_boundary_ring_bands: int = 4
    exact_boundary_relative_bias: float = 0.12
    exact_boundary_confidence_power: float = 2.0
    exact_boundary_low_t_fraction: float = 0.30
    exact_confidence_radius: float = 0.060
    exact_tangent_tolerance: float = 0.040
    exact_use_rgb: bool = True
    exact_teacher_offset: float = 0.060
    exact_teacher_smooth: int = 2
    lambda_exact_offset: float = 1.0
    lambda_exact_confidence: float = 0.50

    # ----- diffusion -----
    timesteps: int = 1000
    beta_start: float = 1e-4
    beta_end: float = 2e-2
    ddim_steps: int = 50
    # First establish a strong conditional model.  CFG is optional and off by
    # default because it previously caused border collapse / overshoot.
    guidance_scale: float = 1.0
    cfg_dropout: float = 0.0
    latent_clamp: float = 4.0
    x0_clamp: float = 4.0  # legacy alias; new code uses latent_clamp

    # ----- loss -----
    loss_weighting: str = "uniform"               # "uniform" | "min_snr"
    snr_gamma: float = 5.0
    lambda_dice: float = 1.0
    lambda_chamfer: float = 0.40
    lambda_edge: float = 0.05
    lambda_uniformity: float = 0.0
    lambda_boundary_band: float = 1.00
    lambda_boundary_head: float = 0.0
    lambda_snap_teacher: float = 0.0
    lambda_hd: float = 0.20
    hd_fraction: float = 0.20
    lambda_curvature: float = 0.05
    # Small one-sided top-k penalty for the few predicted vertices that create spikes.
    lambda_hard_boundary: float = 0.08
    hard_boundary_fraction: float = 0.15
    # Teacher supervision is set-based, not index-to-index.
    snap_teacher_hard_weight: float = 0.50
    snap_teacher_hard_fraction: float = 0.20
    lambda_proposal_chamfer: float = 1.0
    lambda_proposal_dice: float = 1.0
    proposal_dice_size: int = 96
    boundary_band_width: int = 2
    soft_dice_size: int = 96
    cyclic_reverse: bool = True

    # ----- optimization -----
    epochs: int = 300
    batch_size: int = 8
    lr: float = 1e-4
    weight_decay: float = 1e-4
    ema_decay: float = 0.999
    amp: bool = True
    num_workers: int = 4
    grad_clip: float = 1.0
    scheduler: str = "cosine"
    warmup_epochs: int = 5
    # Geometry curriculum: first train diffusion + snapper teacher separately, then
    # gradually let snapped geometry drive Dice/Chamfer losses.
    snap_geometry_warmup_epochs: int = 60
    snap_geometry_ramp_epochs: int = 60
    min_lr: float = 1e-6
    init_checkpoint: str = ""
    init_backbone: str = ""

    # ----- bookkeeping -----
    out_dir: str = "src/runs/ph2_v5_2_exact_boundary"
    seed: int = 0
    # Negative keeps the historical stochastic evaluation behavior.  V5.1
    # experiments set this explicitly so every validation epoch sees the same
    # DDIM initial noise and checkpoint selection is reproducible.
    eval_seed: int = -1
    device: str = "cuda"
    eval_every: int = 20

    @classmethod
    def from_args(cls, args) -> "Config":
        cfg = cls()
        for f in fields(cls):
            if hasattr(args, f.name) and getattr(args, f.name) is not None:
                setattr(cfg, f.name, getattr(args, f.name))
        return cfg
