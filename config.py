"""Central configuration for the P2SDiff baseline.

Every hyperparameter lives here so that `train.py` and `sample.py` stay thin.
Override any field from the CLI, e.g. `python -m src.train --epochs 300 --batch_size 8`.
"""

from dataclasses import dataclass, field, fields
from typing import Tuple


@dataclass
class Config:
    # ----- data -----
    # Datasets are read from preprocessed npy under <skin_root>/<DATASET>/np/ and
    # split by the published index ranges (see data/seg_datasets.py), so the
    # train/val/test partition matches the reference exactly.
    skin_root: str = "data/datasets"
    dataset: str = "ph2"                          # ph2 | isic2017 | isic2018 | ham10000
    npy_size: int = 224                           # resolution of the stored npy arrays
    img_size: Tuple[int, int] = (224, 224)        # (H, W) fed to the model
    n_points: int = 200                           # boundary points per contour
    augment: bool = True                          # train-time augmentation
    aug_level: str = "light"                      # "none" | "light" | "strong"
    # (legacy: file-based PH2 split; unused by the npy split path above)
    data_root: str = "data/data_repro/ph2"
    n_val: int = 20
    n_test: int = 40
    split_seed: int = 42

    # ----- model: conditioning encoder -----
    in_channels: int = 3                         # image-only conditioning (RGB)
    encoder: str = "convnext"                    # "convnext" | "pvt" | "unet"
    cond_channels: int = 64                      # channels of the condition feature map
    stem_dim: int = 32                           # full-res learnable stem prepended to pretrained pyramid (0 = off)

    # pretrained-backbone path ("convnext")
    backbone: str = "convnext_tiny"              # any timm features_only model
    pretrained: bool = True                      # load pretrained weights
    freeze_backbone: bool = False                # if True, train only fusion + denoiser
    backbone_lr: float = 1e-5                    # low LR for pretrained backbone when fine-tuning

    # PVT path ("pvt"): uses local models/pvtv2.py + pretrained_pth/pvt/*.pth
    pvt_variant: str = "pvt_v2_b2"
    pvt_pretrained_path: str = "pretrained_pth/pvt/pvt_v2_b2.pth"

    # from-scratch U-Net path ("unet")
    unet_start_dim: int = 64
    unet_dim_mults: Tuple[int, ...] = (1, 2, 4)
    unet_groupnorm_groups: int = 16

    # ----- model: point denoiser -----
    hidden_dim: int = 128
    n_transformer_layers: int = 4
    n_heads: int = 4
    attn_mask: int = 7
    # ----- multi-scale sampling strategy (mutually exclusive) -----
    # "input"      -> MultiScalePointSampler (legacy: 1 aligned grid_sample/scale)
    # "deformable" -> DeformableScalePointSampler (learned offsets + token-blind softmax)
    # "attention"  -> AttentionScalePointSampler (fixed neighborhood + content-based attention)
    sampler_type: str = "input"
    use_deformable_sampling: bool = False   # deprecated, kept for backward compat -- see denoiser.py
    deform_n_samples: int = 4          # reads/candidates per point per scale (1 aligned/center + K-1 others)
    deform_min_scale_res: int = 28     # scales >= this resolution get multi-sample reads; below -> aligned only
    deform_radius_cells: float = 2.5   # (deformable only) search radius in "grid cells" at each scale's own resolution
    attn_sampler_window: int = 5       # (attention only) side length of the fixed local candidate grid (e.g. 5 -> 5x5=25 candidates, of which deform_n_samples nearest are kept)
    attn_sampler_heads: int = 4        # (attention only) number of attention heads; must divide cond_channels

    # ----- LoRA -----
    use_lora: bool = False                       # if True, inject LoRA into the pretrained
    lora_rank: int = 8                           # LoRA rank (r)
    lora_alpha: int = 16                         # LoRA alpha (scaling)

    # ----- positional encoding -----
    # Band counts are deliberately small; frequencies are capped (see positional.py)
    # so high bands don't become aliasing noise the model overfits to. The
    # coordinate PE enters the denoiser ONLY via an additive path (safe). Injecting
    # position into the guidance *content* map (pos_grid) is the same kind of
    # position/content entanglement that caused validation collapse on this tiny
    # dataset, so it is disabled by default.
    coord_fourier_bands: int = 6    # NeRF-style PE of point coordinates (additive)
    pos_grid_bands: int = 0         # 2D Fourier grid in the guidance map (0 = off)

    # ----- diffusion -----
    timesteps: int = 1000
    beta_start: float = 1e-4
    beta_end: float = 2e-2
    ddim_steps: int = 50
    guidance_scale: float = 2.0                   # ~1.5 (tiny PH2) .. 2.0 (big sets); >=5 over-guides to borders & collapses
    cfg_dropout: float = 0.15                    # prob. of dropping conditioning in training
    x0_clamp: float = 1.2                        # clamp predicted x0 during training

    # ----- loss -----
    lambda_mse: float = 1.0                      # weight of x0 MSE
    lambda_uniformity: float = 0.1              # weight of neighbor-spacing regularizer
    lambda_dice: float = 1.0                    # weight of differentiable soft-Dice (mask-level)
    lambda_boundary: float = 1.0                # weight of boundary loss
    soft_dice_size: int = 64                    # soft-raster resolution (higher = sharper boundary grad)
    snr_gamma: float = 5.0                      # min-SNR-gamma cap for per-sample x0 weighting
    pos_scale: float = 0.15                           # scale of the additive positional embedding in the denoiser
    predict_residual: bool = True                           # if True, denoiser predicts residual to x0 instead of x0 directly
    lambda_biou:float = 1.0
    biou_gamma:float = 1.0
    lambda_nearest:float = 1.0
    nearest_gamma:float = 1.0

    # ----- boundary snapper (post-DDIM, teacher-trained) -----
    # Separates Modul (src/snapper.py). Läuft NICHT im Diffusions-Forward-Pass,
    # sondern als Post-Processing-Schritt nach dem fertigen DDIM-Sampling
    # (siehe sample.py: evaluate(..., snapper=...)). Wird über einen separaten
    # Teacher-Loss trainiert (perturb_along_normals auf GT, kein On-Policy-
    # Training auf echten Diffusions-Zwischenständen) -- additiv zum normalen
    # Diffusions-Loss in train.py.
    lambda_snap: float = 1.0                    # Gewicht des Snapper-Teacher-Loss im Gesamt-Loss
    snap_n_samples: int = 11                    # Profil-Samples entlang der Normalen (ungerade erzwungen)
    snap_radius: float = 0.10                   # max. Normal-Suchradius (gleicher [-1,1]-Koordinatenraum wie Punkte)
    snap_levels: int = 2                        # wie viele (feinste) Backbone-Skalen gesampelt werden
    snap_profile_dim: int = 20                  # Kanalzahl der projizierten Profil-Features
    snap_hidden_dim: int = 64                   # Hidden-Dim im Snapper-Transformer/Conv-Block
    snap_ring_bands: int = 4                    # Harmonische des fixen Ring-Positions-Features
    snap_num_heads: int = 4                     # Attention-Heads im Ring-Kommunikationsblock
    snap_relative_bias: float = 0.12            # Stärke des zyklischen Attention-Bias (nahe Nachbarn bevorzugt)
    snap_confidence_power: float = 2.0          # Schärfe des Confidence-Gates (sigmoid(conf)^power)
    snap_use_rgb: bool = True                   # zusätzlich rohes RGB an den Profil-Punkten sampeln
    snap_perturb_offset: float = 0.08           # max. Normal-Perturbation für die Teacher-Konturen
    snap_perturb_smooth: int = 2                # Glättungs-Pässe der Teacher-Perturbation
    snap_confidence_radius: float = 0.060       # Nearest-GT-Abstand, unterhalb dessen ein Vertex "confident" ist
    snap_tangent_tolerance: float = 0.040       # max. tangentiale Verschiebung für "confident"
    lambda_snap_onpolicy: float = 0.0            # 0.0 = aus; zusaetzlicher On-Policy-Loss
    snap_onpolicy_warmup_epochs: int = 10        # Epochen bis zur Aktivierung
    snap_onpolicy_ramp_epochs: int = 10          # lineare Rampe nach Warmup; 0 = sofort voll
    lambda_snap_biou: float = 1.0
    snap_biou_size: int = 64
    proposal_target: str = "residual"                    # "absolute" | "residual"; see apply_proposal_target()
    proposal_harmonics: int = 4                    # Fourier harmonics in the proposal head
    proposal_hidden_dim: int = 128                 # hidden dim in the proposal head
    proposal_num_heads: int = 4                    # attention heads in the proposal head
    lambda_proposal_dice: float = 1.0                    # weight of the proposal-dice loss (independent of the denoiser)
    lambda_curvature: float = 1.0                    # weight of the curvature penalty loss (see diffusion.py)
    curvature_gamma: float = 1.0                    # gamma for the curvature penalty loss (see diffusion.py)
    # V6: shape parametrization + multi-shape union. n_shapes==1 (default)
    # reproduces the old single-shape V5 proposal exactly; see
    # models/proposal.py's module docstring for the full union/differentiability
    # design (soft-OR union for training gradient, non-differentiable
    # cv2.findContours re-extraction for the point-contour path).
    proposal_type: str = "fourier"                 # "fourier" | "ellipse" -- per-shape parametrization
    proposal_n_shapes: int = 1                     # number of independent shape queries whose union forms the proposal
    proposal_union_raster_size: int = 64           # raster resolution for the multi-shape soft-union mask (n_shapes>1 only)
    guidance_time_scale: float = 0.0        # scale down the guidance signal linearly with timestep (0=off, 1=full)
    lambda_sensitivity: float = 0.0                    # weight of the sensitivity loss (see diffusion.py)
    sensitivity_min_ratio: float = 0.5                    # min ratio of sensitivity loss to MSE loss (see diffusion.py)
    # ----- V6: in-loop boundary snapping (applied during ddim_sample) -----
    # Controls WHEN/HOW the (separately, teacher-)trained BoundarySnapper is
    # applied at eval/sample time -- see sample.py's evaluate() and
    # diffusion.py's ddim_sample. Does NOT affect snapper training itself
    # (still lambda_snap / teacher_loss above, unchanged).
    snap_mode: str = "post"                        # "post" | "loop" | "both" | "none"
    snap_t_threshold_frac: float = 0.15            # in-loop snapping only in the last X% of timesteps (0..1), "loop"/"both"
    snap_every: int = 1                            # snap every N-th eligible low-t DDIM step ("loop"/"both")
    top_k_scales: int = None 
    local_drop_coarsest: int = 0

    use_global_attn: bool = False
    global_attn_levels: int = 2
    global_attn_grid: int = 14
    global_attn_position: str = "pre"          # "pre" | "post" | "both"
    global_attn_gate_schedule: str = "linear"  # "linear" | "quadratic" | "cosine" | "constant"
    global_attn_gate_min: float = 0.20
    global_attn_gate_max: float = 1.00

    # ----- optimization -----
    epochs: int = 300
    batch_size: int = 8
    lr: float = 1e-4
    weight_decay: float = 1e-4
    ema_decay: float = 0.999
    amp: bool = True
    num_workers: int = 4
    grad_clip: float = 1.0
    scheduler: str = "cosine"                    # "none" | "cosine"
    warmup_epochs: int = 5
    min_lr: float = 1e-6
    # Warm-start from a previous EMA checkpoint (transfer learning).
    init_checkpoint: str = ""
    init_from_ema: bool = True                   # unused alias; checkpoint already stores EMA weights

    # ----- bookkeeping -----
    out_dir: str = "src/runs/baseline"
    seed: int = 0
    device: str = "cuda"
    log_every: int = 20
    eval_every: int = 20                         # epochs between validation evals

    gif: bool = False                            # if True, save a GIF of the DDIM trajectory (for debugging)
    gif_steps: int = 10                          # number of DDIM steps to save in the

    @classmethod
    def from_args(cls, args) -> "Config":
        """Build a Config, overriding defaults with any non-None argparse value."""
        cfg = cls()
        for f in fields(cls):
            if hasattr(args, f.name) and getattr(args, f.name) is not None:
                setattr(cfg, f.name, getattr(args, f.name))
        return cfg