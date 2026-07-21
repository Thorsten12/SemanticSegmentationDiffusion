"""Central configuration for the P2SDiff baseline.

Every hyperparameter lives here so that `train.py` and `sample.py` stay thin.
Override any field from the CLI, e.g. `python -m src.train --epochs 300 --batch_size 8`.
"""

from dataclasses import dataclass, field, fields
from typing import Tuple


@dataclass
class Config:
    # ----- data -----
    skin_root: str = "/loctmp/sit28238/SemanticSegmentationDiffusion/data/datasets"
    dataset: str = "ph2"                          # ph2 | isic2017 | isic2018 | ham10000
    npy_size: int = 224
    img_size: Tuple[int, int] = (224, 224)
    n_points: int = 200
    augment: bool = True
    aug_level: str = "light"                      # "none" | "light" | "strong"
    data_root: str = "/hdd/datasets/Skin/PH2"
    n_val: int = 20
    n_test: int = 40
    split_seed: int = 42

    # ----- TTA and Augmentation Extensions -----
    tta: bool = False
    mid_frequency_gain: float = 1.5
    edge_frequency_gain: float = 1.2
    adaptive_uniformity: bool = True

    # ----- model: conditioning encoder -----
    in_channels: int = 3
    encoder: str = "convnext"
    cond_channels: int = 64
    stem_dim: int = 32

    backbone: str = "convnext_tiny"
    pretrained: bool = True
    freeze_backbone: bool = False
    backbone_lr: float = 1e-5
    pretrained_weights: str | None = None

    pvt_variant: str = "pvt_v2_b2"
    pvt_pretrained_path: str = "pretrained_pth/pvt/pvt_v2_b2.pth"

    unet_start_dim: int = 64
    unet_dim_mults: Tuple[int, ...] = (1, 2, 4)
    unet_groupnorm_groups: int = 16

    # ----- model: point denoiser -----
    hidden_dim: int = 128
    n_transformer_layers: int = 4
    n_heads: int = 4

    # ----- positional encoding -----
    coord_fourier_bands: int = 6
    pos_grid_bands: int = 0

    # ----- diffusion -----
    timesteps: int = 1000
    beta_start: float = 1e-4
    beta_end: float = 2e-2
    ddim_steps: int = 50
    guidance_scale: float = 2.0
    cfg_dropout: float = 0.15
    x0_clamp: float = 1.2

    # ----- loss -----
    lambda_uniformity: float = 0.1
    lambda_dice: float = 1.0
    # lambda_boundary: kept small so boundary term stays O(1) relative to
    #   the Kendall noise loss after the NOISE_SCALE fix in diffusion.py.
    lambda_boundary: float = 0.1
    snr_gamma: float = 5.0

    # ------------------------------------------------------------------ #
    # Uncertainty head                                                     #
    # ------------------------------------------------------------------ #
    # lambda_uncertainty: scales the log(sigma) penalty in the Kendall loss.
    #   0.5  = model can express more uncertainty (sigma drifts lower freely)
    #   1.0  = exact Kendall & Gal formulation
    #   2.0  = tight regularization, sigma stays close to 1
    #
    # With NOISE_SCALE=100 on the precision term, lambda_uncertainty=0.5
    # gives the penalty roughly 1/200 the weight of the reconstruction term
    # at sigma=1, which is a reasonable starting balance.
    lambda_uncertainty: float = 0.5

    # uncertainty_skip_threshold: sigma = exp(log_sigma) below this value
    #   causes a point to be frozen in ddim_sample after the first DDIM step.
    #   None = disabled (full 50-step sampling for all points, safe default).
    #   0.3  = good starting value once the uncertainty head is calibrated
    #          (run ~50 epochs without skipping first to let sigma stabilise).
    #   0.5  = aggressive skipping; faster but may hurt accuracy on fine edges.
    uncertainty_skip_threshold: float | None = None

    # ----- optimization -----
    epochs: int = 300
    batch_size: int = 8
    lr: float = 1e-4
    weight_decay: float = 1e-4
    ema_decay: float = 0.999
    amp: bool = True
    num_workers: int = 4
    grad_clip: float = 1.0

    # ----- bookkeeping -----
    out_dir: str = "src/runs/baseline"
    seed: int = 0
    device: str = "cuda"
    log_every: int = 20
    eval_every: int = 20

    @classmethod
    def from_args(cls, args) -> "Config":
        """Build a Config, overriding defaults with any non-None argparse value."""
        cfg = cls()
        for f in fields(cls):
            if hasattr(args, f.name) and getattr(args, f.name) is not None:
                setattr(cfg, f.name, getattr(args, f.name))
        return cfg