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
    skin_root: str = "/hdd/datasets/Skin"
    dataset: str = "ph2"                          # ph2 | isic2017 | isic2018 | ham10000
    npy_size: int = 224                           # resolution of the stored npy arrays
    img_size: Tuple[int, int] = (224, 224)        # (H, W) fed to the model
    n_points: int = 200                           # boundary points per contour
    augment: bool = True                          # train-time augmentation
    aug_level: str = "light"                      # "none" | "light" | "strong"
    # (legacy: file-based PH2 split; unused by the npy split path above)
    data_root: str = "/hdd/datasets/Skin/PH2"
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
    lambda_uniformity: float = 0.1              # weight of neighbor-spacing regularizer
    lambda_dice: float = 1.0                    # weight of differentiable soft-Dice (mask-level)
    snr_gamma: float = 5.0                      # min-SNR-gamma cap for per-sample x0 weighting

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
    eval_every: int = 20                         # epochs between validation evals

    @classmethod
    def from_args(cls, args) -> "Config":
        """Build a Config, overriding defaults with any non-None argparse value."""
        cfg = cls()
        for f in fields(cls):
            if hasattr(args, f.name) and getattr(args, f.name) is not None:
                setattr(cfg, f.name, getattr(args, f.name))
        return cfg
