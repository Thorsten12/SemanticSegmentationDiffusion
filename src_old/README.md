# P2SDiff — boundary-point diffusion for segmentation

A clean, self-contained implementation of **segmentation-as-boundary-point-generation**.
Instead of denoising per-pixel intensities, the diffusion process runs on the **2D
coordinates of `N` ordered boundary points**; the denoised polygon is rasterized
into a binary mask and scored with Dice / IoU.

This package is independent of the older `scr/` code — nothing here imports from it.

## Idea

```
mask ──contour──► N ordered points (arc-length uniform, [-1,1])   (ground-truth x0)
                         │
        forward q(x_t|x0): add Gaussian noise to the coordinates   (DDPM, T=1000)
                         │
   image ─► backbone ─► multi-scale feature pyramid + full-res stem  (extracted ONCE)
                         │
   reverse: ContourDenoiser predicts x0, reading every feature scale at each
            point's location (grid_sample), time-gated across scales
                         │
            DDIM (50 steps) + classifier-free guidance ──► clean points
                         │
            fillPoly ──► binary mask ──► Dice / IoU
```

- **Parameterization:** `x0`-prediction (the denoiser predicts the clean points).
- **Conditioning (image only, RGB):** a pretrained backbone (PVT-v2 or ConvNeXt)
  produces a 4-scale feature pyramid; a small **learnable full-resolution stem**
  on the raw image is prepended so points can read pixel-precise local cues, not
  just the backbone's coarse (stride-4 = 56×56) features. The denoiser samples
  **every scale per point** (`grid_sample`) and **time-gates** the scales (coarse
  context early in sampling, fine edges late). The heavy backbone runs once per
  image; the per-point query/gate runs each diffusion step.
- **Loss:** per-sample **min-SNR-γ weighted x0 MSE** + a **differentiable soft-Dice**
  (winding-number polygon rasterizer → mask-level gradient that pulls the boundary
  onto image edges) + a **uniformity** term (keeps points evenly spaced around the
  closed contour).
- **Tricks:** EMA (with warmup), **per-sample** classifier-free-guidance dropout in
  training + CFG at sampling, deterministic DDIM.
- **Augmentation:** synchronous flips / affine (rotation, translate, scale, shear)
  on image+mask with a validity-retry loop, plus image-only colour jitter / blur.
  Boundary points are always recomputed *from the augmented mask*, so image and
  points can never drift out of alignment.

## Layout

| File | Role |
|------|------|
| `config.py` | all hyperparameters (`Config` dataclass) |
| `data/seg_datasets.py` | multi-dataset loader with **published index splits** (reads preprocessed npy) |
| `data/ph2_dataset.py` | contour dataset: augmentation + arc-length contour sampling (`ArrayContourDataset`) |
| `data/skin/`, `data/dataset_*.py` | reference per-dataset loaders the splits follow |
| `models/encoder.py` | conditioning encoders: `PVTConditioner`, `ConvNeXtConditioner` (both + high-res stem), `UNetConditioner` |
| `models/pvtv2.py` | local Pyramid Vision Transformer v2 backbone |
| `models/feature_unet.py` | from-scratch image → full-res condition map (the `unet` encoder) |
| `models/denoiser.py` | `ContourDenoiser` + `MultiScalePointSampler` (per-point, time-gated multi-scale) |
| `models/attention.py`, `models/positional.py` | shared attention/MLP blocks; Fourier coordinate encodings |
| `diffusion/gaussian_diffusion.py` | schedule, `q_sample`, training losses, DDIM sampling (CFG) |
| `utils/rasterize.py` | `points_to_mask` (cv2 eval) + `soft_rasterize` / `soft_dice_loss` (differentiable) + Dice/IoU |
| `utils/ema.py`, `utils/viz.py` | EMA with warmup; prediction-grid visualization |
| `train.py` | training loop (CLI) |
| `sample.py` | evaluation / inference (CLI) + `evaluate()` |

## Conditioning encoders (`--encoder`)

| Encoder | Backbone | Feature scales (224 input) | Notes |
|---|---|---|---|
| `pvt` | PVT-v2-b2 (`models/pvtv2.py` + `pretrained_pth/pvt/pvt_v2_b2.pth`) | `[224², 56², 28², 14², 7²]` | stem + 4-scale pyramid |
| `convnext` | timm ConvNeXt-Tiny (ImageNet) | `[224², 56², 28², 14², 7²]` | stem + 4-scale pyramid |
| `unet` | from-scratch `FeatureUNet` | `[224²]` | single full-res map (ablation) |

The leading `224²` scale is the learnable stem (`stem_dim` channels, default 32;
set `stem_dim=0` in `config.py` to disable). Pretrained backbones fine-tune at
`--backbone_lr` (default 1e-5) or freeze with `--freeze_backbone`; the stem is
always trainable.

## Data

Datasets are read from the shared **preprocessed npy** under
`<skin_root>/<DATASET>/np/X_tr_{S}x{S}.npy` (+ `Y_`), in glob order, and split by
the **published index ranges** so our partition matches the reference loaders
exactly — only the target differs (we derive boundary points from each mask).
See `data/seg_datasets.py`.

| `--dataset` | train / val / test | total |
|---|---|---|
| `ph2` | 80 / 20 / 100 | 200 |
| `isic2017` | 1250 / 150 / 600 | 2000 |
| `isic2018` | 1815 / 259 / 520 | 2594 |
| `ham10000` | 7200 / 1800 / 1015 | 10015 |

Defaults: `--skin_root /hdd/datasets/Skin`, `--npy_size 224`, `img_size 224`.

## Usage

All commands run from the repository root.

```bash
# train: ConvNeXt-Tiny backbone + high-res stem (best recipe) on PH2
python -m src.train --dataset ph2 --encoder convnext --guidance_scale 1.5 \
    --epochs 800 --out_dir src/runs/my_run

# PVT-v2 backbone + high-res stem
python -m src.train --dataset ph2 --encoder pvt --guidance_scale 1.5 \
    --epochs 200 --out_dir src/runs/my_run_pvt

# from-scratch U-Net (ablation / strong baseline on tiny data)
python -m src.train --dataset ph2 --encoder unet --guidance_scale 1.5 --epochs 300

# a bigger dataset wants a slightly higher guidance scale
python -m src.train --dataset isic2018 --encoder pvt --guidance_scale 2.0 --epochs 200
```

Run long jobs in the background and watch the log:

```bash
nohup python -m src.train --dataset ph2 --encoder pvt --guidance_scale 1.5 \
    --epochs 200 --out_dir src/runs/my_run > src/runs_my_run.log 2>&1 &
grep "val Dice" src/runs_my_run.log     # validation (EMA weights) every 20 epochs
```

### Useful flags

| Flag | Default | Notes |
|---|---|---|
| `--guidance_scale` | 2.0 | **~1.5 for tiny PH2**, ~2.0 for larger sets; ≥5 over-guides points to image borders and collapses |
| `--encoder` | convnext | `pvt` / `convnext` / `unet` |
| `--lambda_dice` | 1.0 | weight of the differentiable soft-Dice term |
| `--snr_gamma` | 5.0 | min-SNR-γ cap for the per-sample x0 weight |
| `--aug_level` | light | `none` / `light` / `strong` |
| `--freeze_backbone` | off | train only stem + fusion + denoiser |
| `--backbone_lr` | 1e-5 | discriminative LR for the pretrained backbone |
| `--eval_every` | 20 | epochs between validations |

(`stem_dim`, `n_points`, diffusion schedule, etc. live in `config.py`.)

Artifacts per run land in `--out_dir`: `best.pth`, `last.pth`, `history.json`,
`loss_curve.png`, and periodic `val_epochN.png` grids (GT boundary | predicted
boundary | rasterized mask). Checkpoints store the architecture config, so
`sample.py` rebuilds the matching encoder/denoiser automatically.

## Evaluate on the test set

`sample.py` loads a checkpoint, runs DDIM sampling over the chosen split,
rasterizes the predicted polygons and reports mean **Dice** and **IoU**. The
architecture is read back from the checkpoint, so no model flags are needed.

```bash
python -m src.sample \
    --ckpt    src/runs/my_run/best.pth \   # the trained checkpoint (EMA weights)
    --dataset ph2 \                         # must match what it was trained on
    --split   test \                        # test | val
    --guidance_scale 1.5 \                  # use the same scale as training
    --viz     src/runs/my_run/test_grid.png # optional qualitative grid
```

Prints e.g.:

```
[test] Dice 0.9117 | IoU 0.8413 | viz -> src/runs/my_run/test_grid.png
```

Notes:
- Use `--split val` for the validation partition, `--split test` for the held-out
  test set (see the split table above).
- Keep `--guidance_scale` the same as training; a mismatched scale changes the
  numbers (too high collapses the prediction toward the image borders).
- `--ckpt .../best.pth` evaluates the best validation checkpoint; `last.pth` is the
  final epoch.
- To score programmatically, call `evaluate(encoder, denoiser, diffusion, loader,
  cfg, device)` from `sample.py`, which returns `(mean_dice, mean_iou)`.

## Results (PH2, gs=1.5)

Best model — **ConvNeXt-Tiny + stem**:

| Encoder (best recipe) | epochs | val Dice | test Dice | test IoU |
|---|---|---|---|---|
| **ConvNeXt-Tiny + stem** | 800 | **0.925** | **0.912** | **0.841** |
| PVT-v2-b2 + stem | 200 | 0.824 | 0.765 | 0.635 |

Ablation (PVT, 200 epochs) — contribution of each change to the recipe:

| Recipe | best val Dice | test Dice |
|---|---|---|
| x0-MSE + uniformity (baseline) | 0.795 | — |
| + soft-Dice, per-sample CFG, min-SNR, sigmoid scale-gate | 0.815 | 0.748 |
| + full-resolution stem | 0.824 | 0.765 |

## Notes & gotchas

- **Coordinate convention:** points are `(x, y)` in `[-1, 1]` with the
  `align_corners=True` mapping (`coord/(size-1)*2-1`), consistent across the
  dataset, `grid_sample` in the denoiser, and rasterization. GT points rasterize
  back to the GT mask at Dice ≈ 0.999.
- **Guidance scale** is the single most failure-prone knob — keep it low (see table).
- **Soft-Dice** rasterizes at 64×64 in fp32 (atan2 is touchy under AMP); raise the
  size in `utils/rasterize.py` for a sharper boundary gradient at quadratic cost.
- The method models a **single external contour** (largest contour, no holes);
  it is a binary single-object segmenter, not a multi-class semantic one.
```
