# P2SDiff — Boundary-Point Diffusion for Segmentation (V6)

A clean, self-contained implementation of **segmentation-as-boundary-point-generation**.

Instead of denoising per-pixel intensities, the diffusion process runs on the **2D coordinates of `N` ordered boundary points**; the denoised polygon is rasterized into a binary mask and scored with Dice / IoU.

This package is independent of the older `scr/` code — nothing here imports from it.

## Idea

```text
mask
  │
  └── contour ──► N ordered points
                   (arc-length uniform, [-1,1])
                   │
                   │ ground-truth x₀
                   ▼
            forward q(xₜ|x₀)
            add Gaussian noise
            (DDPM, T=1000)
                   │
                   │
image ─► backbone ─► multi-scale feature pyramid
                   │
                   │ extracted ONCE per image
                   ▼
        optional (V5/V6):
        ContourProposalHead
                   │
                   └──► deterministic coarse-contour proposal
                        (Fourier harmonics)
                        supplies global position/size/coarse shape
                   │
                   ▼
        reverse diffusion:
        ContourDenoiser predicts x₀
        (or residual on top of proposal)
                   │
                   ├── reads multi-scale features per point
                   │   via swappable sampler (`sampler_type`)
                   │
                   └── time-gated across scales
                   │
                   ▼
        DDIM
        deterministic, configurable steps
        + classifier-free guidance
                   │
                   ▼
        optional (V6):
        BoundarySnapper
        post-DDIM (or in-loop, low-t only)
        normal-profile boundary correction
        confidence-gated
                   │
                   ▼
              fillPoly
                   │
                   ▼
             binary mask
                   │
                   ▼
             Dice / IoU
```


- **Parameterization:** `x0`-prediction. With `--proposal_target residual` (V5/V6
  default), the denoiser predicts only the *residual* on top of a deterministic
  coarse-contour proposal (`ContourProposalHead`) instead of the full contour —
  see "Deterministic coarse-contour proposal (V5/V6)" below.
- **Conditioning (image only, RGB):** a pretrained backbone (ConvNeXt, PVT-v2,
  ResNet, Swin, VMamba — see `models/encoder.py`) produces a multi-scale feature
  pyramid. The denoiser reads **every scale per point** via one of three
  interface-compatible, mutually-exclusive **samplers** (`--sampler_type`, see
  table below), and **time-gates** the scales with a forced cosine schedule
  (coarse context early in sampling, fine edges late) plus a small bounded
  learned correction. The heavy backbone runs once per image; the per-point
  query/gate runs each diffusion step.
- **Loss:** per-sample **min-SNR-γ weighted x0 MSE**, plus several optional
  auxiliary terms — differentiable **soft-Dice**, **Boundary-IoU**, an
  asymmetric **nearest-point** term, a **uniformity** regularizer, and (V6,
  experimental) a **curvature penalty** — each individually toggleable and
  independently time-weighted (see "Loss terms" below).
- **Tricks:** EMA (with warmup), **per-sample** classifier-free-guidance dropout
  in training + CFG at sampling, deterministic DDIM, an optional (V6)
  **sensitivity regularizer** that penalizes the denoiser for ignoring the
  actual noise realization.
- **Augmentation:** synchronous flips / affine (rotation, translate, scale,
  shear) on image+mask with a validity-retry loop, plus image-only colour
  jitter / blur. Boundary points are always recomputed *from the augmented
  mask*, so image and points can never drift out of alignment.

## Layout

| File | Role |
|------|------|
| `config.py` | all hyperparameters (`Config` dataclass) |
| `data/seg_datasets.py` | multi-dataset loader with **published index splits** (reads preprocessed npy) |
| `data/ph2_dataset.py` | contour dataset: augmentation + arc-length contour sampling (`ArrayContourDataset`) |
| `data/skin/`, `data/dataset_*.py` | reference per-dataset loaders the splits follow |
| `models/encoder.py` | conditioning encoders: ConvNeXt / PVT / ResNet / Swin / VMamba conditioners |
| `models/pvtv2.py` | local Pyramid Vision Transformer v2 backbone |
| `models/feature_unet.py` | from-scratch image → full-res condition map (the `unet` encoder) |
| `models/denoiser.py` | `ContourDenoiser` + the three multi-scale samplers (`MultiScalePointSampler`, `DeformableScalePointSampler`, `AttentionScalePointSampler`) |
| `models/proposal.py` | `ContourProposalHead` — deterministic coarse-contour proposal (V5/V6) |
| `models/snapper.py` | `BoundarySnapper` — post-DDIM / in-loop normal-profile boundary correction (V6) |
| `models/attention.py`, `models/positional.py` | shared attention/MLP blocks; Fourier coordinate encodings |
| `diffusion/gaussian_diffusion.py` | schedule, `q_sample`, training losses, DDIM sampling (CFG, in-loop snapping) |
| `utils/rasterize.py` | `points_to_mask` (cv2 eval) + `soft_rasterize` / `soft_dice_loss` / `soft_boundary_iou_loss` (differentiable) + Dice/IoU |
| `utils/ema.py`, `utils/viz.py` | EMA with warmup; prediction-grid visualization |
| `train.py` | training loop (CLI) |
| `sample.py` | evaluation / inference (CLI) + `evaluate()` |
| `baseline_seg.py` | non-diffusion encoder-decoder baseline (swappable decoder) for head-to-head comparison |

## Conditioning encoders (`--encoder`)

| Encoder | Backbone | Notes |
|---|---|---|
| `convnext` | timm ConvNeXt-Tiny/Small/Base (ImageNet) | default; 5-stage pyramid |
| `pvt` | PVT-v2 (`models/pvtv2.py`, variant configurable) | 5-stage pyramid |
| `resnet` | ResNet (ImageNet) | |
| `swin` | Swin Transformer (ImageNet) | |
| `vmamba` | VMamba | requires a local checkpoint (see `encoder.py`) |
| `unet` | from-scratch `FeatureUNet` | single full-res map (ablation) |

Pretrained backbones fine-tune at `--backbone_lr` (default 1e-5) or freeze with
`--freeze_backbone`.

## Multi-scale point samplers (`--sampler_type`)

The denoiser conditions each contour point on the full multi-scale feature
pyramid. How it reads and fuses that pyramid is swappable via `--sampler_type`,
a three-way, mutually-exclusive ablation — all three share the identical
forced cosine time-gate prior (which scale dominates at which timestep never
changes across the three), so the ablation isolates exactly one axis: **how**
each scale's local evidence is aggregated.

| `--sampler_type` | Class | Behavior |
|---|---|---|
| `input` (default) | `MultiScalePointSampler` | One aligned `grid_sample` per scale per point — the original, single-location read. |
| `deformable` | `DeformableScalePointSampler` | `--deform_n_samples` reads per scale (1 aligned + learned-offset samples within `--deform_radius_cells`); both the offsets and the fusion weights are predicted blind from the point/time token, before seeing any sampled content. |
| `attention` | `AttentionScalePointSampler` | Reads a *fixed* local neighborhood (`--attn_sampler_window`, nearest `--deform_n_samples` cells kept), fused via real scaled-dot-product attention — Query from the point/time token, Key/Value from the actually-sampled content. The model sees what's at each candidate before deciding how much to trust it. |

`--deform_min_scale_res` (default 28) gates which scales get a local
neighborhood at all (`deformable`/`attention`) vs. a single aligned sample
(very coarse scales have little sub-grid neighborhood worth exploiting).

### Scale-count reduction (`--top_k_scales`)

All three samplers additionally accept `--top_k_scales`. When set (smaller
than the encoder's actual scale count), the sampler still computes **every**
scale in full — no compute is skipped — but only the `top_k_scales` scales
with the strongest time-gate weight (per sample) are gathered into the final
fusion MLP, which shrinks accordingly (`Linear(top_k_scales * proj_dim + ...)`
instead of `Linear(n_scales * proj_dim + ...)`). This is a genuine
architecture change (not just a masked-out subset of an unchanged network),
so checkpoints are **not** compatible across different `--top_k_scales`
values. `None` (default) disables this and keeps all scales in the fusion.

## Deterministic coarse-contour proposal (`--proposal_target`, V5/V6)

`ContourProposalHead` predicts a deterministic coarse contour directly from
the coarsest backbone features (Fourier-harmonic parameterization,
`--proposal_harmonics`). With `--proposal_target residual` (default), the
denoiser's own head is zero-initialized to predict a pure **delta on top of
the proposal** (`x0 = proposal + denoiser_out`), so training starts at
exactly the proposal and the model only ever has to learn *local* corrections
— it never has to re-derive global position/size/coarse-shape from noise the
way plain x0-prediction implicitly requires. `--proposal_target absolute`
disables the proposal (denoiser predicts the full contour, pre-V5 behavior).

## Boundary snapper (V6, `--lambda_snap`, `--snap_mode`)

`BoundarySnapper` is a small, separately-trained module that reads a short
1-D profile along each point's local normal from the raw backbone features
(+ optional RGB) and predicts a confidence-gated normal offset — a final,
exact-boundary correction on top of whatever the diffusion path produced.
Trained on its own teacher loss (`--lambda_snap`) against artificially
normal-perturbed ground truth, **not** on real diffusion rollouts (which are
often far from GT early in training and would collapse the confidence head
to "always uncertain"). `--snap_mode` controls whether it runs once after
DDIM finishes (`post`), inside the sampling loop on low-noise steps only
(`loop`, gated by `--snap_t_threshold_frac`), both (`both`), or not at all
(`none`).

## Loss terms (`diffusion.training_losses`)

Every term below is only computed when its `lambda` is > 0, and each is
independently time-weighted (some strongest at high `t` / pure noise, some at
low `t` / near-clean) — see `diffusion.py` docstrings for the exact schedule
of each.

| Flag | Term | Time-weighting |
|---|---|---|
| `--lambda_mse` | min-SNR-γ weighted x0 MSE | per-sample SNR-clamped |
| `--lambda_boundary` | boundary-attention-weighted x0 MSE | — |
| `--lambda_dice` | differentiable soft-Dice (winding-number rasterization) | strong at high `t` (coarse shape) |
| `--lambda_biou` | differentiable Boundary-IoU (thin boundary band) | strong at low `t` (fine edge) |
| `--lambda_nearest` | asymmetric nearest-neighbor (pred → GT) | strong at low `t` |
| `--lambda_uniformity` | contour point spacing regularizer | — |
| `--lambda_curvature` | (V6, experimental) self-only curvature penalty on the prediction — no GT comparison, a pure smoothness constraint | strong at high `t` |
| `--lambda_proposal_dice` | soft-Dice directly on the coarse proposal | — |
| `--lambda_sensitivity` | (V6) penalizes low pred_x0 sensitivity to the actual noise realization at fixed `(x0, t, image)` | — |

Note on `--lambda_curvature`: with `--proposal_target residual` active, this
term as currently wired measures the curvature of `proposal + denoiser_out`,
not the denoiser's own residual — since the Fourier proposal is already very
smooth, the term is easily dominated/diluted by the proposal's smoothness and
may show near-zero values even at high weight. Isolating it to the raw
residual (pre-proposal-addition) is a known follow-up, not yet wired in
`train.py`.

## Data

Datasets are read from the shared **preprocessed npy** under
`<skin_root>/<DATASET>/np/X_tr_{S}x{S}.npy` (+ `Y_`), in glob order, and split by
the **published index ranges** so our partition matches the reference loaders
exactly — only the target differs (we derive boundary points from each mask).
See `data/seg_datasets.py`.

| `--dataset` | train / val / test | total | notes |
|---|---|---|---|---|
| `ph2` | — | 200 | **zero-shot test set only** — no tr/vl split; train on `isic2018` instead |
| `isic2017` | 1250 / 150 / 600 | 2000 | |
| `isic2018` | 1815 / 259 / 520 | 2594 | |
| `ham10000` | 7200 / 1800 / 1015 | 10015 | |
| `busi` | seeded 70/10/20 | — | |
| `tn3k` | seeded 80/20 tr/vl + official test | — | |
| `polyp_clinicdb`/`polyp_kvasir`/`polyp_colondb`/`polyp_etis`/`polyp_cvc300` | shared PraNet tr/vl, own test | — | 5 separate test sets |

Defaults: `--skin_root /hdd/datasets/Skin`, `--npy_size 224`, `img_size 224`.

## Usage

All commands run from the repository root.

```bash
# train: ConvNeXt-Tiny backbone, deterministic proposal + residual diffusion
# + boundary snapper (the current recommended V6 recipe) on ISIC-2018
python -m x0_prediction.train --dataset isic2018 --encoder convnext \
    --guidance_scale 1.5 --epochs 300 \
    --proposal_target residual --proposal_type fourier \
    --snap_mode both --lambda_snap 2.0 \
    --sampler_type attention --deform_n_samples 4 --attn_sampler_window 5 \
    --out_dir x0_prediction/runs/my_run

# PH2 is evaluated automatically as a zero-shot test at the end of an
# isic2018 run — see the PH2 zero-shot block in train.py / baseline_seg.py.

# ablation: swap the multi-scale sampler
python -m x0_prediction.train --dataset isic2018 --sampler_type deformable \
    --deform_n_samples 8 --deform_radius_cells 2.5 --out_dir .../deform_run

python -m x0_prediction.train --dataset isic2018 --sampler_type input \
    --out_dir .../legacy_sampler_run   # byte-for-byte old MultiScalePointSampler behavior

# baseline (no diffusion) for head-to-head comparison
python -m baseline_seg --dataset isic2018 --encoder convnext --decoder unet \
    --out_dir baseline_models/runs/my_baseline
```

Run long jobs in the background and watch the log:

```bash
nohup python -m x0_prediction.train --dataset isic2018 --encoder convnext \
    --epochs 300 --out_dir x0_prediction/runs/my_run > runs_my_run.log 2>&1 &
grep "val Dice" runs_my_run.log     # validation (EMA weights) every --eval_every epochs
```

### Useful flags

| Flag | Default | Notes |
|---|---|---|
| `--guidance_scale` | 2.0 | over-guiding (≥5) collapses points to image borders |
| `--encoder` | convnext | `convnext` / `pvt` / `resnet` / `swin` / `vmamba` / `unet` |
| `--sampler_type` | input | `input` / `deformable` / `attention` — see table above |
| `--top_k_scales` | None | restrict the fusion MLP to the `k` strongest-gated scales |
| `--proposal_target` | residual | `residual` (V5/V6) / `absolute` (pre-V5) |
| `--snap_mode` | post | `post` / `loop` / `both` / `none` |
| `--lambda_dice` | 1.0 | weight of the differentiable soft-Dice term |
| `--lambda_biou` | 1.0 | weight of the differentiable Boundary-IoU term |
| `--snr_gamma` | 5.0 | min-SNR-γ cap for the per-sample x0 weight |
| `--aug_level` | light | `none` / `light` / `strong` |
| `--freeze_backbone` | off | train only stem + fusion + denoiser |
| `--backbone_lr` | 1e-5 | discriminative LR for the pretrained backbone |
| `--eval_every` | 20 | epochs between validations |
| `--guidance_time_scale` | 0.0 | (V7 ablation) linearly damps guidance amplitude with `t` |
| `--lambda_sensitivity` | 0.0 | (V7 ablation) penalizes low noise-sensitivity of pred_x0 |

(`stem_dim`, `n_points`, diffusion schedule, etc. live in `config.py`.)

Artifacts per run land in `--out_dir`: `best.pth`, `last.pth`, `history.json`,
`loss_curve.png`, `scale_gates.png` (per-timestep-bin scale-gate importance
over training), periodic `val_epochN.png` grids, and `summary.json` /
`test_metrics.json` at the end of training — the latter includes per-component
(`encoder`/`denoiser`/`snapper`) parameter counts, FLOPs, latency, and peak
memory (`model_stats`, mirroring `baseline_seg.py`'s comparison summary).
Checkpoints store the architecture config, so `sample.py` rebuilds the
matching encoder/denoiser/snapper/proposal-head automatically.

## Evaluate on the test set

`sample.py` loads a checkpoint, runs DDIM sampling over the chosen split,
rasterizes the predicted polygons and reports mean **Dice** and **IoU**. The
architecture is read back from the checkpoint, so no model flags are needed.

```bash
python -m x0_prediction.sample \
    --ckpt    x0_prediction/runs/my_run/best.pth \
    --dataset isic2018 \
    --split   test \
    --guidance_scale 1.5 \
    --viz     x0_prediction/runs/my_run/test_grid.png
```

Notes:
- Use `--split val` for the validation partition, `--split test` for the
  held-out test set (see the split table above).
- `ph2` cannot be passed to `--split`/`--dataset` for training (no tr/vl); it
  is only ever an additional, unseen zero-shot evaluation run automatically
  at the end of an `isic2018` training run.
- Keep `--guidance_scale` the same as training; a mismatched scale changes the
  numbers (too high collapses the prediction toward the image borders).
- `--ckpt .../best.pth` evaluates the best validation checkpoint; `last.pth`
  is the final epoch.
- To score programmatically, call `evaluate(encoder, denoiser, diffusion,
  loader, cfg, device, snapper=..., proposal_head=...)` from `sample.py`,
  which returns `(mean_dice, mean_iou)`.

## Results

_Benchmark numbers for the current V6 recipe (proposal + residual diffusion +
snapper, `sampler_type`/`top_k_scales` ablations) are pending — the previous
README's PH2 table reflected an earlier architecture (no proposal head, no
snapper, single-sample `grid_sample` conditioning) and no longer applies.
This section will be filled in once the full multi-dataset benchmark
(ISIC-2018, PH2 zero-shot, BUSI, polyp, TN3K) is finalized — see
`x0_prediction/runs/*/summary.json` / `test_metrics.json` for individual run
results in the meantime._

## Notes & gotchas

- **Coordinate convention:** points are `(x, y)` in `[-1, 1]` with the
  `align_corners=True` mapping (`coord/(size-1)*2-1`), consistent across the
  dataset, `grid_sample` in the denoiser, and rasterization. GT points
  rasterize back to the GT mask at Dice ≈ 0.999.
- **Guidance scale** is one of the most failure-prone knobs — keep it low
  (start around 1.5–2.0).
- **`top_k_scales` and `sampler_type` changes alter the network architecture**
  (fusion MLP input dimension) — checkpoints trained with one setting cannot
  be warm-started or loaded into a differently-configured model.
- **PH2 is zero-shot-only** — it has no tr/vl split by design and must never
  be used for training or checkpoint selection; both `train.py` and
  `baseline_seg.py` raise a hard error if you try.
- **Soft-Dice/Boundary-IoU** rasterize at a configurable size (`--soft_dice_size`,
  default 64) in fp32 (atan2 is touchy under AMP); raising it sharpens the
  boundary gradient at quadratic cost.
- The method models a **single external contour** (largest contour, no
  holes); it is a binary single-object segmenter, not a multi-class semantic
  one.