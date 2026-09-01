# P2SDiff V5 — Spatial Proposal + Residual Contour Diffusion

V5 keeps the proven V3.1 encoder, global/local decoder, boundary head, normal
snapper, N=100 representation, and geometry curriculum.  It isolates the
remaining coarse-localization bottleneck with two independently selectable
changes:

1. `proposal_type=fourier`: a learned shape query cross-attends to the global
   feature memory **with 2-D positional encoding** and predicts a four-harmonic
   low-frequency contour instead of the radius-limited ellipse.
2. `diffusion_target=residual`: diffusion predicts
   `GT contour - deterministic proposal`, so global location/scale is
   deterministic and only the remaining boundary displacement is generative.

The Fourier proposal receives direct symmetric-Chamfer and coarse soft-Dice
supervision.  The three controlled ablations are available in one codebase:

```text
ellipse + absolute  = V3.1-compatible architectural baseline
fourier + absolute  = spatial-proposal-only ablation
fourier + residual  = complete proposed model
```

Run all three PVT-v2-b2/PH2 experiments with:

```bash
bash scripts/run_ph2_v5_ablation.sh
```

CPU smoke checks for the three paths:

```bash
python -m src.smoke_test --proposal_type ellipse --diffusion_target absolute
python -m src.smoke_test --proposal_type fourier --diffusion_target absolute
python -m src.smoke_test --proposal_type fourier --diffusion_target residual
```

Every validation checkpoint and final test evaluation writes a
`*_case_metrics.csv` containing lesion-area fraction, border contact, proposal
Dice, pre-snap coarse Dice, final Dice, and IoU. Analyze the final test file via:

```bash
python scripts/analyze_size_bias.py \
  src/runs/ph2_v5_fourier_residual/test_case_metrics.csv
```

To run the original V3.1 behavior from this source, use:

```bash
python -m src.train --proposal_type ellipse --diffusion_target absolute \
  --lambda_proposal_chamfer 0 --lambda_proposal_dice 0 [other V3.1 arguments]
```

## V3.1 baseline retained below

V3.1 is a **conservative branch from V3**, not from V4/V4.1.

The reason is empirical: on PH2 the original lightweight V3 reached **0.935 test Dice** from scratch (5.6M U-Net encoder), while V4.1 dropped badly. Therefore V3.1 preserves the exact successful V3 model/refiner and changes only training supervision around the hard boundary vertices.

## What is unchanged from V3

- same 2-D encoder and global-to-local point decoder;
- same bounded latent contour diffusion;
- same explicit boundary head;
- same V3 normal-profile snapper;
- **no confidence gate**;
- **no arc-length reparameterization inside the snapper**;
- **no pre-snap head**;
- **no area loss**;
- **no synthetic border-touch augmentation**;
- same inference path and N=100 contour representation.

With the from-scratch U-Net encoder (`unet_start_dim=64`) the full network remains about **5.605M parameters**.

## V3.1 changes

### 1. Small hard-vertex loss

V3 already has a symmetric HD-like top-k term. V3.1 adds a small **one-sided predicted-vertex** top-k term:

```text
lambda_hard_boundary = 0.08
hard_boundary_fraction = 0.15
```

This specifically targets the few bad predicted vertices that can create polygon spikes while leaving the rest of the contour untouched.

### 2. Correspondence-free snapper teacher loss

V3 trained the snapper teacher with index-wise Smooth-L1:

```text
teacher_out[i] <-> gt[i]
```

V3.1 replaces that with:

```text
symmetric Chamfer + small top-k predicted-boundary distance
```

So a point is rewarded for landing on the correct boundary even if its exact contour index differs slightly.

### 3. Geometry curriculum

During the first epochs, Dice/Chamfer/HD geometry losses use the stable **coarse diffusion contour** while the snapper learns separately from the teacher path.

Then snapped geometry is blended in gradually:

```text
epoch 1..60      : alpha = 0  (coarse geometry)
epoch 61..120    : alpha 0 -> 1
epoch >120       : alpha = 1  (exact V3 snapped geometry objective)
```

The diffusion x0 target itself is unchanged throughout.

## Install / check

```bash
pip install -r requirements.txt
python -m src.smoke_test
python -m src.model_info --encoder unet --unet_start_dim 64
```

Expected model size is approximately:

```text
encoder: 4.048 M
point decoder + boundary refiner: 1.557 M
total: 5.605 M
```

# Recommended PH2 runs

## A. Main from-scratch run — 800 epochs

This is the clean comparison with the previous V3 e800 result.

```bash
python -m src.train \
  --dataset ph2 \
  --encoder unet \
  --pretrained false \
  --unet_start_dim 64 \
  --n_points 100 \
  --epochs 800 \
  --batch_size 8 \
  --lr 2e-4 \
  --aug_level light \
  --eval_every 10 \
  --guidance_scale 1.0 \
  --cfg_dropout 0.0 \
  --loss_weighting uniform \
  --lambda_hard_boundary 0.08 \
  --hard_boundary_fraction 0.15 \
  --snap_teacher_hard_weight 0.50 \
  --snap_geometry_warmup_epochs 60 \
  --snap_geometry_ramp_epochs 60 \
  --out_dir src/runs/ph2_v3_1_scratch_e800
```

A ready-to-run shell script is included as `run_ph2_scratch.sh`.

## B. PVT-v2-B1 ImageNet — full end-to-end fine-tuning

The backbone is initialized from ImageNet but **not frozen**. All encoder and decoder weights are fine-tuned.

```bash
python -m src.train \
  --dataset ph2 \
  --encoder timm \
  --backbone pvt_v2_b1 \
  --pretrained true \
  --freeze_backbone false \
  --backbone_lr 2e-5 \
  --lr 1e-4 \
  --n_points 100 \
  --epochs 500 \
  --batch_size 8 \
  --aug_level light \
  --eval_every 10 \
  --guidance_scale 1.0 \
  --cfg_dropout 0.0 \
  --loss_weighting uniform \
  --lambda_hard_boundary 0.08 \
  --hard_boundary_fraction 0.15 \
  --snap_teacher_hard_weight 0.50 \
  --snap_geometry_warmup_epochs 40 \
  --snap_geometry_ramp_epochs 40 \
  --out_dir src/runs/ph2_v3_1_pvt_b1_imagenet
```

A ready-to-run shell script is included as `run_ph2_pvt_b1.sh`.

## C. PVT-v2-B2 ImageNet — stronger encoder ablation

If GPU memory permits, B2 is also included as a script: `run_ph2_pvt_b2.sh`.

For a local PVT checkpoint instead of timm:

```bash
python -m src.train \
  --dataset ph2 \
  --encoder pvt \
  --pvt_variant pvt_v2_b1 \
  --pretrained true \
  --pvt_pretrained_path /path/to/pvt_v2_b1.pth \
  --freeze_backbone false \
  --backbone_lr 2e-5 \
  --lr 1e-4 \
  --n_points 100 \
  --epochs 500 \
  --out_dir src/runs/ph2_v3_1_pvt_b1_local
```

# Evaluation

```bash
python -m src.sample \
  --ckpt src/runs/ph2_v3_1_scratch_e800/best.pth \
  --dataset ph2 \
  --split test \
  --ddim_steps 50 \
  --viz src/runs/ph2_v3_1_scratch_e800/test_grid_eval.png
```

Single-image inference:

```bash
python -m src.infer \
  --ckpt src/runs/ph2_v3_1_scratch_e800/best.pth \
  --image /path/to/image.jpg \
  --out_mask predicted.png
```

## Important ablation switches

To recover the original V3-style geometry behavior while keeping the set-based teacher loss:

```bash
--snap_geometry_warmup_epochs 0 --snap_geometry_ramp_epochs 0
```

To disable the new hard-vertex loss:

```bash
--lambda_hard_boundary 0
```

This makes V3.1 easy to diagnose without changing the model architecture.

## Validation performed before packaging

The build environment does not contain `/hdd/datasets`, so no new PH2 number is claimed. The package was checked with:

- `python -m compileall src`
- `python -m src.smoke_test`
- `python -m src.train --help`
- `python -m src.sample --help`
- model parameter count for the 5.605M from-scratch configuration

The important comparison is now V3 e800 (0.935 test Dice) vs. V3.1 e800 on the same PH2 split.
