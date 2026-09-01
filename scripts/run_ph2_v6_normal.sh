#!/usr/bin/env bash
set -euo pipefail

# Primary V6-A experiment: Fourier proposal + 1-D normal residual diffusion.
# No dense boundary head and no post-diffusion snapper are active.
PYTHON_BIN=${PYTHON_BIN:-python}

"$PYTHON_BIN" -m src.train \
  --dataset ph2 \
  --encoder timm \
  --backbone pvt_v2_b2 \
  --pretrained true \
  --freeze_backbone false \
  --backbone_lr 2e-5 \
  --lr 1e-4 \
  --n_points 100 \
  --epochs 400 \
  --batch_size 8 \
  --aug_level light \
  --eval_every 10 \
  --eval_seed 314159 \
  --proposal_type fourier \
  --fourier_harmonics 4 \
  --proposal_arclength_resample true \
  --diffusion_target normal_residual \
  --residual_scale 0.25 \
  --pure_diffusion_decoder true \
  --normal_profile_levels 2 \
  --normal_profile_samples 11 \
  --normal_profile_radius_min 0.015 \
  --normal_profile_radius_max 0.18 \
  --guidance_scale 1.0 \
  --cfg_dropout 0.0 \
  --loss_weighting uniform \
  --lambda_proposal_chamfer 1.0 \
  --lambda_proposal_dice 1.0 \
  --lambda_boundary_head 0.0 \
  --lambda_snap_teacher 0.0 \
  --lambda_hard_boundary 0.08 \
  --hard_boundary_fraction 0.15 \
  --ddim_steps 50 \
  --out_dir src/runs/ph2_v6_normal_residual
