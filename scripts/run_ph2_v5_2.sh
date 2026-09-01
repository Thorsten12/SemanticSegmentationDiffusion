#!/usr/bin/env bash
set -euo pipefail

# Controlled V5.2 run: preserve V5 Fourier+XY-residual and replace the external
# snapper with a confidence-gated sparse exact correction inside low-t diffusion.
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
  --diffusion_target residual \
  --fourier_harmonics 4 \
  --proposal_arclength_resample false \
  --residual_scale 1.0 \
  --lambda_proposal_chamfer 1.0 \
  --lambda_proposal_dice 1.0 \
  --guidance_scale 1.0 \
  --cfg_dropout 0.0 \
  --loss_weighting uniform \
  --timesteps 1000 \
  --beta_start 1e-4 \
  --beta_end 2e-2 \
  --ddim_steps 50 \
  --lambda_boundary_head 0.0 \
  --lambda_snap_teacher 0.0 \
  --exact_boundary_enabled true \
  --exact_boundary_levels 2 \
  --exact_boundary_samples 11 \
  --exact_boundary_radius 0.10 \
  --exact_boundary_profile_dim 20 \
  --exact_boundary_hidden 64 \
  --exact_boundary_ring_bands 4 \
  --exact_boundary_relative_bias 0.12 \
  --exact_boundary_confidence_power 2.0 \
  --exact_boundary_low_t_fraction 0.30 \
  --exact_confidence_radius 0.060 \
  --exact_tangent_tolerance 0.040 \
  --exact_use_rgb true \
  --exact_teacher_offset 0.060 \
  --exact_teacher_smooth 2 \
  --lambda_exact_offset 1.0 \
  --lambda_exact_confidence 0.50 \
  --lambda_hard_boundary 0.08 \
  --hard_boundary_fraction 0.15 \
  --out_dir src/runs/ph2_v5_2_exact_boundary
