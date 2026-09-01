#!/usr/bin/env bash
set -euo pipefail

# V5.1-A: one-time Fourier arc-length reparameterization plus normalized
# residual diffusion.  Only residual_scale differs across the three runs.

COMMON=(
  --dataset ph2
  --encoder timm
  --backbone pvt_v2_b2
  --pretrained true
  --freeze_backbone false
  --backbone_lr 2e-5
  --lr 1e-4
  --n_points 100
  --epochs 400
  --batch_size 8
  --aug_level light
  --eval_every 10
  --eval_seed 314159
  --guidance_scale 1.0
  --cfg_dropout 0.0
  --loss_weighting uniform
  --lambda_hard_boundary 0.08
  --hard_boundary_fraction 0.15
  --snap_teacher_hard_weight 0.50
  --snap_geometry_warmup_epochs 40
  --snap_geometry_ramp_epochs 40
  --proposal_type fourier
  --diffusion_target residual
  --fourier_harmonics 4
  --proposal_arclength_resample true
  --lambda_proposal_chamfer 1.0
  --lambda_proposal_dice 1.0
)

PYTHON_BIN=/home/boa52995/venvs/pytorch-p310/bin/python

for SCALE in 0.25 0.35 0.50; do
  TAG=${SCALE/./p}
  "$PYTHON_BIN" -m src.train "${COMMON[@]}" \
    --residual_scale "$SCALE" \
    --out_dir "src/runs/ph2_v5_1_fourier_residual_s${TAG}"
done
