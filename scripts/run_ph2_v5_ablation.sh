#!/usr/bin/env bash
set -euo pipefail

# Controlled PH2 ablation: only proposal_type and diffusion_target differ.
# All runs retain V3.1 PVT-v2-b2, N=100, curriculum, and boundary snapper.

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
  --guidance_scale 1.0
  --cfg_dropout 0.0
  --loss_weighting uniform
  --lambda_hard_boundary 0.08
  --hard_boundary_fraction 0.15
  --snap_teacher_hard_weight 0.50
  --snap_geometry_warmup_epochs 40
  --snap_geometry_ramp_epochs 40
)

python -m src.train "${COMMON[@]}" \
  --proposal_type ellipse \
  --diffusion_target absolute \
  --lambda_proposal_chamfer 0 \
  --lambda_proposal_dice 0 \
  --out_dir src/runs/ph2_v5_ellipse_absolute

python -m src.train "${COMMON[@]}" \
  --proposal_type fourier \
  --diffusion_target absolute \
  --fourier_harmonics 4 \
  --lambda_proposal_chamfer 1.0 \
  --lambda_proposal_dice 1.0 \
  --out_dir src/runs/ph2_v5_fourier_absolute

python -m src.train "${COMMON[@]}" \
  --proposal_type fourier \
  --diffusion_target residual \
  --fourier_harmonics 4 \
  --lambda_proposal_chamfer 1.0 \
  --lambda_proposal_dice 1.0 \
  --out_dir src/runs/ph2_v5_fourier_residual
