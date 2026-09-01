#!/usr/bin/env bash
set -euo pipefail
PYTHON_BIN=${PYTHON_BIN:-python}
"$PYTHON_BIN" -m src.train \
  --dataset ph2 \
  --encoder unet \
  --pretrained false \
  --unet_start_dim 64 \
  --hidden_dim 112 \
  --n_points 100 \
  --epochs 500 \
  --batch_size 8 \
  --lr 2e-4 \
  --aug_level light \
  --eval_every 10 \
  --eval_seed 314159 \
  --noise_schedule linear \
  --ddim_steps 50 \
  --normal_scale 0.25 \
  --tangent_scale 0.07 \
  --profile_samples 9 \
  --profile_levels 2 \
  --out_dir src/runs/ph2_v7_scratch
