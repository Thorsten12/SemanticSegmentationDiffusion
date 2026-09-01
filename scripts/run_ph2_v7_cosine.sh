#!/usr/bin/env bash
set -euo pipefail
PYTHON_BIN=${PYTHON_BIN:-python}
"$PYTHON_BIN" -m src.train \
  --dataset ph2 --encoder timm --backbone pvt_v2_b2 --pretrained true \
  --freeze_backbone false --backbone_lr 2e-5 --lr 1e-4 \
  --n_points 100 --hidden_dim 112 --epochs 400 --batch_size 8 \
  --aug_level light --eval_every 10 --eval_seed 314159 \
  --noise_schedule cosine --ddim_steps 50 \
  --normal_scale 0.25 --tangent_scale 0.07 \
  --profile_samples 9 --profile_levels 2 \
  --low_t_fraction 0.50 --low_t_max_fraction 0.20 \
  --guidance_scale 1.0 --cfg_dropout 0.0 \
  --out_dir src/runs/ph2_v7_sparse_contour_dit_cosine
