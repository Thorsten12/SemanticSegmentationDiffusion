#!/usr/bin/env bash
set -euo pipefail
CKPT=${1:-src/runs/ph2_v8_boundary_residual/best.pth}
OUT=${2:-src/runs/ph2_v8_boundary_residual}
PYTHON_BIN=${PYTHON_BIN:-python}
for S in 50 30 20 10 8; do
  "$PYTHON_BIN" -m src.sample --ckpt "$CKPT" --dataset ph2 --split test     --ddim_steps "$S" --eval_seed 314159     --viz "$OUT/test_grid_ddim${S}.png"     --metrics "$OUT/test_case_metrics_ddim${S}.csv"
done
