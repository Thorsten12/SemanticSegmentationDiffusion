#!/usr/bin/env bash
set -euo pipefail
PYTHON_BIN=${PYTHON_BIN:-python}
CKPT=${1:-src/runs/ph2_v6_normal_residual/best.pth}
OUT=${2:-src/runs/ph2_v6_normal_residual}

for STEPS in 50 20 10 8; do
  "$PYTHON_BIN" -m src.sample \
    --ckpt "$CKPT" --dataset ph2 --split test \
    --ddim_steps "$STEPS" --eval_seed 314159 \
    --viz "$OUT/test_grid_${STEPS}step.png" \
    --metrics "$OUT/test_case_metrics_${STEPS}step.csv" \
    | tee "$OUT/test_${STEPS}step.log"
done
