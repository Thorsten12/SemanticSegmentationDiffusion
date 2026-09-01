#!/usr/bin/env bash
set -euo pipefail
PYTHON_BIN="${PYTHON_BIN:-python}"
CKPT="${1:-src/runs/ph2_v9_1_protected_boundary/best.pth}"
OUT="${2:-src/runs/ph2_v9_1_protected_boundary}"
for STEPS in 50 30 20 10 8; do
  "$PYTHON_BIN" -m src.sample \
    --ckpt "$CKPT" --dataset ph2 --split test \
    --ddim_steps "$STEPS" --eval_seed 314159 \
    --viz "$OUT/test_grid_ddim${STEPS}.png" \
    --metrics "$OUT/test_case_metrics_ddim${STEPS}.csv"
done
