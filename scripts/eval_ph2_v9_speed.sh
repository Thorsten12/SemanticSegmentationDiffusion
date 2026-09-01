#!/usr/bin/env bash
set -euo pipefail
CKPT="${1:?usage: $0 CHECKPOINT [OUTDIR]}"
OUT="${2:-src/runs/ph2_v9_speed}"
PYTHON_BIN="${PYTHON_BIN:-python}"
mkdir -p "$OUT"
for S in 50 30 20 10 8; do
  "$PYTHON_BIN" -m src.sample --ckpt "$CKPT" --dataset ph2 --split test \
    --ddim_steps "$S" --eval_seed 314159 \
    --viz "$OUT/grid_${S}.png" --metrics "$OUT/cases_${S}.csv" \
    | tee "$OUT/test_${S}.txt"
done
