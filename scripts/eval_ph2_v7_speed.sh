#!/usr/bin/env bash
set -euo pipefail
CKPT=${1:?usage: eval_ph2_v7_speed.sh /path/to/best.pth}
PYTHON_BIN=${PYTHON_BIN:-python}
OUT=$(dirname "$CKPT")
for STEPS in 50 20 10 8; do
  "$PYTHON_BIN" -m src.sample --ckpt "$CKPT" --dataset ph2 --split test \
    --ddim_steps "$STEPS" --eval_seed 314159 \
    --metrics "$OUT/test_cases_ddim${STEPS}.csv" \
    --viz "$OUT/test_grid_ddim${STEPS}.png" | tee "$OUT/eval_ddim${STEPS}.log"
done
