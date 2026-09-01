#!/usr/bin/env bash
set -euo pipefail
if [[ $# -lt 1 ]]; then
  echo "usage: $0 path/to/best.pth [output_dir]"
  exit 2
fi
CKPT=$1
OUT=${2:-$(dirname "$CKPT")}
PYTHON_BIN=${PYTHON_BIN:-python}
for STEPS in 50 30 20 10 8; do
  "$PYTHON_BIN" -m src.sample \
    --ckpt "$CKPT" --split test --ddim_steps "$STEPS" \
    --viz "$OUT/test_grid_ddim${STEPS}.png" \
    --metrics "$OUT/test_case_metrics_ddim${STEPS}.csv" \
    | tee "$OUT/test_ddim${STEPS}.txt"
done
