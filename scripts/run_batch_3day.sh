#!/usr/bin/env bash
# Unattended ~3-day experiment batch for afshin / P2SDiff on RTX A5000.
# Continues past failures; skips jobs that already have summary.json.
#
#   nohup bash scripts/run_batch_3day.sh > src/runs/batch_3day/nohup.out 2>&1 &
#
# Rough GPU budget (A5000): ~40–55h useful work (fits a 3-day window).

set -u
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

VENV="${VENV:-$HOME/venvs/pytorch/bin/activate}"
# shellcheck disable=SC1090
source "$VENV"

export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=4
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"

BATCH_DIR="src/runs/batch_3day"
mkdir -p "$BATCH_DIR"
MASTER_LOG="$BATCH_DIR/master.log"
STATUS_JSON="$BATCH_DIR/job_status.json"
SUMMARY_MD="$BATCH_DIR/RESULTS.md"

ts() { date '+%Y-%m-%d %H:%M:%S'; }
log() { echo "[$(ts)] $*" | tee -a "$MASTER_LOG"; }

record_status() {
  local name="$1" status="$2" extra="${3:-}"
  python - "$STATUS_JSON" "$name" "$status" "$extra" <<'PY'
import json, sys, os, time
path, name, status, extra = sys.argv[1:5]
data = {}
if os.path.isfile(path):
    try:
        data = json.load(open(path))
    except Exception:
        data = {}
    if not isinstance(data, dict):
        data = {}
entry = {"status": status, "updated": time.strftime("%Y-%m-%d %H:%M:%S")}
if extra:
    entry["detail"] = extra
data[name] = entry
json.dump(data, open(path, "w"), indent=2)
PY
}

run_job() {
  local name="$1"; shift
  local out_dir="src/runs/${name}"
  local log_file="$BATCH_DIR/${name}.log"

  if [[ -f "$out_dir/summary.json" ]]; then
    log "SKIP $name (summary.json already exists)"
    record_status "$name" "skipped_exists"
    return 0
  fi

  mkdir -p "$out_dir"
  log "START $name"
  record_status "$name" "running"

  local rc=0
  python -m src.train --out_dir "$out_dir" "$@" >"$log_file" 2>&1
  rc=$?

  if [[ $rc -eq 0 ]]; then
    log "DONE  $name (ok)"
    record_status "$name" "ok"
  else
    log "FAIL  $name (exit $rc) — see $log_file"
    record_status "$name" "failed" "exit=$rc"
  fi
  return 0
}

log "======= BATCH START root=$ROOT gpu=${CUDA_VISIBLE_DEVICES} ======="
python - <<'PY' | tee -a "$MASTER_LOG"
import torch
print("torch", torch.__version__, "cuda", torch.cuda.is_available())
if torch.cuda.is_available():
    print("device", torch.cuda.get_device_name(0))
PY

CN=(
  --encoder convnext --backbone convnext_tiny
  --batch_size 16 --num_workers 4
  --scheduler cosine --warmup_epochs 5
  --aug_level light --ddim_steps 50
)

# =============================================================================
# Phase A — ISIC2017 hyperparameter sweep (short, comparable)
# =============================================================================
run_job isic17_base_lr1e4_gs2 \
  "${CN[@]}" --dataset isic2017 --epochs 100 --eval_every 10 \
  --lr 1e-4 --weight_decay 1e-4 --guidance_scale 2.0 --backbone_lr 1e-5

run_job isic17_lr3e4 \
  "${CN[@]}" --dataset isic2017 --epochs 100 --eval_every 10 \
  --lr 3e-4 --weight_decay 1e-4 --guidance_scale 2.0 --backbone_lr 3e-5

run_job isic17_lr3e5 \
  "${CN[@]}" --dataset isic2017 --epochs 100 --eval_every 10 \
  --lr 3e-5 --weight_decay 1e-4 --guidance_scale 2.0 --backbone_lr 3e-6

run_job isic17_wd1e3 \
  "${CN[@]}" --dataset isic2017 --epochs 100 --eval_every 10 \
  --lr 1e-4 --weight_decay 1e-3 --guidance_scale 2.0 --backbone_lr 1e-5

run_job isic17_wd1e5 \
  "${CN[@]}" --dataset isic2017 --epochs 100 --eval_every 10 \
  --lr 1e-4 --weight_decay 1e-5 --guidance_scale 2.0 --backbone_lr 1e-5

run_job isic17_nosched \
  "${CN[@]}" --dataset isic2017 --epochs 100 --eval_every 10 \
  --scheduler none --lr 1e-4 --weight_decay 1e-4 --guidance_scale 2.0 --backbone_lr 1e-5

run_job isic17_gs15 \
  "${CN[@]}" --dataset isic2017 --epochs 100 --eval_every 10 \
  --lr 1e-4 --weight_decay 1e-4 --guidance_scale 1.5 --backbone_lr 1e-5

run_job isic17_gs25 \
  "${CN[@]}" --dataset isic2017 --epochs 100 --eval_every 10 \
  --lr 1e-4 --weight_decay 1e-4 --guidance_scale 2.5 --backbone_lr 1e-5

run_job isic17_strong_aug \
  "${CN[@]}" --dataset isic2017 --epochs 100 --eval_every 10 \
  --lr 1e-4 --weight_decay 1e-4 --guidance_scale 2.0 --aug_level strong --backbone_lr 1e-5

run_job isic17_freeze_bb \
  "${CN[@]}" --dataset isic2017 --epochs 100 --eval_every 10 \
  --lr 1e-4 --weight_decay 1e-4 --guidance_scale 2.0 --freeze_backbone true

run_job isic17_dice2 \
  "${CN[@]}" --dataset isic2017 --epochs 100 --eval_every 10 \
  --lr 1e-4 --weight_decay 1e-4 --guidance_scale 2.0 --lambda_dice 2.0 --backbone_lr 1e-5

run_job isic17_dice05 \
  "${CN[@]}" --dataset isic2017 --epochs 100 --eval_every 10 \
  --lr 1e-4 --weight_decay 1e-4 --guidance_scale 2.0 --lambda_dice 0.5 --backbone_lr 1e-5

run_job isic17_npts100 \
  "${CN[@]}" --dataset isic2017 --epochs 100 --eval_every 10 \
  --lr 1e-4 --weight_decay 1e-4 --guidance_scale 2.0 --n_points 100 --backbone_lr 1e-5

run_job isic17_stem0 \
  "${CN[@]}" --dataset isic2017 --epochs 100 --eval_every 10 \
  --lr 1e-4 --weight_decay 1e-4 --guidance_scale 2.0 --stem_dim 0 --backbone_lr 1e-5

run_job isic17_convnext_small \
  "${CN[@]}" --dataset isic2017 --epochs 100 --eval_every 10 \
  --backbone convnext_small --lr 1e-4 --weight_decay 1e-4 --guidance_scale 2.0 --backbone_lr 1e-5

run_job isic17_pvt \
  --encoder pvt --batch_size 16 --num_workers 4 --scheduler cosine --warmup_epochs 5 \
  --dataset isic2017 --epochs 100 --eval_every 10 --aug_level light --ddim_steps 50 \
  --lr 1e-4 --weight_decay 1e-4 --guidance_scale 2.0 --backbone_lr 1e-5

run_job isic17_seed1 \
  "${CN[@]}" --dataset isic2017 --epochs 100 --eval_every 10 \
  --lr 1e-4 --weight_decay 1e-4 --guidance_scale 2.0 --backbone_lr 1e-5 --seed 1

run_job isic17_seed2 \
  "${CN[@]}" --dataset isic2017 --epochs 100 --eval_every 10 \
  --lr 1e-4 --weight_decay 1e-4 --guidance_scale 2.0 --backbone_lr 1e-5 --seed 2

# =============================================================================
# Phase B — longer ISIC2017 (mainline candidates)
# =============================================================================
run_job isic17_long600_base \
  "${CN[@]}" --dataset isic2017 --epochs 600 --eval_every 20 \
  --lr 1e-4 --weight_decay 1e-4 --guidance_scale 2.0 --backbone_lr 1e-5

run_job isic17_long600_strong \
  "${CN[@]}" --dataset isic2017 --epochs 600 --eval_every 20 \
  --lr 1e-4 --weight_decay 1e-4 --guidance_scale 2.0 --aug_level strong --backbone_lr 1e-5

run_job isic17_long400_small \
  "${CN[@]}" --dataset isic2017 --epochs 400 --eval_every 20 \
  --backbone convnext_small --lr 1e-4 --weight_decay 1e-4 --guidance_scale 2.0 --backbone_lr 1e-5

run_job isic17_long400_lr3e4 \
  "${CN[@]}" --dataset isic2017 --epochs 400 --eval_every 20 \
  --lr 3e-4 --weight_decay 1e-4 --guidance_scale 2.0 --backbone_lr 3e-5

# =============================================================================
# Phase C — HAM10000
# =============================================================================
run_job ham_e80_base \
  "${CN[@]}" --dataset ham10000 --epochs 80 --eval_every 5 \
  --lr 1e-4 --weight_decay 1e-4 --guidance_scale 2.0 --backbone_lr 1e-5

run_job ham_e100_strong \
  "${CN[@]}" --dataset ham10000 --epochs 100 --eval_every 5 \
  --lr 1e-4 --weight_decay 1e-4 --guidance_scale 2.0 --aug_level strong --backbone_lr 1e-5

run_job ham_e80_lr3e4 \
  "${CN[@]}" --dataset ham10000 --epochs 80 --eval_every 5 \
  --lr 3e-4 --weight_decay 1e-4 --guidance_scale 2.0 --backbone_lr 3e-5

run_job ham_e60_small \
  "${CN[@]}" --dataset ham10000 --epochs 60 --eval_every 5 \
  --backbone convnext_small --lr 1e-4 --weight_decay 1e-4 --guidance_scale 2.0 --backbone_lr 1e-5

run_job ham_e80_gs15 \
  "${CN[@]}" --dataset ham10000 --epochs 80 --eval_every 5 \
  --lr 1e-4 --weight_decay 1e-4 --guidance_scale 1.5 --backbone_lr 1e-5

# =============================================================================
# Phase D — transfer + ISIC2018
# =============================================================================
HAM_CKPT=""
for cand in ham_e100_strong ham_e80_base ham_e80_lr3e4 ham_e60_small ham_e80_gs15; do
  if [[ -f "src/runs/${cand}/best.pth" ]]; then
    HAM_CKPT="src/runs/${cand}/best.pth"
    break
  fi
done

if [[ -n "$HAM_CKPT" ]]; then
  log "Using HAM checkpoint for transfer: $HAM_CKPT"
  run_job isic17_ft_ham \
    "${CN[@]}" --dataset isic2017 --epochs 200 --eval_every 10 \
    --lr 5e-5 --weight_decay 1e-4 --guidance_scale 2.0 --backbone_lr 5e-6 \
    --init_checkpoint "$HAM_CKPT"

  run_job isic18_ft_ham \
    "${CN[@]}" --dataset isic2018 --epochs 200 --eval_every 10 \
    --lr 5e-5 --weight_decay 1e-4 --guidance_scale 2.0 --backbone_lr 5e-6 \
    --init_checkpoint "$HAM_CKPT"

  run_job ph2_ft_ham \
    --dataset ph2 --encoder convnext --backbone convnext_tiny \
    --epochs 400 --batch_size 8 --eval_every 20 \
    --lr 5e-5 --weight_decay 1e-4 --guidance_scale 1.5 \
    --scheduler cosine --warmup_epochs 10 --aug_level light \
    --backbone_lr 5e-6 --num_workers 4 --init_checkpoint "$HAM_CKPT"
else
  log "WARN: no HAM ckpt for transfer"
fi

run_job isic18_scratch400 \
  "${CN[@]}" --dataset isic2018 --epochs 400 --eval_every 20 \
  --lr 1e-4 --weight_decay 1e-4 --guidance_scale 2.0 --backbone_lr 1e-5

run_job isic18_scratch400_strong \
  "${CN[@]}" --dataset isic2018 --epochs 400 --eval_every 20 \
  --lr 1e-4 --weight_decay 1e-4 --guidance_scale 2.0 --aug_level strong --backbone_lr 1e-5

# =============================================================================
# Phase E — PH2 long confirmation (+ seeds)
# =============================================================================
run_job ph2_cosine_e800 \
  --dataset ph2 --encoder convnext --backbone convnext_tiny \
  --epochs 800 --batch_size 8 --eval_every 20 \
  --lr 1e-4 --weight_decay 1e-4 --guidance_scale 1.5 \
  --scheduler cosine --warmup_epochs 20 --aug_level light \
  --backbone_lr 1e-5 --num_workers 4

run_job ph2_cosine_e800_seed1 \
  --dataset ph2 --encoder convnext --backbone convnext_tiny \
  --epochs 800 --batch_size 8 --eval_every 20 \
  --lr 1e-4 --weight_decay 1e-4 --guidance_scale 1.5 \
  --scheduler cosine --warmup_epochs 20 --aug_level light \
  --backbone_lr 1e-5 --num_workers 4 --seed 1

run_job ph2_cosine_e800_seed2 \
  --dataset ph2 --encoder convnext --backbone convnext_tiny \
  --epochs 800 --batch_size 8 --eval_every 20 \
  --lr 1e-4 --weight_decay 1e-4 --guidance_scale 1.5 \
  --scheduler cosine --warmup_epochs 20 --aug_level light \
  --backbone_lr 1e-5 --num_workers 4 --seed 2

run_job ph2_strong_e800 \
  --dataset ph2 --encoder convnext --backbone convnext_tiny \
  --epochs 800 --batch_size 8 --eval_every 20 \
  --lr 1e-4 --weight_decay 1e-4 --guidance_scale 1.5 \
  --scheduler cosine --warmup_epochs 20 --aug_level strong \
  --backbone_lr 1e-5 --num_workers 4

# =============================================================================
# Phase F — extra ISIC2017 seeds / BS=8 (slower, more updates) for robustness
# =============================================================================
run_job isic17_bs8_e300 \
  "${CN[@]}" --dataset isic2017 --epochs 300 --eval_every 20 --batch_size 8 \
  --lr 1e-4 --weight_decay 1e-4 --guidance_scale 2.0 --backbone_lr 1e-5

run_job isic17_long600_seed1 \
  "${CN[@]}" --dataset isic2017 --epochs 600 --eval_every 20 \
  --lr 1e-4 --weight_decay 1e-4 --guidance_scale 2.0 --backbone_lr 1e-5 --seed 1

run_job isic17_long600_seed2 \
  "${CN[@]}" --dataset isic2017 --epochs 600 --eval_every 20 \
  --lr 1e-4 --weight_decay 1e-4 --guidance_scale 2.0 --backbone_lr 1e-5 --seed 2

# =============================================================================
# Aggregate
# =============================================================================
log "======= Writing RESULTS.md ======="
python - "$BATCH_DIR" "$SUMMARY_MD" <<'PY'
import json, os, sys, glob
batch_dir, out_md = sys.argv[1:3]
rows = []
for path in sorted(glob.glob("src/runs/*/summary.json")):
    try:
        s = json.load(open(path))
    except Exception:
        continue
    name = os.path.basename(os.path.dirname(path))
    rows.append({
        "name": name,
        "dataset": s.get("dataset"),
        "best_val_dice": s.get("best_val_dice"),
        "test_dice": s.get("test_dice"),
        "test_iou": s.get("test_iou"),
        "lr": s.get("lr"),
        "scheduler": s.get("scheduler"),
        "backbone": s.get("backbone"),
        "guidance_scale": s.get("guidance_scale"),
        "aug_level": s.get("aug_level"),
        "freeze_backbone": s.get("freeze_backbone"),
        "epochs": s.get("epochs"),
    })

rows.sort(key=lambda r: (r["best_val_dice"] is None, -(r["best_val_dice"] or -1)))
lines = [
    "# P2SDiff batch results (afshin)",
    "",
    "| run | dataset | best val Dice | test Dice | test IoU | lr | sched | backbone | gs | aug | freeze | epochs |",
    "|---|---|---:|---:|---:|---|---|---|---:|---|---|---:|",
]
def fmt(x, nd=4):
    if x is None: return "-"
    if isinstance(x, float): return f"{x:.{nd}f}"
    return str(x)

for r in rows:
    lines.append(
        f"| {r['name']} | {r['dataset']} | {fmt(r['best_val_dice'])} | {fmt(r['test_dice'])} | "
        f"{fmt(r['test_iou'])} | {fmt(r['lr'],0)} | {r['scheduler']} | {r['backbone']} | "
        f"{fmt(r['guidance_scale'],2)} | {r['aug_level']} | {r['freeze_backbone']} | {r['epochs']} |"
    )

# per-dataset winners
lines += ["", "## Best per dataset", ""]
for ds in ["ph2", "isic2017", "isic2018", "ham10000"]:
    cand = [r for r in rows if r["dataset"] == ds and r["best_val_dice"] is not None]
    if not cand:
        lines.append(f"- **{ds}**: no completed runs")
        continue
    b = max(cand, key=lambda r: r["best_val_dice"])
    lines.append(
        f"- **{ds}**: `{b['name']}` val={fmt(b['best_val_dice'])} test={fmt(b['test_dice'])}"
    )

open(out_md, "w").write("\n".join(lines) + "\n")
print(f"Wrote {out_md} ({len(rows)} runs)")
print("Top 15 by val Dice:")
for r in rows[:15]:
    print(f"  {fmt(r['best_val_dice'])}  {r['name']} ({r['dataset']}) test={fmt(r['test_dice'])}")
PY

log "======= BATCH COMPLETE ======="
log "See $SUMMARY_MD and $STATUS_JSON"
