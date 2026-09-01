#!/usr/bin/env bash
# ~12h focused v2 batch from previous sweep winners.
# Winners: n_points=100, guidance=1.5, lambda_dice=0.5, lr=1e-4, cosine, stem on.
# Also: fix HAM empty-contour, try soft_dice_size=96, HAM + transfer.
#
#   export SCREENDIR=$HOME/.screen
#   screen -dmS p2sdiff_v2 bash -lc 'source $HOME/venvs/pytorch/bin/activate; cd /hdd/deeplearning/p2sdiff; bash scripts/run_batch_12h_v2.sh'

set -u
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

VENV="${VENV:-$HOME/venvs/pytorch/bin/activate}"
# shellcheck disable=SC1090
source "$VENV"

export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=4
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"

BATCH_DIR="src/runs/batch_12h_v2"
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
    try: data = json.load(open(path))
    except Exception: data = {}
entry = {"status": status, "updated": time.strftime("%Y-%m-%d %H:%M:%S")}
if extra: entry["detail"] = extra
data[name] = entry
json.dump(data, open(path, "w"), indent=2)
PY
}

run_job() {
  local name="$1"; shift
  local out_dir="src/runs/${name}"
  local log_file="$BATCH_DIR/${name}.log"
  if [[ -f "$out_dir/summary.json" ]]; then
    log "SKIP $name"; record_status "$name" "skipped_exists"; return 0
  fi
  mkdir -p "$out_dir"
  log "START $name"; record_status "$name" "running"
  local rc=0
  python -m src.train --out_dir "$out_dir" "$@" >"$log_file" 2>&1
  rc=$?
  if [[ $rc -eq 0 ]]; then
    log "DONE  $name (ok)"; record_status "$name" "ok"
  else
    log "FAIL  $name (exit $rc) — $log_file"; record_status "$name" "failed" "exit=$rc"
  fi
  return 0
}

log "======= V2 BATCH START ======="
python - <<'PY' | tee -a "$MASTER_LOG"
import torch
print("torch", torch.__version__, "cuda", torch.cuda.is_available())
if torch.cuda.is_available():
    print("device", torch.cuda.get_device_name(0))
PY

# Shared "v2 recipe" from sweep winners
V2=(
  --encoder convnext --backbone convnext_tiny
  --batch_size 16 --num_workers 4
  --scheduler cosine --warmup_epochs 5
  --n_points 100 --guidance_scale 1.5 --lambda_dice 0.5
  --lr 1e-4 --weight_decay 1e-4 --backbone_lr 1e-5
  --aug_level light --ddim_steps 50
)

# ---------------------------------------------------------------------------
# A. Validate v2 recipe (short) then long ISIC2017
# ---------------------------------------------------------------------------
run_job v2_isic17_n100_gs15_d05_e120 \
  "${V2[@]}" --dataset isic2017 --epochs 120 --eval_every 10

run_job v2_isic17_n100_gs15_d05_e600 \
  "${V2[@]}" --dataset isic2017 --epochs 600 --eval_every 20

# Soft-dice higher-res boundary gradient
run_job v2_isic17_softdice96_e400 \
  "${V2[@]}" --dataset isic2017 --epochs 400 --eval_every 20 --soft_dice_size 96

# n_points=150 middle ground
run_job v2_isic17_n150_gs15_d05_e400 \
  "${V2[@]}" --dataset isic2017 --epochs 400 --eval_every 20 --n_points 150

# strong aug + v2
run_job v2_isic17_strong_e400 \
  "${V2[@]}" --dataset isic2017 --epochs 400 --eval_every 20 --aug_level strong

# ---------------------------------------------------------------------------
# B. ISIC2018 with v2
# ---------------------------------------------------------------------------
run_job v2_isic18_n100_gs15_d05_e400 \
  "${V2[@]}" --dataset isic2018 --epochs 400 --eval_every 20

run_job v2_isic18_strong_e400 \
  "${V2[@]}" --dataset isic2018 --epochs 400 --eval_every 20 --aug_level strong

# ---------------------------------------------------------------------------
# C. HAM (empty-contour fix) + transfer
# ---------------------------------------------------------------------------
run_job v2_ham_n100_gs15_d05_e50 \
  "${V2[@]}" --dataset ham10000 --epochs 50 --eval_every 5

run_job v2_ham_strong_e60 \
  "${V2[@]}" --dataset ham10000 --epochs 60 --eval_every 5 --aug_level strong

HAM_CKPT=""
for cand in v2_ham_strong_e60 v2_ham_n100_gs15_d05_e50; do
  if [[ -f "src/runs/${cand}/best.pth" ]]; then HAM_CKPT="src/runs/${cand}/best.pth"; break; fi
done

if [[ -n "$HAM_CKPT" ]]; then
  log "Transfer from $HAM_CKPT"
  run_job v2_isic17_ft_ham_e150 \
    "${V2[@]}" --dataset isic2017 --epochs 150 --eval_every 10 \
    --lr 5e-5 --backbone_lr 5e-6 --init_checkpoint "$HAM_CKPT"
  run_job v2_isic18_ft_ham_e150 \
    "${V2[@]}" --dataset isic2018 --epochs 150 --eval_every 10 \
    --lr 5e-5 --backbone_lr 5e-6 --init_checkpoint "$HAM_CKPT"
  run_job v2_ph2_ft_ham_e400 \
    "${V2[@]}" --dataset ph2 --epochs 400 --eval_every 20 --batch_size 8 \
    --lr 5e-5 --backbone_lr 5e-6 --init_checkpoint "$HAM_CKPT"
else
  log "WARN: no HAM ckpt"
fi

# ---------------------------------------------------------------------------
# D. PH2 with v2 recipe
# ---------------------------------------------------------------------------
run_job v2_ph2_n100_strong_e800 \
  "${V2[@]}" --dataset ph2 --epochs 800 --eval_every 20 --batch_size 8 \
  --aug_level strong --warmup_epochs 20

run_job v2_ph2_n100_gs15_d05_e800 \
  "${V2[@]}" --dataset ph2 --epochs 800 --eval_every 20 --batch_size 8 \
  --warmup_epochs 20

# ---------------------------------------------------------------------------
# Aggregate
# ---------------------------------------------------------------------------
log "======= Writing RESULTS.md ======="
python - "$SUMMARY_MD" <<'PY'
import json, os, glob, sys
out_md = sys.argv[1]
rows=[]
for path in sorted(glob.glob("src/runs/*/summary.json")):
    try: s=json.load(open(path))
    except Exception: continue
    name=os.path.basename(os.path.dirname(path))
    if not name.startswith("v2_") and "batch" not in name:
        # include v2 + still show prior bests later
        pass
    rows.append({**s, "name": name})
rows.sort(key=lambda r: (r.get("best_val_dice") is None, -(r.get("best_val_dice") or -1)))
lines=["# P2SDiff v2 / 12h batch results","","| run | dataset | val | test | lr | gs | notes |","|---|---|---:|---:|---|---:|---|"]
def fmt(x,nd=4):
    if x is None: return "-"
    if isinstance(x,float): return f"{x:.{nd}f}"
    return str(x)
v2=[r for r in rows if str(r["name"]).startswith("v2_")]
for r in v2:
    lines.append(f"| {r['name']} | {r.get('dataset')} | {fmt(r.get('best_val_dice'))} | {fmt(r.get('test_dice'))} | {fmt(r.get('lr'),0)} | {fmt(r.get('guidance_scale'),2)} | n={r.get('epochs')} |")
lines += ["", "## Best per dataset (v2 only)", ""]
for ds in ["ph2","isic2017","isic2018","ham10000"]:
    cand=[r for r in v2 if r.get("dataset")==ds and r.get("best_val_dice") is not None]
    if not cand:
        lines.append(f"- **{ds}**: none"); continue
    b=max(cand, key=lambda r: r["best_val_dice"])
    lines.append(f"- **{ds}**: `{b['name']}` val={fmt(b['best_val_dice'])} test={fmt(b.get('test_dice'))}")
# compare to previous batch winners
lines += ["", "## vs previous batch winners", ""]
prev = {
    "ph2": ("ph2_strong_e800", 0.9214, 0.8981),
    "isic2017": ("isic17_long600_seed1", 0.8745, 0.8773),
    "isic2018": ("isic18_scratch400_strong", 0.8694, 0.8704),
}
for ds,(n,v,t) in prev.items():
    lines.append(f"- prev **{ds}**: `{n}` val={v:.4f} test={t:.4f}")
open(out_md,"w").write("\n".join(lines)+"\n")
print("wrote", out_md, "v2 runs", len(v2))
for r in v2[:20]:
    print(f"  {fmt(r.get('best_val_dice'))} test={fmt(r.get('test_dice'))} {r['name']}")
PY
log "======= V2 BATCH COMPLETE ======="
