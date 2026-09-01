#!/usr/bin/env python
"""Summarize the last part of a V5.2 dp2seg_stats.jsonl log."""
import argparse, json
from pathlib import Path

KEYS = [
    "proposal_dice_loss", "exact_loss_offset", "exact_loss_conf",
    "exact_target_conf_rate", "exact_teacher_conf_rate",
    "exact_conf_mean", "exact_gate_mean", "exact_applied_abs",
    "residual_abs", "t_batch",
]

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('path')
    ap.add_argument('--last', type=int, default=100)
    a=ap.parse_args()
    rows=[]
    with open(a.path) as f:
        for line in f:
            if line.strip(): rows.append(json.loads(line))
    if not rows:
        raise SystemExit('no rows')
    tail=rows[-a.last:]
    print(f'rows={len(rows)}  averaging last {len(tail)}')
    for k in KEYS:
        vals=[float(r[k]) for r in tail if k in r]
        if vals:
            print(f'{k:28s} {sum(vals)/len(vals):.6f}')
    print('last epoch/step:', rows[-1].get('epoch'), rows[-1].get('step'))

if __name__ == '__main__': main()
