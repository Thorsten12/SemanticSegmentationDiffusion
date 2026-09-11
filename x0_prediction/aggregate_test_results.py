"""Recover missing test_metrics.json for every run, then aggregate all runs
(across seeds) into a mean +/- std summary table.

WHY this exists: train.py's final test-eval block is wrapped in a bare
`try/except Exception: traceback.print_exc()` (see train.py, right after
`_plot_scale_gates(...)`). If `load_checkpoint(best_path, cfg_eval, device)`
raises for ANY reason at that point -- e.g. a newer config field like
`attn_sampler_heads` was saved with a value that doesn't evenly divide
`cond_channels` for some `sampler_type=attention` run, or any other
mismatch -- the exception is only printed to that run's log file, and the
training process still exits 0 with best.pth/last.pth/history.json fully
written. Nothing on the outside (this bash ablation harness included)
notices anything went wrong, so you end up with a directory tree where some
runs have test_metrics.json and others silently don't -- exactly the
situation in your `ls` output.

This script does two independent jobs:

  1. `recover`: for every `<out_dir>/*/best.pth` that has NO sibling
     metrics file (name depends on `--sampler`, see below), re-run the
     test-split evaluation standalone (reusing sample.py's own
     `load_checkpoint`/`evaluate`, not train.py's inline copy) and write
     the metrics file next to it. Skips any run that already has one
     (fast on repeated calls, safe to re-run). Prints a clear error (not
     just a silent skip) for any run that still fails, so failures are
     visible instead of just absent.

  2. `aggregate`: after (or independent of) recovery, scan every
     `<out_dir>/<run_name>_seed_NN/<metrics_filename>`, group by run_name
     (stripping the trailing `_seed_NN`), and print a table of
     `mean +/- std` Dice/IoU per run, sorted by mean Dice descending --
     same shape as the tables you've been building by hand from the V7
     ablation logs.

--- NEW: --sampler {ddim,ddpm} -------------------------------------------
Controls which diffusion-sampling mode is used during the `recover` step,
mirroring sample.py's own CLI `--sampler` flag (same convenience shortcut:
`ddpm` forces `cfg.eta = 1.0` and `cfg.ddim_steps = cfg.timesteps`, i.e.
full ancestral sampling over every original diffusion step; `ddim`, the
default, leaves `cfg.eta`/`cfg.ddim_steps` exactly as saved in each
checkpoint's own config -- unchanged from this script's previous
behavior).

Since a run's DDIM and DDPM test metrics are NOT comparable and shouldn't
overwrite each other, the metrics filename is now sampler-dependent:
  --sampler ddim (default) -> test_metrics.json          (unchanged name)
  --sampler ddpm           -> test_metrics_ddpm.json
This means you can run `--recover_only` once with each `--sampler` value
against the SAME --runs_dir, and both sets of per-run metrics files will
coexist on disk. `aggregate` (and the `aggregated_test_results*.json` it
writes) is likewise sampler-scoped: it only ever reads/writes the metrics
filename matching the `--sampler` value it was called with, so aggregating
one sampler never touches or overwrites the other's files.

Usage:
    # Both steps, DDIM (default, same behavior as before):
    python -m x0_prediction_V6.aggregate_test_results \\
        --runs_dir x0_prediction_V6/runs/ablation_v8_sampler_comparison

    # Both steps, full DDPM ancestral sampling:
    python -m x0_prediction_V6.aggregate_test_results \\
        --runs_dir x0_prediction_V6/runs/ablation_v8_sampler_comparison --sampler ddpm

    # Only recover missing metrics for a given sampler (e.g. before all
    # seeds of a run are done -- this only touches runs whose best.pth
    # already exists):
    python -m x0_prediction_V6.aggregate_test_results \\
        --runs_dir x0_prediction_V6/runs/ablation_v8_sampler_comparison --sampler ddpm --recover_only

    # Only aggregate (skip recovery -- e.g. you already ran --recover_only
    # and just want to reprint the table) for a given sampler:
    python -m x0_prediction_V6.aggregate_test_results \\
        --runs_dir x0_prediction_V6/runs/ablation_v8_sampler_comparison --sampler ddpm --aggregate_only
"""

import argparse
import json
import os
import re
import traceback
from collections import defaultdict

import torch
from torch.utils.data import DataLoader

from .config import Config
from .data import build_contour_dataset
from .diffusion import GaussianDiffusion
from .sample import evaluate, load_checkpoint


# Matches "<anything>_seed_<digits>" -> group key is "<anything>".
_SEED_SUFFIX_RE = re.compile(r"^(.*)_seed_\d+$")


def _run_group(dir_name: str) -> str:
    """'baseline_seed_03' -> 'baseline'. Falls back to the full name if it
    doesn't match the expected '<run>_seed_NN' pattern (so unexpected
    directory names still show up in the table instead of being dropped
    silently)."""
    m = _SEED_SUFFIX_RE.match(dir_name)
    return m.group(1) if m else dir_name


def _metrics_filename(sampler: str) -> str:
    """Sampler-scoped metrics filename, so ddim/ddpm results never collide.
    Kept as its own function since both `recover_missing_test_metrics` and
    `aggregate` need to agree on the exact same name for a given sampler."""
    return "test_metrics.json" if sampler == "ddim" else "test_metrics_ddpm.json"


def recover_missing_test_metrics(runs_dir: str, device_str: str = "cuda",
                                  sampler: str = "ddim", batch_size: int = None):
    """For every '<runs_dir>/*/best.pth' without a sibling metrics file
    (see `_metrics_filename`), re-run the test-split evaluation using the
    requested diffusion `sampler` and write the metrics file.

    Uses sample.py's own load_checkpoint/evaluate (the same code path used
    for --split test at the CLI), NOT train.py's inline copy -- this avoids
    train.py's specific bug of merging the checkpoint's saved config onto a
    single mutable `Config()` instance across (potentially) several fields
    that were never meant to be read together.

    `sampler`: "ddim" (default) leaves cfg.eta/cfg.ddim_steps exactly as
    saved in each checkpoint's own config (unchanged from before this
    option existed). "ddpm" overrides them to cfg.eta=1.0 and
    cfg.ddim_steps=cfg.timesteps AFTER load_checkpoint has populated cfg
    from the checkpoint (same override-after-load ordering as sample.py's
    CLI --sampler ddpm, so it always wins over whatever the checkpoint's
    own saved eta/ddim_steps were).

    `batch_size`: if given, overrides cfg.batch_size (after load_checkpoint,
    same override-after-load ordering as the sampler override above) for
    the DataLoader used here. Useful for DDPM, which runs the full
    `cfg.timesteps` steps per batch instead of `ddim_steps` -- a smaller
    batch_size trades more (smaller, faster) forward passes for less peak
    memory, but does NOT reduce total sampling steps per run. None
    (default) leaves cfg.batch_size exactly as saved in the checkpoint,
    same as before this option existed.
    """
    device = torch.device(device_str if torch.cuda.is_available() else "cpu")
    entries = sorted(os.listdir(runs_dir))
    metrics_filename = _metrics_filename(sampler)

    recovered, skipped_existing, failed = [], [], []

    for name in entries:
        run_dir = os.path.join(runs_dir, name)
        best_path = os.path.join(run_dir, "best.pth")
        metrics_path = os.path.join(run_dir, metrics_filename)

        if not os.path.isfile(best_path):
            continue  # not a run directory (or run never produced a checkpoint)
        if os.path.isfile(metrics_path):
            skipped_existing.append(name)
            continue

        print(f"[recover:{sampler}] {name}: no {metrics_filename}, re-evaluating test split...")
        try:
            # Fresh Config PER RUN (not reused/mutated across runs) --
            # load_checkpoint fills in every architecture-relevant field
            # from the checkpoint's own saved config, so this starts from
            # a clean slate every time, unlike train.py's reused cfg_eval.
            # Requires sample.py's load_checkpoint to know about
            # `sampler_type` (input/deformable/attention) -- see the
            # updated sample.py that added _resolve_sampler_type() and
            # threads sampler_type/attn_sampler_window/attn_sampler_heads
            # through to ContourDenoiser. Without that, any run trained
            # with sampler_type="attention" fails here with a state_dict
            # key mismatch (query_heads/key_heads/value_heads not
            # recognized), exactly like the traceback that prompted this.
            cfg = Config()
            ds = build_contour_dataset(cfg.skin_root, cfg.dataset, "te", cfg.n_points,
                                       cfg.img_size, augment=False, npy_size=cfg.npy_size)
            # dataset/img_size/etc. might differ per run if the ablation
            # ever varies them -- load_checkpoint below overwrites cfg's
            # architecture fields from the checkpoint anyway, but the
            # DATASET ITSELF must already match what cfg.dataset says here.
            # If your ablation always uses the same --dataset (as in the
            # V8 script), this is fine as-is; otherwise see the note in
            # the module docstring about reading `dataset` from the
            # checkpoint's saved config BEFORE building the dataset.
            encoder, denoiser, snapper, proposal_head = load_checkpoint(best_path, cfg, device)

            # NOTE: this override happens AFTER load_checkpoint, which is
            # exactly what makes it override the checkpoint's own saved
            # eta/ddim_steps rather than being clobbered by them -- same
            # ordering as sample.py's CLI `--sampler ddpm` handling.
            if sampler == "ddpm":
                cfg.eta = 1.0
                cfg.ddim_steps = cfg.timesteps
            if batch_size is not None:
                cfg.batch_size = batch_size

            diffusion = GaussianDiffusion(cfg.timesteps, cfg.beta_start, cfg.beta_end, device=device)
            loader = DataLoader(ds, batch_size=cfg.batch_size, shuffle=False,
                                num_workers=cfg.num_workers)

            snap_t_threshold = int(round(cfg.snap_t_threshold_frac * cfg.timesteps))
            dice, iou = evaluate(
                encoder, denoiser, diffusion, loader, cfg, device,
                snapper=snapper, proposal_head=proposal_head,
                viz_path=os.path.join(run_dir, f"test_grid_recovered_{sampler}.png"),
                snap_mode=cfg.snap_mode, snap_t_threshold=snap_t_threshold,
                snap_every=cfg.snap_every,
            )
            metrics = {"test_dice": dice, "test_iou": iou, "recovered": True,
                      "sampler": sampler,  "ddim_steps": cfg.ddim_steps}
            with open(metrics_path, "w") as f:
                json.dump(metrics, f, indent=2)
            print(f"[recover:{sampler}] {name}: OK -> Dice {dice:.4f} | IoU {iou:.4f}")
            recovered.append(name)
        except Exception:
            print(f"[recover:{sampler}] {name}: FAILED")
            traceback.print_exc()
            failed.append(name)

    print(f"\n[recover:{sampler}] done: {len(recovered)} recovered, "
          f"{len(skipped_existing)} already had {metrics_filename}, "
          f"{len(failed)} failed.")
    if failed:
        print(f"[recover:{sampler}] failed runs (see traceback above for each): {failed}")
    return recovered, skipped_existing, failed


def aggregate(runs_dir: str, sampler: str = "ddim"):
    """Group every '<runs_dir>/<run>_seed_NN/<metrics_filename>' by run name
    and print mean +/- std Dice/IoU, sorted by mean Dice descending.
    `metrics_filename` depends on `sampler` -- see `_metrics_filename` --
    so this only ever reads the metrics produced by that same sampler."""
    metrics_filename = _metrics_filename(sampler)
    groups = defaultdict(list)  # run_name -> list of (dice, iou) tuples

    for name in sorted(os.listdir(runs_dir)):
        run_dir = os.path.join(runs_dir, name)
        metrics_path = os.path.join(run_dir, metrics_filename)
        if not os.path.isfile(metrics_path):
            continue
        with open(metrics_path) as f:
            m = json.load(f)
        if "test_dice" not in m or "test_iou" not in m:
            print(f"[aggregate:{sampler}] warn: {metrics_path} missing test_dice/test_iou, skipping")
            continue
        group = _run_group(name)
        groups[group].append((m["test_dice"], m["test_iou"]))

    if not groups:
        print(f"[aggregate:{sampler}] no {metrics_filename} files found -- nothing to aggregate. "
              "Run without --aggregate_only (or run recovery first, with the same "
              "--sampler) to generate them.")
        return

    import statistics as stats

    rows = []
    for run_name, vals in groups.items():
        dices = [d * 100.0 for d, _ in vals]
        ious = [i * 100.0 for _, i in vals]
        n = len(vals)
        dice_mean = stats.mean(dices)
        dice_std = stats.stdev(dices) if n > 1 else 0.0
        iou_mean = stats.mean(ious)
        iou_std = stats.stdev(ious) if n > 1 else 0.0
        rows.append((run_name, n, dice_mean, dice_std, iou_mean, iou_std))

    rows.sort(key=lambda r: -r[2])  # sort by mean Dice, descending

    name_w = max(len(r[0]) for r in rows) + 2
    print(f"\n[{sampler}] {'run':<{name_w}} | {'n':>5} | {'Dice (%)':>15} | {'IoU (%)':>15}")
    print("-" * (name_w + 5 + 20 + 20 + len(sampler) + 3))
    for run_name, n, dm, ds, im, istd in rows:
        flag = "  <-- incomplete (expected 5 seeds)" if n != 5 else ""
        print(f"{run_name:<{name_w}} | {n:>5} | {dm:6.2f} ± {ds:<5.2f}   | {im:6.2f} ± {istd:<5.2f}{flag}")

    # Also dump machine-readable summary next to the runs, for later reuse
    # (e.g. plotting, or diffing against a previous ablation's summary).
    # Sampler-scoped filename, same reasoning as the per-run metrics files.
    summary_name = "aggregated_test_results.json" if sampler == "ddim" \
        else "aggregated_test_results_ddpm.json"
    out_path = os.path.join(runs_dir, summary_name)
    with open(out_path, "w") as f:
        json.dump(
            [{"run": r[0], "n_seeds": r[1],
              "dice_mean": r[2], "dice_std": r[3],
              "iou_mean": r[4], "iou_std": r[5]} for r in rows],
            f, indent=2,
        )
    print(f"\n[aggregate:{sampler}] wrote {out_path}")


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                      formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--runs_dir", type=str, required=True,
                        help="Directory containing '<run_name>_seed_NN' subdirectories, "
                             "e.g. x0_prediction_V6/runs/ablation_v8_sampler_comparison")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--sampler", choices=["ddim", "ddpm"], default="ddim",
                        help="Diffusion sampling mode used during the recover step (and "
                             "which metrics filename aggregate reads/writes). 'ddim' "
                             "(default): leaves each checkpoint's own saved eta/ddim_steps "
                             "untouched, writes/reads 'test_metrics.json' (unchanged from "
                             "before this flag existed). 'ddpm': forces eta=1.0 and "
                             "ddim_steps=timesteps (full ancestral sampling), writes/reads "
                             "'test_metrics_ddpm.json' instead, so it never collides with "
                             "or overwrites DDIM results for the same runs_dir.")
    parser.add_argument("--batch_size", type=int, default=None,
                        help="Override the DataLoader batch size during recovery (default: "
                             "use each checkpoint's own saved cfg.batch_size, unchanged from "
                             "before this option existed). Mainly useful with --sampler ddpm: "
                             "DDPM runs cfg.timesteps steps per batch instead of ddim_steps, "
                             "so a smaller batch_size here trades more (smaller) forward "
                             "passes for lower peak memory -- it does NOT reduce the number "
                             "of sampling steps per run, so it speeds things up only if the "
                             "current batch_size is memory-bound (e.g. causing swapping or "
                             "forcing smaller effective throughput), not otherwise.")
    parser.add_argument("--recover_only", action="store_true",
                        help="Only re-evaluate missing metrics files; skip aggregation.")
    parser.add_argument("--aggregate_only", action="store_true",
                        help="Only print the aggregated table; skip re-evaluation "
                             "(use if you already ran --recover_only, or just want "
                             "to reprint the table from existing metrics files).")
    args = parser.parse_args()

    if args.recover_only and args.aggregate_only:
        raise SystemExit("--recover_only and --aggregate_only are mutually exclusive")

    if not args.aggregate_only:
        recover_missing_test_metrics(args.runs_dir, device_str=args.device, sampler=args.sampler,
                                     batch_size=args.batch_size)

    if not args.recover_only:
        aggregate(args.runs_dir, sampler=args.sampler)


if __name__ == "__main__":
    main()