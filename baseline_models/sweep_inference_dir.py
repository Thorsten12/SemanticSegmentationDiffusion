"""Sweep guidance_scale / ddim_steps over MULTIPLE trained P2SDiff checkpoints.

Instead of pointing at a single --ckpt like the old sweep_inference.py, you
point this at a directory. It recursively searches for every "best.pth" it
can find underneath that directory, treats each one as a separate trained
model, and re-runs the exact same val sweep + top-k test confirmation logic
per checkpoint. Results are pooled across ALL checkpoints, so "top-5" means
the best 5 (checkpoint, guidance_scale, ddim_steps) combinations overall --
not top-5 per checkpoint.

Why pool instead of per-checkpoint top-5:
    If you trained e.g. 6 model variants, you almost always want to know
    "which model+setting combo is actually best", not "the best setting for
    each model separately" (which just gives you 6 winners with no ranking
    between them). Use --top_k_per_ckpt if you want the latter instead (see
    below).

Robustness note: checkpoints can be old / from a slightly different version
of the codebase (config fields renamed, extra keys, etc.). A single bad or
incompatible checkpoint should NOT kill the whole sweep. Every checkpoint
is therefore loaded and evaluated inside its own try/except; failures are
logged with the checkpoint path + exception and the run continues with the
next checkpoint. A summary of skipped checkpoints is printed at the end.

Usage (run from the repository root, i.e. one level above `src/`):

    python -m src.sweep_inference_dir \
        --ckpt_dir src/runs/baseline \
        --dataset ph2 \
        --guidance_scales 1.0 1.5 2.0 2.5 3.0 4.0 \
        --ddim_steps_list 25 50 100

Or as a standalone script next to train.py/sample.py:

    python sweep_inference_dir.py --ckpt_dir runs/baseline --dataset ph2

Notes:
- --dataset (and --skin_root) still act as an override; if omitted, each
  checkpoint's own saved config decides its dataset, so you can point this
  at a directory containing checkpoints for different datasets and it will
  do the right thing for each one.
- --top_k controls the pooled top-k that get test-confirmed (default: 5).
- --top_k_per_ckpt additionally confirms the best combo(s) per checkpoint
  on test, even if they didn't make the pooled top-k (0 = off, default).
- --skip_test_confirm skips the test pass entirely (val sweep only).
- --ckpt_name lets you search for a different filename than "best.pth"
  (e.g. "last.pth"), in case that's ever needed.
"""

import argparse
import itertools
import json
import os
import traceback

import torch
from torch.utils.data import DataLoader

from .config import Config
from .data import build_contour_dataset
from .diffusion import GaussianDiffusion
from .sample import evaluate, load_checkpoint


def parse_args():
    p = argparse.ArgumentParser(description="Sweep guidance_scale / ddim_steps over all "
                                             "checkpoints found in a directory")
    p.add_argument("--ckpt_dir", type=str, required=True,
                   help="directory to search recursively for checkpoint files")
    p.add_argument("--ckpt_name", type=str, default="best.pth",
                   help="checkpoint filename to search for (default: best.pth)")
    p.add_argument("--dataset", choices=["ph2", "isic2017", "isic2018", "ham10000"], default=None,
                   help="override dataset for ALL checkpoints; default: use each checkpoint's "
                        "own saved config")
    p.add_argument("--skin_root", type=str, default=None)
    p.add_argument("--split", choices=["val", "test"], default="val",
                   help="sweep split (default: val -- keep test untouched until confirmation)")
    p.add_argument("--batch_size", type=int, default=None)
    p.add_argument("--device", type=str, default=None)

    p.add_argument("--guidance_scales", type=float, nargs="+",
                   default=[1.0, 1.5, 2.0, 2.5, 3.0, 4.0])
    p.add_argument("--ddim_steps_list", type=int, nargs="+",
                   default=[25, 50, 75, 100, 200])

    p.add_argument("--out_json", type=str, default="sweep_results_dir.json")
    p.add_argument("--viz_best", type=str, default=None,
                   help="if set, saves a prediction grid for the single overall-best "
                        "(checkpoint, gs, steps) combo to this path")

    p.add_argument("--top_k", type=int, default=5,
                   help="how many top (checkpoint, gs, steps) combos, pooled across ALL "
                        "checkpoints, to re-confirm on test (default: 5)")
    p.add_argument("--top_k_per_ckpt", type=int, default=0,
                   help="additionally confirm this many top combos PER checkpoint on test, "
                        "even if not in the pooled top-k (default: 0 = off)")
    p.add_argument("--skip_test_confirm", action="store_true",
                   help="skip the automatic test confirmation pass entirely "
                        "(only meaningful when --split val)")
    p.add_argument("--tta", action="store_true",
                   help="enable test-time augmentation (TTA) for the sweep; "
                        "if the checkpoint was trained with TTA, this should be set")
    return p.parse_args()


def find_checkpoints(ckpt_dir, ckpt_name):
    found = []
    for root, _dirs, files in os.walk(ckpt_dir):
        if ckpt_name in files:
            found.append(os.path.join(root, ckpt_name))
    found.sort()
    return found


def build_loader(cfg, split_key, device):
    ds = build_contour_dataset(cfg.skin_root, cfg.dataset, split_key, cfg.n_points,
                               cfg.img_size, augment=False, npy_size=cfg.npy_size)
    loader = DataLoader(ds, batch_size=cfg.batch_size, shuffle=False)
    return ds, loader


def sweep_one_checkpoint(ckpt_path, args, base_kwargs):
    """Loads one checkpoint and runs the full val sweep on it.

    Wrapped entirely in try/except by the caller -- this function is allowed
    to raise, the caller decides what to do with a failure.
    """
    cfg = Config()
    if args.dataset:
        cfg.dataset = args.dataset
    if args.skin_root:
        cfg.skin_root = args.skin_root
    if args.batch_size:
        cfg.batch_size = args.batch_size
    if args.device:
        cfg.device = args.device
    if args.tta:
        cfg.tta = args.tta

    device = torch.device(cfg.device if torch.cuda.is_available() else "cpu")

    print(f"\n=== Loading checkpoint: {ckpt_path} ===")
    encoder, denoiser = load_checkpoint(ckpt_path, cfg, device)
    print(f"  encoder={cfg.encoder} backbone={cfg.backbone} n_points={cfg.n_points} "
          f"dataset={cfg.dataset}")

    diffusion = GaussianDiffusion(cfg.timesteps, cfg.beta_start, cfg.beta_end, device=device)

    split_key = "vl" if args.split == "val" else "te"
    ds, loader = build_loader(cfg, split_key, device)
    print(f"  sweeping on split='{args.split}' ({len(ds)} images)")

    combos = list(itertools.product(args.guidance_scales, args.ddim_steps_list))
    ckpt_results = []
    for gs, steps in combos:
        cfg.guidance_scale = gs
        cfg.ddim_steps = steps
        dice, iou = evaluate(encoder, denoiser, diffusion, loader, cfg, device, viz_path=None, tta=args.tta)
        row = {"ckpt": ckpt_path, "dataset": cfg.dataset,
               "guidance_scale": gs, "ddim_steps": steps, "dice": dice, "iou": iou}
        ckpt_results.append(row)
        print(f"    gs={gs:<5} steps={steps:<4} -> Dice {dice:.4f} | IoU {iou:.4f}")

    return cfg, encoder, denoiser, diffusion, ds, ckpt_results


def main():
    args = parse_args()

    ckpts = find_checkpoints(args.ckpt_dir, args.ckpt_name)
    if not ckpts:
        print(f"No '{args.ckpt_name}' files found under {args.ckpt_dir!r}. Nothing to do.")
        return
    print(f"Found {len(ckpts)} checkpoint(s) under {args.ckpt_dir!r}:")
    for c in ckpts:
        print(f"  - {c}")

    combos_desc = f"guidance_scales={args.guidance_scales} x ddim_steps_list={args.ddim_steps_list}"
    print(f"\n{len(list(itertools.product(args.guidance_scales, args.ddim_steps_list)))} "
          f"combinations per checkpoint: {combos_desc}")

    all_results = []       # pooled val results across every checkpoint
    per_ckpt_results = {}  # ckpt_path -> list of its own val results (for top_k_per_ckpt)
    failed_ckpts = []      # (ckpt_path, error_str) for anything that raised

    # Cache the loaded model objects per checkpoint so the test-confirm pass
    # below doesn't have to reload from disk -- but keep it best-effort: if
    # something goes wrong we just reload on demand during confirmation.
    loaded_cache = {}

    for ckpt_path in ckpts:
        try:
            cfg, encoder, denoiser, diffusion, ds, ckpt_results = sweep_one_checkpoint(
                ckpt_path, args, base_kwargs={})
            all_results.extend(ckpt_results)
            per_ckpt_results[ckpt_path] = ckpt_results
            loaded_cache[ckpt_path] = (cfg, encoder, denoiser, diffusion)
        except Exception as e:
            # Robustness: codebase may have evolved since a checkpoint was saved
            # (renamed config fields, changed architecture defaults, missing
            # keys in the state dict, etc.). Skip this checkpoint, keep going.
            print(f"\n!!! Skipping checkpoint {ckpt_path!r} -- failed to load/evaluate.")
            print(f"    {type(e).__name__}: {e}")
            traceback.print_exc()
            failed_ckpts.append((ckpt_path, f"{type(e).__name__}: {e}"))
            continue

    if not all_results:
        print("\nAll checkpoints failed to load/evaluate -- nothing to report.")
        if failed_ckpts:
            print("Failures:")
            for c, err in failed_ckpts:
                print(f"  - {c}: {err}")
        return

    all_results.sort(key=lambda r: r["dice"], reverse=True)

    print(f"\n=== Pooled top {min(args.top_k, len(all_results))} across all checkpoints "
          f"(split={args.split}) ===")
    for r in all_results[:args.top_k]:
        print(f"  {os.path.relpath(r['ckpt'])} | gs={r['guidance_scale']:<5} "
              f"steps={r['ddim_steps']:<4} Dice {r['dice']:.4f} | IoU {r['iou']:.4f}")

    best = all_results[0]
    print(f"\nOverall best on {args.split}: {best['ckpt']} "
          f"guidance_scale={best['guidance_scale']} ddim_steps={best['ddim_steps']} "
          f"-> Dice {best['dice']:.4f}")

    # ------------------------------------------------------------------ #
    # Test confirmation pass: re-evaluate the pooled top-k (and, if       #
    # requested, top-k-per-checkpoint) combos on test, once each.        #
    # ------------------------------------------------------------------ #
    test_confirm = []
    if args.split == "val" and not args.skip_test_confirm:
        to_confirm = list(all_results[:args.top_k])

        if args.top_k_per_ckpt > 0:
            for ckpt_path, results_for_ckpt in per_ckpt_results.items():
                results_for_ckpt_sorted = sorted(results_for_ckpt, key=lambda r: r["dice"],
                                                  reverse=True)
                for r in results_for_ckpt_sorted[:args.top_k_per_ckpt]:
                    if r not in to_confirm:
                        to_confirm.append(r)

        print(f"\nConfirming {len(to_confirm)} combo(s) on TEST "
              f"(each sampling the full test split once)...")

        for r in to_confirm:
            ckpt_path = r["ckpt"]
            try:
                if ckpt_path in loaded_cache:
                    cfg, encoder, denoiser, diffusion = loaded_cache[ckpt_path]
                else:
                    # Shouldn't normally happen (only successfully loaded
                    # checkpoints end up in `to_confirm`), but reload
                    # defensively just in case.
                    cfg2 = Config()
                    if args.dataset:
                        cfg2.dataset = args.dataset
                    if args.skin_root:
                        cfg2.skin_root = args.skin_root
                    if args.batch_size:
                        cfg2.batch_size = args.batch_size
                    if args.device:
                        cfg2.device = args.device
                    device2 = torch.device(cfg2.device if torch.cuda.is_available() else "cpu")
                    encoder, denoiser = load_checkpoint(ckpt_path, cfg2, device2)
                    diffusion = GaussianDiffusion(cfg2.timesteps, cfg2.beta_start,
                                                  cfg2.beta_end, device=device2)
                    cfg = cfg2

                device = torch.device(cfg.device if torch.cuda.is_available() else "cpu")
                test_ds, test_loader = build_loader(cfg, "te", device)

                cfg.guidance_scale = r["guidance_scale"]
                cfg.ddim_steps = r["ddim_steps"]
                t_dice, t_iou = evaluate(encoder, denoiser, diffusion, test_loader, cfg, device,
                                         viz_path=None, tta=args.tta)
                row = {**r, "test_dice": t_dice, "test_iou": t_iou}
                test_confirm.append(row)
                print(f"  {os.path.relpath(ckpt_path)} | gs={r['guidance_scale']:<5} "
                      f"steps={r['ddim_steps']:<4} val Dice {r['dice']:.4f} -> "
                      f"test Dice {t_dice:.4f} | test IoU {t_iou:.4f}")
            except Exception as e:
                print(f"\n!!! Skipping test confirmation for {ckpt_path!r} "
                      f"(gs={r['guidance_scale']}, steps={r['ddim_steps']}) -- error below.")
                print(f"    {type(e).__name__}: {e}")
                traceback.print_exc()
                failed_ckpts.append((f"{ckpt_path} [test-confirm gs={r['guidance_scale']} "
                                     f"steps={r['ddim_steps']}]", f"{type(e).__name__}: {e}"))
                continue

        if test_confirm:
            val_best_row = test_confirm[0]
            test_best_row = max(test_confirm, key=lambda r: r["test_dice"])
            print(f"\nVal-best combo ({os.path.relpath(val_best_row['ckpt'])}, "
                  f"gs={val_best_row['guidance_scale']}, steps={val_best_row['ddim_steps']}) "
                  f"scores test Dice {val_best_row['test_dice']:.4f}.")
            if test_best_row is not val_best_row:
                print(f"NOTE: a different combo scores higher on test: "
                      f"{os.path.relpath(test_best_row['ckpt'])}, "
                      f"gs={test_best_row['guidance_scale']}, "
                      f"steps={test_best_row['ddim_steps']} -> test Dice "
                      f"{test_best_row['test_dice']:.4f}. The val ranking did not fully "
                      f"transfer to test -- with small val splits this can easily be noise "
                      f"rather than a real difference between these combos/checkpoints.")
            else:
                print("Val-best combo is also test-best among the confirmed candidates -- "
                      "consistent ranking, less likely to be a val-specific fluke.")

    with open(args.out_json, "w") as f:
        json.dump({
            "ckpt_dir": args.ckpt_dir,
            "checkpoints_found": ckpts,
            "checkpoints_failed": failed_ckpts,
            "split": args.split,
            "results": all_results,
            "test_confirm": test_confirm,
        }, f, indent=2)

    if args.split == "val" and args.skip_test_confirm:
        print("\n(--skip_test_confirm set) Confirm manually on test with e.g.:")
        print(f"  python -m src.sample --ckpt {best['ckpt']} --dataset {best['dataset']} "
              f"--split test --guidance_scale {best['guidance_scale']} "
              f"--ddim_steps {best['ddim_steps']} --viz test_grid_tuned.png")

    if args.viz_best:
        ckpt_path = best["ckpt"]
        if ckpt_path in loaded_cache:
            cfg, encoder, denoiser, diffusion = loaded_cache[ckpt_path]
            device = torch.device(cfg.device if torch.cuda.is_available() else "cpu")
            _, loader = build_loader(cfg, "vl" if args.split == "val" else "te", device)
            cfg.guidance_scale = best["guidance_scale"]
            cfg.ddim_steps = best["ddim_steps"]
            evaluate(encoder, denoiser, diffusion, loader, cfg, device, viz_path=args.viz_best, tta=args.tta)
            print(f"Saved viz for overall-best combo -> {args.viz_best}")
        else:
            print(f"Could not save viz_best: checkpoint {ckpt_path!r} is not loaded "
                  f"(it may have failed earlier).")

    if failed_ckpts:
        print(f"\n{len(failed_ckpts)} item(s) were skipped due to errors:")
        for c, err in failed_ckpts:
            print(f"  - {c}: {err}")

    print(f"\nFull results -> {args.out_json}")


if __name__ == "__main__":
    main()