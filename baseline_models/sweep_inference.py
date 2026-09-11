"""Sweep guidance_scale / ddim_steps for a trained P2SDiff checkpoint on VAL.

Loads the model ONCE (backbone + denoiser), then re-runs DDIM sampling for
every (guidance_scale, ddim_steps) combination on the *validation* split
only -- so the sweep itself never touches test.

After the val sweep, the top-5 combinations (by val Dice) are automatically
re-evaluated on the *test* split, once each. This is the confirmation step:
with only ~20 val images, picking "the best" combo out of many is noisy, so
seeing val vs. test side-by-side for the top-5 tells you whether the winner
is a real improvement or a val-specific fluke (if the ranking flips between
val and test, that's a strong signal of exactly that). This does NOT get
cheaper by only looking at the #1 val combo -- the whole point is comparing
several candidates against test to see how stable the ranking is.

Usage (run from the repository root, i.e. one level above `src/`):

    python -m src.sweep_inference \
        --ckpt src/runs/baseline/ph2_ft_ham_e400/best.pth \
        --dataset ph2 \
        --guidance_scales 1.0 1.5 2.0 2.5 3.0 4.0 \
        --ddim_steps_list 25 50 100

Or, if you'd rather keep it as a standalone script outside the package,
copy it next to train.py/sample.py and run:

    python sweep_inference.py --ckpt <path> --dataset ph2

Use --skip_test_confirm to skip the top-5 test pass (val sweep only, e.g.
for a quick first look), and --top_k to change how many combos get the
test confirmation pass (default: 5).

Results are written to <out_json> (default: sweep_results.json): the full
val sweep, sorted best-Dice-first, plus (unless skipped) a `test_confirm`
list with val/test Dice+IoU for each of the top-k combinations.
"""

import argparse
import itertools
import json
import os

import torch
from torch.utils.data import DataLoader

from .config import Config
from .data import build_contour_dataset
from .diffusion import GaussianDiffusion
from .sample import evaluate, load_checkpoint


def parse_args():
    p = argparse.ArgumentParser(description="Sweep guidance_scale / ddim_steps on VAL")
    p.add_argument("--ckpt", type=str, required=True, help="path to best.pth / last.pth")
    p.add_argument("--dataset", choices=["ph2", "isic2017", "isic2018", "ham10000"], default=None,
                   help="override dataset; defaults to whatever the checkpoint's config used")
    p.add_argument("--skin_root", type=str, default=None)
    p.add_argument("--split", choices=["val", "test"], default="val",
                   help="sweep split (default: val -- keep test untouched until the final run)")
    p.add_argument("--batch_size", type=int, default=None)
    p.add_argument("--device", type=str, default=None)

    p.add_argument("--guidance_scales", type=float, nargs="+",
                   default=[1.0, 1.5, 2.0, 2.5, 3.0, 4.0])
    p.add_argument("--ddim_steps_list", type=int, nargs="+",
                   default=[25, 50, 75, 100, 200])  # add e.g. 25 100 to also sweep step count

    p.add_argument("--out_json", type=str, default="sweep_results.json")
    p.add_argument("--viz_best", type=str, default=None,
                   help="if set, saves a prediction grid for the best combo to this path")

    p.add_argument("--top_k", type=int, default=5,
                   help="how many top val combos to re-confirm on test (default: 5)")
    p.add_argument("--skip_test_confirm", action="store_true",
                   help="skip the automatic top-k test confirmation pass "
                        "(only meaningful when --split val)")
    p.add_argument("--tta", action="store_true",
                   help="enable test-time augmentation (TTA) for the sweep; "
                        "if the checkpoint was trained with TTA, this should be set")
    return p.parse_args()


def main():
    args = parse_args()

    cfg = Config()
    if args.dataset:
        cfg.dataset = args.dataset
    if args.skin_root:
        cfg.skin_root = args.skin_root
    if args.batch_size:
        cfg.batch_size = args.batch_size
    if args.device:
        cfg.device = args.device

    device = torch.device(cfg.device if torch.cuda.is_available() else "cpu")

    # load_checkpoint overwrites cfg fields (encoder/backbone/n_points/...) from
    # the checkpoint's saved config, so architecture always matches the weights.
    print(f"Loading checkpoint: {args.ckpt}")
    encoder, denoiser = load_checkpoint(args.ckpt, cfg, device)
    print(f"  encoder={cfg.encoder} backbone={cfg.backbone} n_points={cfg.n_points} "
          f"dataset={cfg.dataset}")

    diffusion = GaussianDiffusion(cfg.timesteps, cfg.beta_start, cfg.beta_end, device=device)

    split_key = "vl" if args.split == "val" else "te"
    ds = build_contour_dataset(cfg.skin_root, cfg.dataset, split_key, cfg.n_points,
                               cfg.img_size, augment=False, npy_size=cfg.npy_size)
    loader = DataLoader(ds, batch_size=cfg.batch_size, shuffle=False)
    print(f"Sweeping on split='{args.split}' ({len(ds)} images)")

    combos = list(itertools.product(args.guidance_scales, args.ddim_steps_list))
    print(f"{len(combos)} combinations: guidance_scales={args.guidance_scales} "
          f"x ddim_steps_list={args.ddim_steps_list}")

    results = []
    best = None
    for gs, steps in combos:
        cfg.guidance_scale = gs
        cfg.ddim_steps = steps
        dice, iou = evaluate(encoder, denoiser, diffusion, loader, cfg, device, viz_path=None, tta=args.tta)
        row = {"guidance_scale": gs, "ddim_steps": steps, "dice": dice, "iou": iou}
        results.append(row)
        print(f"  gs={gs:<5} steps={steps:<4} -> Dice {dice:.4f} | IoU {iou:.4f}")
        if best is None or dice > best["dice"]:
            best = row

    results.sort(key=lambda r: r["dice"], reverse=True)

    print(f"\nTop {min(args.top_k, len(results))} on {args.split}:")
    for r in results[:args.top_k]:
        print(f"  gs={r['guidance_scale']:<5} steps={r['ddim_steps']:<4} "
              f"Dice {r['dice']:.4f} | IoU {r['iou']:.4f}")

    print(f"\nBest on {args.split}: guidance_scale={best['guidance_scale']} "
          f"ddim_steps={best['ddim_steps']} -> Dice {best['dice']:.4f}")

    # ------------------------------------------------------------------ #
    # Test confirmation pass: re-evaluate the top-k val combos on test,  #
    # once each. Only makes sense if the sweep itself ran on val.        #
    # ------------------------------------------------------------------ #
    test_confirm = []
    if args.split == "val" and not args.skip_test_confirm:
        top_k = results[:args.top_k]
        print(f"\nConfirming top {len(top_k)} combo(s) on TEST "
              f"(sampling {len(top_k)} time(s), each on the full test split)...")

        test_ds = build_contour_dataset(cfg.skin_root, cfg.dataset, "te", cfg.n_points,
                                        cfg.img_size, augment=False, npy_size=cfg.npy_size)
        test_loader = DataLoader(test_ds, batch_size=cfg.batch_size, shuffle=False)
        print(f"  test split: {len(test_ds)} images")

        for r in top_k:
            cfg.guidance_scale = r["guidance_scale"]
            cfg.ddim_steps = r["ddim_steps"]
            t_dice, t_iou = evaluate(encoder, denoiser, diffusion, test_loader, cfg, device,
                                     viz_path=None, tta=args.tta)
            row = {**r, "test_dice": t_dice, "test_iou": t_iou}
            test_confirm.append(row)
            print(f"  gs={r['guidance_scale']:<5} steps={r['ddim_steps']:<4} "
                  f"val Dice {r['dice']:.4f} -> test Dice {t_dice:.4f} | test IoU {t_iou:.4f}")

        # Does the val-best combo still win on test? If not, flag it --
        # that's the val-overfitting signal this confirmation pass exists for.
        val_best_row = test_confirm[0]
        test_best_row = max(test_confirm, key=lambda r: r["test_dice"])
        print(f"\nVal-best combo (gs={val_best_row['guidance_scale']}, "
              f"steps={val_best_row['ddim_steps']}) scores test Dice "
              f"{val_best_row['test_dice']:.4f}.")
        if test_best_row is not val_best_row:
            print(f"NOTE: a different combo among the top-{len(top_k)} scores higher on "
                  f"test: gs={test_best_row['guidance_scale']}, "
                  f"steps={test_best_row['ddim_steps']} -> test Dice "
                  f"{test_best_row['test_dice']:.4f}. The val ranking did not fully transfer "
                  f"to test -- with only {len(ds)} val images this can easily be "
                  f"noise rather than a real difference between these combos.")
        else:
            print("Val-best combo is also test-best among the top candidates -- "
                  "consistent ranking, less likely to be a val-specific fluke.")

    with open(args.out_json, "w") as f:
        json.dump({"ckpt": args.ckpt, "dataset": cfg.dataset, "split": args.split,
                   "results": results, "test_confirm": test_confirm}, f, indent=2)

    if args.split == "val" and args.skip_test_confirm:
        print("\n(--skip_test_confirm set) Confirm manually on test with:")
        print(f"  python -m src.sample --ckpt {args.ckpt} --dataset {cfg.dataset} "
              f"--split test --guidance_scale {best['guidance_scale']} "
              f"--ddim_steps {best['ddim_steps']} --viz test_grid_tuned.png")

    if args.viz_best:
        cfg.guidance_scale = best["guidance_scale"]
        cfg.ddim_steps = best["ddim_steps"]
        evaluate(encoder, denoiser, diffusion, loader, cfg, device, viz_path=args.viz_best, tta=args.tta)
        print(f"Saved viz for best combo (val split) -> {args.viz_best}")

    print(f"\nFull results -> {args.out_json}")


if __name__ == "__main__":
    main()