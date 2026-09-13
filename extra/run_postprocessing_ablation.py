"""
Runs the full pre-registered post-processing ablation and writes a table.

Configs tested (in order of increasing cost) when >1 experiment is given:
    1. single_best                  -- one checkpoint (first seed of --best_experiment), no tricks.
    2. single_best+tta              -- (1) + 4-way TTA.
    3. seed_ensemble                -- --best_experiment, all its seeds, majority vote.
    4. seed_ensemble+cc             -- (3) + keep_largest_component.
    5. seed_ensemble+smooth         -- (3) + contour smoothing.
    6. seed_ensemble+cc+smooth      -- (3) + both.
    7. seed_ensemble+tta             -- (3) + 4-way TTA per checkpoint.
    8. cross_run_ensemble            -- all --cross_run_experiments pooled, majority vote.
    9. cross_run_ensemble+cc+smooth  -- (8) + both post-processing steps.
   10. cross_run_ensemble+tta        -- (8) + TTA -- most expensive, run last.

If --cross_run_experiments is omitted or resolves to the SAME single
experiment as --best_experiment (e.g. a run directory with only one config's
5 seeds, like best_v17/boundary_05_seed_01..05), the cross_run_* configs are
automatically skipped -- there is nothing to pool across, and re-running the
seed-ensemble under a different name would just waste compute.

Usage (single-experiment run directory, e.g. best_v17):
    python -m x0_prediction_backup.run_postprocessing_ablation \
        --runs_root x0_prediction_backup/runs/best_v17 \
        --best_experiment boundary_05 \
        --skin_root data/datasets \
        --dataset isic2018 --split test \
        --out x0_prediction_backup/runs/best_v17/post_process.csv

Usage (multi-experiment run directory, e.g. ablation_v15a_local_drop):
    python -m x0_prediction_backup.run_postprocessing_ablation \
        --runs_root x0_prediction_backup/runs/ablation_v15a_local_drop \
        --best_experiment drop_1 \
        --cross_run_experiments drop_1 drop_2 drop_2_topk_none \
        --skin_root data/datasets \
        --dataset isic2018 --split test \
        --out ablation_results.csv
"""

import argparse
import csv
import time

import torch
from torch.utils.data import DataLoader

from .config import Config
from .data import build_contour_dataset
from .ensemble_predict import find_checkpoints, run_ensemble


def build_ablation_configs(
    ckpt_map_best,
    best_experiment,
    ckpt_map_cross,
    cross_experiments,
):
    """Returns a list of (name, ckpt_paths, kwargs) tuples defining every
    ablation cell.

    `kwargs` are passed straight through to run_ensemble().

    Cross-run configs are only included if `cross_experiments` names
    more than one distinct experiment -- pooling a single experiment
    with itself would just reproduce seed_ensemble under a different label.
    """

    best_ckpts = ckpt_map_best[best_experiment]

    # No test-time augmentation.
    none_tta = ("none",)

    # Four-way test-time augmentation.
    #
    # The predictions for hflip/vflip/hvflip must be transformed back
    # to the original image orientation inside run_ensemble() before
    # they are combined.
    full_tta = (
        "none",
        "hflip",
        "vflip",
        "hvflip",
    )

    configs = [
        # ---------------------------------------------------------------
        # Single best checkpoint
        # ---------------------------------------------------------------

        (
            "single_best",
            best_ckpts[:1],
            dict(
                tta_variants=none_tta,
                keep_largest=False,
                smooth=False,
            ),
        ),

        # Single best checkpoint + 4-way TTA
        (
            "single_best+tta",
            best_ckpts[:1],
            dict(
                tta_variants=full_tta,
                keep_largest=False,
                smooth=False,
            ),
        ),

        # ---------------------------------------------------------------
        # Seed ensemble
        # ---------------------------------------------------------------

        (
            "seed_ensemble",
            best_ckpts,
            dict(
                tta_variants=none_tta,
                keep_largest=False,
                smooth=False,
            ),
        ),

        (
            "seed_ensemble+cc",
            best_ckpts,
            dict(
                tta_variants=none_tta,
                keep_largest=True,
                smooth=False,
            ),
        ),

        (
            "seed_ensemble+smooth",
            best_ckpts,
            dict(
                tta_variants=none_tta,
                keep_largest=False,
                smooth=True,
            ),
        ),

        (
            "seed_ensemble+cc+smooth",
            best_ckpts,
            dict(
                tta_variants=none_tta,
                keep_largest=True,
                smooth=True,
            ),
        ),

        # Seed ensemble + 4-way TTA
        (
            "seed_ensemble+tta",
            best_ckpts,
            dict(
                tta_variants=full_tta,
                keep_largest=False,
                smooth=False,
            ),
        ),
    ]

    # ---------------------------------------------------------------
    # Cross-run ensemble
    # ---------------------------------------------------------------

    has_real_cross_run = len(set(cross_experiments)) > 1

    if has_real_cross_run:
        cross_ckpts = [
            p
            for paths in ckpt_map_cross.values()
            for p in paths
        ]

        configs += [
            (
                "cross_run_ensemble",
                cross_ckpts,
                dict(
                    tta_variants=none_tta,
                    keep_largest=False,
                    smooth=False,
                ),
            ),

            (
                "cross_run_ensemble+cc+smooth",
                cross_ckpts,
                dict(
                    tta_variants=none_tta,
                    keep_largest=True,
                    smooth=True,
                ),
            ),

            # Cross-run ensemble + 4-way TTA
            (
                "cross_run_ensemble+tta",
                cross_ckpts,
                dict(
                    tta_variants=full_tta,
                    keep_largest=False,
                    smooth=False,
                ),
            ),
        ]

    else:
        print(
            "Only one distinct experiment given -- skipping cross_run_* "
            "configs (nothing to pool across; pass --cross_run_experiments "
            "with 2+ names to enable them)."
        )

    return configs


def main():
    parser = argparse.ArgumentParser(
        description="Full post-processing ablation sweep"
    )

    parser.add_argument(
        "--runs_root",
        type=str,
        required=True,
    )

    parser.add_argument(
        "--skin_root",
        type=str,
        required=True,
    )

    parser.add_argument(
        "--dataset",
        choices=["ph2", "isic2017", "isic2018", "ham10000"],
        required=True,
    )

    parser.add_argument(
        "--split",
        choices=["val", "test"],
        default="test",
    )

    parser.add_argument(
        "--device",
        type=str,
        default=None,
    )

    parser.add_argument(
        "--out",
        type=str,
        default="ablation_results.csv",
    )

    parser.add_argument(
        "--best_experiment",
        type=str,
        required=True,
        help=(
            "Experiment name whose seeds form the seed_ensemble "
            "(and single_best baseline), e.g. 'drop_1' or 'boundary_05'. "
            "Must match the '<experiment>_seed_NN' folder prefix under "
            "--runs_root."
        ),
    )

    parser.add_argument(
        "--cross_run_experiments",
        type=str,
        nargs="*",
        default=None,
        help=(
            "Experiment names to pool for cross_run_ensemble configs. "
            "Omit (or pass just --best_experiment's name) to skip "
            "cross_run_* configs entirely, e.g. for a runs_root that "
            "only contains one experiment's seeds (like best_v17)."
        ),
    )

    parser.add_argument(
        "--skip",
        type=str,
        nargs="*",
        default=[],
        help=(
            "Names of configs to skip, e.g. "
            "--skip cross_run_ensemble+tta "
            "if you want to run the cheap ones first and add the "
            "expensive ones in a second pass."
        ),
    )

    args = parser.parse_args()

    # ---------------------------------------------------------------
    # Configuration / device
    # ---------------------------------------------------------------

    cfg = Config.from_args(args)

    device = torch.device(
        args.device
        or ("cuda" if torch.cuda.is_available() else "cpu")
    )

    # ---------------------------------------------------------------
    # Dataset
    # ---------------------------------------------------------------

    split = "vl" if args.split == "val" else "te"

    ds = build_contour_dataset(
        cfg.skin_root,
        args.dataset,
        split,
        cfg.n_points,
        cfg.img_size,
        augment=False,
        npy_size=cfg.npy_size,
    )

    loader = DataLoader(
        ds,
        batch_size=cfg.batch_size,
        shuffle=False,
    )

    # ---------------------------------------------------------------
    # Find checkpoints
    # ---------------------------------------------------------------

    cross_experiments = (
        args.cross_run_experiments
        or [args.best_experiment]
    )

    ckpt_map_best = find_checkpoints(
        args.runs_root,
        [args.best_experiment],
    )

    ckpt_map_cross = None

    if len(set(cross_experiments)) > 1:
        ckpt_map_cross = find_checkpoints(
            args.runs_root,
            cross_experiments,
        )

    # ---------------------------------------------------------------
    # Build ablation configs
    # ---------------------------------------------------------------

    configs = build_ablation_configs(
        ckpt_map_best,
        args.best_experiment,
        ckpt_map_cross,
        cross_experiments,
    )

    # ---------------------------------------------------------------
    # Run ablation
    # ---------------------------------------------------------------

    rows = []

    for name, ckpt_paths, kwargs in configs:

        if name in args.skip:
            print(f"-- skipping {name} --")
            continue

        n_votes = (
            len(ckpt_paths)
            * len(kwargs["tta_variants"])
        )

        print(
            f"\n=== {name} "
            f"({len(ckpt_paths)} checkpoint(s), "
            f"{n_votes} votes/image) ==="
        )

        t0 = time.time()

        dice, iou = run_ensemble(
            ckpt_paths,
            cfg,
            device,
            loader,
            **kwargs,
        )

        elapsed = time.time() - t0

        print(
            f"    Dice {dice:.4f} | "
            f"IoU {iou:.4f} | "
            f"{elapsed:.1f}s"
        )

        rows.append(
            {
                "config": name,
                "n_checkpoints": len(ckpt_paths),
                "n_votes_per_image": n_votes,
                "dice": round(dice, 4),
                "iou": round(iou, 4),
                "seconds": round(elapsed, 1),
            }
        )

        # -----------------------------------------------------------
        # Write incrementally.
        #
        # This prevents losing the cheaper results if the process
        # crashes/is killed while running one of the expensive TTA
        # configurations.
        # -----------------------------------------------------------

        with open(
            args.out,
            "w",
            newline="",
        ) as f:
            writer = csv.DictWriter(
                f,
                fieldnames=list(rows[0].keys()),
            )

            writer.writeheader()
            writer.writerows(rows)

    # ---------------------------------------------------------------
    # Final summary
    # ---------------------------------------------------------------

    print(f"\nResults written to {args.out}")

    print(
        f"{'config':<32} "
        f"{'dice':>8} "
        f"{'iou':>8}"
    )

    for r in rows:
        print(
            f"{r['config']:<32} "
            f"{r['dice']:>8.4f} "
            f"{r['iou']:>8.4f}"
        )


if __name__ == "__main__":
    main()