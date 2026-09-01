#!/usr/bin/env python3
"""Export exact dataset splits and reference configs for P2SDiff v2 runs."""

from __future__ import annotations

import argparse
import csv
import glob
import hashlib
import importlib.metadata
import json
import platform
import shutil
import subprocess
from pathlib import Path

import numpy as np


DATASETS = {
    "ph2": {
        "directory": "PH2",
        "images": "trainx/*.bmp",
        "masks": "trainy/{id}_lesion.bmp",
        "splits": (80, 20, 100),
    },
    "isic2017": {
        "directory": "ISIC2017",
        "images": "ISIC-2017_Training_Data/*.jpg",
        "masks": "ISIC-2017_Training_Part1_GroundTruth/{id}_segmentation.png",
        "splits": (1250, 150, 600),
    },
    "isic2018": {
        "directory": "ISIC2018",
        "images": "ISIC2018_Task1-2_Training_Input/*.jpg",
        "masks": "ISIC2018_Task1_Training_GroundTruth/{id}_segmentation.png",
        "splits": (1815, 259, 520),
    },
    "ham10000": {
        "directory": "HAM10000",
        "images": "images/*.jpg",
        "masks": "masks/{id}_segmentation.png",
        "splits": (7200, 1800, 1015),
    },
}

REFERENCE_RUNS = (
    "v2_ham_n100_gs15_d05_e50",
    "v2_ham_strong_e60",
    "v2_isic17_ft_ham_e150",
    "v2_isic17_n100_gs15_d05_e120",
    "v2_isic17_n100_gs15_d05_e600",
    "v2_isic17_n150_gs15_d05_e400",
    "v2_isic17_softdice96_e400",
    "v2_isic17_strong_e400",
    "v2_isic18_ft_ham_e150",
    "v2_isic18_n100_gs15_d05_e400",
    "v2_isic18_strong_e400",
    "v2_ph2_ft_ham_e400",
    "v2_ph2_n100_gs15_d05_e800",
    "v2_ph2_n100_strong_e800",
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def split_for(index: int, lengths: tuple[int, int, int]) -> str:
    train, val, _ = lengths
    if index < train:
        return "train"
    if index < train + val:
        return "val"
    return "test"


def export_splits(skin_root: Path, output: Path) -> dict:
    metadata = {}
    split_dir = output / "splits"
    split_dir.mkdir(parents=True, exist_ok=True)

    for name, spec in DATASETS.items():
        dataset_root = skin_root / spec["directory"]
        # Deliberately preserve glob.glob order: this is the order used when the
        # reference npy arrays were originally generated.
        images = [Path(path) for path in glob.glob(str(dataset_root / spec["images"]))]
        expected = sum(spec["splits"])
        if len(images) != expected:
            raise RuntimeError(f"{name}: found {len(images)} images, expected {expected}")

        rows = []
        ids_by_split = {"train": [], "val": [], "test": []}
        for index, image in enumerate(images):
            image_id = image.stem
            mask = dataset_root / spec["masks"].format(id=image_id)
            if not mask.is_file():
                raise FileNotFoundError(f"{name}: missing mask for {image_id}: {mask}")
            split = split_for(index, spec["splits"])
            ids_by_split[split].append(image_id)
            rows.append(
                {
                    "index": index,
                    "split": split,
                    "image_id": image_id,
                    "image_path": image.relative_to(dataset_root).as_posix(),
                    "mask_path": mask.relative_to(dataset_root).as_posix(),
                }
            )

        with (split_dir / f"{name}.csv").open("w", newline="") as file:
            writer = csv.DictWriter(file, fieldnames=rows[0].keys())
            writer.writeheader()
            writer.writerows(rows)

        for split, ids in ids_by_split.items():
            (split_dir / f"{name}_{split}.txt").write_text("\n".join(ids) + "\n")

        arrays = {}
        for prefix in ("X", "Y"):
            path = dataset_root / "np" / f"{prefix}_tr_224x224.npy"
            array = np.load(path, mmap_mode="r")
            arrays[prefix] = {
                "filename": path.name,
                "shape": list(array.shape),
                "dtype": str(array.dtype),
                "min": int(array.min()),
                "max": int(array.max()),
                "bytes": path.stat().st_size,
                "sha256": sha256(path),
            }

        metadata[name] = {
            "directory": spec["directory"],
            "counts": dict(zip(("train", "val", "test"), spec["splits"])),
            "arrays": arrays,
        }

    (output / "array_metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")
    return metadata


def export_runs(runs_root: Path, output: Path) -> None:
    destination = output / "reference_runs"
    destination.mkdir(parents=True, exist_ok=True)
    index = {}

    for run in REFERENCE_RUNS:
        source = runs_root / run
        if not source.is_dir():
            raise FileNotFoundError(f"Missing reference run: {source}")

        copied = []
        for filename in ("config.json", "summary.json", "test_metrics.json"):
            path = source / filename
            if path.is_file():
                shutil.copy2(path, destination / f"{run}.{filename}")
                copied.append(filename)

        summary_path = source / "summary.json"
        index[run] = {
            "files": copied,
            "summary": json.loads(summary_path.read_text()) if summary_path.is_file() else None,
        }

    (destination / "index.json").write_text(json.dumps(index, indent=2) + "\n")


def export_environment(output: Path) -> None:
    try:
        commit = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], text=True
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        commit = None

    packages = {}
    for name in ("torch", "torchvision", "numpy", "opencv-python"):
        try:
            packages[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            packages[name] = None

    environment = {
        "git_commit": commit,
        "python": platform.python_version(),
        "packages": packages,
    }
    (output / "reference_environment.json").write_text(
        json.dumps(environment, indent=2) + "\n"
    )


def write_readme(output: Path) -> None:
    text = """# P2SDiff v2 reproducibility bundle

This bundle records the exact data order and configurations used for the
reported v2 experiments. It does not contain the medical images.

## Dataset splits

`splits/<dataset>_{train,val,test}.txt` contains the image IDs in each split.
`splits/<dataset>.csv` additionally records the original array index and the
relative image/mask paths.

The order matters. The original preprocessing used unsorted `glob.glob()`
output and then split the resulting arrays by fixed index ranges. Recreating
the arrays with a different file order produces different partitions.

Counts:

- PH2: 80 train / 20 validation / 100 test
- ISIC2017: 1250 train / 150 validation / 600 test
- ISIC2018: 1815 train / 259 validation / 520 test
- HAM10000: 7200 train / 1800 validation / 1015 test

## Verification

`array_metadata.json` contains the exact shape, dtype, value range, file size,
and SHA-256 checksum of each reference `X_` and `Y_` npy file. Matching hashes
mean that preprocessing and ordering are identical. Different hashes do not
necessarily mean the source images differ; a different array order also
changes the hash.

## Reference runs

`reference_runs/` contains the exact `config.json`, `summary.json`, and
`test_metrics.json` files from the reported runs. Local paths such as
`skin_root`, `out_dir`, and `init_checkpoint` must be adapted to the target
machine. Other parameters should match for replication.

`reference_environment.json` records the source commit and key package
versions. Exact GPU kernels can still introduce small numerical differences,
but they should not explain large Dice-score gaps.

For `ft_ham` runs, initialize from the `best.pth` checkpoint of the selected
HAM10000 reference run. Evaluation should use `best.pth`; the guidance scale
is read from the saved configuration and is 1.5 for these v2 runs.
"""
    (output / "README.md").write_text(text)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--skin-root", type=Path, default=Path("/hdd/datasets/Skin"))
    parser.add_argument("--runs-root", type=Path, default=Path("src/runs"))
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("reproducibility/p2sdiff_v2_reference"),
    )
    args = parser.parse_args()

    args.output.mkdir(parents=True, exist_ok=True)
    export_splits(args.skin_root, args.output)
    export_runs(args.runs_root, args.output)
    export_environment(args.output)
    write_readme(args.output)
    print(f"Wrote reproducibility bundle to {args.output}")


if __name__ == "__main__":
    main()
