"""Multi-dataset boundary-point loaders with *published* index splits.

The reference pipeline (src/data/skin/, src/data/dataset_*.py) preprocesses each
dataset into channels-first uint8 arrays saved as
`<root>/<DATASET>/np/X_tr_{S}x{S}.npy` (+ Y), in glob order, and splits them by a
fixed index range. We read the *same* arrays and apply the *same* slices, so our
train/val/test partition is identical to the reference — only the target differs
(we derive boundary points from each mask instead of using the pixel mask).

Splits (train / val / test), matching the reference loaders:
    ph2        80   / 20   / 100    (total 200)
    isic2017   1250 / 150  / 600    (total 2000)
    isic2018   1815 / 259  / 520    (total 2594)
    ham10000   7200 / 1800 / 1015   (total 10015)

BUSI and Polyp use different conventions (random 80/20 with seed; predefined
train/test folders) and are added separately when needed.
"""

import os

import numpy as np

from .ph2_dataset import ArrayContourDataset

DATASET_DIRS = {
    "ph2": "PH2",
    "isic2017": "ISIC2017",
    "isic2018": "ISIC2018",
    "ham10000": "HAM10000",
}

# (train, val, test) lengths over the glob-ordered npy arrays.
DATASET_SPLITS = {
    "ph2": (80, 20, 100),
    "isic2017": (1250, 150, 600),
    "isic2018": (1815, 259, 520),
    "ham10000": (7200, 1800, 1015),   # te = X[9000:] = 10015-9000 (ref comment "2015" is wrong)
}

_SPLIT_ALIASES = {"tr": "tr", "train": "tr", "vl": "vl", "val": "vl",
                  "te": "te", "test": "te"}


def _load_npy(skin_root, dataset, npy_size):
    d = DATASET_DIRS[dataset]
    npy_dir = os.path.join(skin_root, d, "np")
    x = os.path.join(npy_dir, f"X_tr_{npy_size}x{npy_size}.npy")
    y = os.path.join(npy_dir, f"Y_tr_{npy_size}x{npy_size}.npy")
    if not (os.path.isfile(x) and os.path.isfile(y)):
        raise FileNotFoundError(
            f"Missing preprocessed arrays for '{dataset}':\n  {x}\n  {y}\n"
            f"Generate them with the reference loader (src/data/skin/) first."
        )
    return np.load(x), np.load(y)


def _slices(dataset, n):
    tr, vl, te = DATASET_SPLITS[dataset]
    if tr + vl + te != n:
        raise ValueError(
            f"{dataset}: npy has {n} samples but split sums to {tr+vl+te}. "
            f"The npy ordering/size may differ from the reference."
        )
    return {"tr": slice(0, tr), "vl": slice(tr, tr + vl), "te": slice(tr + vl, n)}


def build_contour_dataset(skin_root, dataset, split, n_points=200,
                          img_size=(224, 224), augment=False, aug_level="strong",
                          npy_size=224):
    """Return an ArrayContourDataset for one split of a named dataset."""
    dataset = dataset.lower()
    if dataset not in DATASET_DIRS:
        raise ValueError(f"Unknown dataset '{dataset}'. Known: {list(DATASET_DIRS)}")
    split = _SPLIT_ALIASES.get(split, split)
    if split not in ("tr", "vl", "te"):
        raise ValueError(f"Unknown split '{split}'. Use tr/vl/te (or train/val/test).")

    X, Y = _load_npy(skin_root, dataset, npy_size)
    sel = _slices(dataset, len(X))[split]
    return ArrayContourDataset(
        X[sel], Y[sel], n_points=n_points, img_size=img_size,
        augment=augment, aug_level=aug_level,
    )


def split_counts(skin_root, dataset, npy_size=224):
    """(train, val, test) sizes — handy for logging / sanity checks."""
    X, _ = _load_npy(skin_root, dataset.lower(), npy_size)
    s = _slices(dataset.lower(), len(X))
    return {k: v.stop - v.start for k, v in s.items()}
