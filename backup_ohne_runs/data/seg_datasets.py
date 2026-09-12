"""Multi-dataset boundary-point loaders with *published* index splits.

The reference pipeline (src/data/skin/, src/data/dataset_*.py) preprocesses each
dataset into channels-first uint8 arrays saved as
`<root>/<DATASET>/np/X_tr_{S}x{S}.npy` (+ Y), in glob order, and splits them by a
fixed index range. We read the *same* arrays and apply the *same* slices, so our
train/val/test partition is identical to the reference — only the target differs
(we derive boundary points from each mask instead of using the pixel mask).

Three dataset "kinds" now exist:

1. INDEXED_DATASETS -- a single preprocessed pool ('tr' npy file) that we
   slice by fixed index ranges into train/val/test, matching the reference
   loaders exactly:
       isic2017   1250 / 150  / 600    (total 2000)
       isic2018   1815 / 259  / 520    (total 2594)
       ham10000   7200 / 1800 / 1015   (total 10015)

2. TEST_ONLY_DATASETS -- preprocessed into a single 'te' npy file, no tr/vl
   exist at all, by design:
       ph2   200 (all samples) -- a standalone, unseen zero-shot test set.
             Protocol: train on isic2018, evaluate zero-shot on ph2. PH2 is
             NEVER trained on and NEVER used for early stopping/checkpoint
             selection -- doing so would leak "unseen-ness" into model
             selection and undermine the zero-shot claim. It is only ever
             loaded with split="te", as an *additional* test alongside
             isic2018's own held-out test set.

3. DIRECT_SPLIT_DATASETS -- preprocessed directly into separate
   X_tr/X_vl/X_te (+Y) npy files per split, no index slicing needed:
       busi              seeded 70/10/20 random split (benign+malignant only)
       tn3k              seeded 80/20 tr/vl from trainval pool + official test
       polyp_clinicdb    PraNet protocol: Kvasir-SEG+CVC-ClinicDB train pool
       polyp_kvasir        (seeded 90/10 tr/vl), each of the 5 official
       polyp_colondb       TestDataset subfolders as its own 'te', all 5
       polyp_etis          sharing the SAME tr/vl pool.
       polyp_cvc300
"""

import os

import numpy as np

from .ph2_dataset import ArrayContourDataset

DATASET_DIRS = {
    "isic2017": "isic2017",
    "isic2018": "isic2018",
    "ham10000": "ham10000",
    "ph2": "ph2",
    "busi": "busi",
    "tn3k": "tn3k",
    "polyp_clinicdb": "polyp_clinicdb",
    "polyp_kvasir": "polyp_kvasir",
    "polyp_colondb": "polyp_colondb",
    "polyp_etis": "polyp_etis",
    "polyp_cvc300": "polyp_cvc300",
}

INDEXED_DATASETS = {"isic2017", "isic2018", "ham10000"}

DIRECT_SPLIT_DATASETS = {
    "busi", "tn3k",
    "polyp_clinicdb", "polyp_kvasir", "polyp_colondb", "polyp_etis", "polyp_cvc300",
}

# PH2 is neither indexed nor a normal direct-split dataset: it only ever has
# a 'te' split. Kept as its own set so build_contour_dataset / split_counts
# can enforce that explicitly (clear error on 'tr'/'vl') instead of silently
# falling into either of the two paths above.
TEST_ONLY_DATASETS = {"ph2"}

# (train, val, test) lengths over the glob-ordered npy arrays -- only used
# for INDEXED_DATASETS. (No entries for ph2/busi: ph2 is test-only now, busi
# is direct-split and was never sliced through this table.)
DATASET_SPLITS = {
    "isic2017": (1250, 150, 600),
    "isic2018": (1815, 259, 520),
    "ham10000": (7200, 1800, 1013),   # te = X[9000:] = 10015-9000 (ref comment "2015" is wrong)
}

_SPLIT_ALIASES = {"tr": "tr", "train": "tr", "vl": "vl", "val": "vl",
                  "te": "te", "test": "te"}


def _load_npy(skin_root, dataset, split, npy_size):
    """Load the npy pair for a dataset. For INDEXED_DATASETS this always
    loads the 'tr' pool file (to be sliced later); for TEST_ONLY_DATASETS
    and DIRECT_SPLIT_DATASETS it loads the file matching the requested
    split directly."""
    d = DATASET_DIRS[dataset]
    npy_dir = os.path.join(skin_root, d, "np")

    prefix = "tr" if dataset in INDEXED_DATASETS else split
    x = os.path.join(npy_dir, f"X_{prefix}_{npy_size}x{npy_size}.npy")
    y = os.path.join(npy_dir, f"Y_{prefix}_{npy_size}x{npy_size}.npy")
    if not (os.path.isfile(x) and os.path.isfile(y)):
        raise FileNotFoundError(
            f"Missing preprocessed arrays for '{dataset}' (split='{prefix}'):\n  {x}\n  {y}\n"
            f"Generate them with the preprocessing script first."
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
                          npy_size=224, adaptive_sampling=True):
    """Return an ArrayContourDataset for one split of a named dataset.

    Raises a clear ValueError (not a confusing downstream crash) if a
    tr/vl split is requested for a test-only dataset like ph2.
    """
    dataset = dataset.lower()
    if dataset not in DATASET_DIRS:
        raise ValueError(f"Unknown dataset '{dataset}'. Known: {list(DATASET_DIRS)}")
    split = _SPLIT_ALIASES.get(split, split)
    if split not in ("tr", "vl", "te"):
        raise ValueError(f"Unknown split '{split}'. Use tr/vl/te (or train/val/test).")

    if dataset in TEST_ONLY_DATASETS:
        if split != "te":
            raise ValueError(
                f"'{dataset}' is a test-only, zero-shot dataset -- no 'tr'/'vl' exist "
                f"by design. It is meant to be evaluated (split='te' only) against a "
                f"model trained on isic2018, never trained on directly and never used "
                f"for early stopping/checkpoint selection. Requested split='{split}' "
                f"is not available."
            )
        X, Y = _load_npy(skin_root, dataset, split, npy_size)
    elif dataset in INDEXED_DATASETS:
        X, Y = _load_npy(skin_root, dataset, split, npy_size)
        sel = _slices(dataset, len(X))[split]
        X, Y = X[sel], Y[sel]
    else:
        assert dataset in DIRECT_SPLIT_DATASETS
        X, Y = _load_npy(skin_root, dataset, split, npy_size)

    return ArrayContourDataset(
        X, Y, n_points=n_points, img_size=img_size,
        augment=augment, aug_level=aug_level,
        adaptive_sampling=adaptive_sampling,
    )


def split_counts(skin_root, dataset, npy_size=224):
    """(train, val, test) sizes — handy for logging / sanity checks."""
    dataset = dataset.lower()
    if dataset in TEST_ONLY_DATASETS:
        X, _ = _load_npy(skin_root, dataset, "te", npy_size)
        return {"tr": 0, "vl": 0, "te": len(X)}
    elif dataset in INDEXED_DATASETS:
        X, _ = _load_npy(skin_root, dataset, "tr", npy_size)
        s = _slices(dataset, len(X))
        return {k: v.stop - v.start for k, v in s.items()}
    else:
        assert dataset in DIRECT_SPLIT_DATASETS
        counts = {}
        for split in ("tr", "vl", "te"):
            X, _ = _load_npy(skin_root, dataset, split, npy_size)
            counts[split] = len(X)
        return counts