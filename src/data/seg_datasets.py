"""Contour-dataset catalog: skin, BUSI, polyp/Kvasir, ACDC, Synapse.

Skin datasets keep the published npy index splits under `/hdd/datasets/Skin`.
The others are read from `/hdd/datasets/` in the same layout MaLViL used, then
converted to (image, ordered-contour, binary mask) for P2SDiff.

ACDC and Synapse are multi-class sources; we binarize (any organ vs background)
so they fit the single-contour model. Empty-mask slices are dropped.
"""

from __future__ import annotations

import os
import random
from functools import lru_cache

import cv2
import numpy as np

from .ph2_dataset import ArrayContourDataset, PH2ContourDataset, _as_uint8_rgb

DATASET_NAMES = (
    "ph2", "isic2017", "isic2018", "ham10000",
    "busi",
    "polyp", "kvasir",
    "acdc",
    "synapse",
)

# Skin: directory under skin_root and (train, val, test) lengths over glob-ordered npy.
DATASET_DIRS = {
    "ph2": "PH2",
    "isic2017": "ISIC2017",
    "isic2018": "ISIC2018",
    "ham10000": "HAM10000",
}

DATASET_SPLITS = {
    "ph2": (80, 20, 100),
    "isic2017": (1250, 150, 600),
    "isic2018": (1815, 259, 520),
    "ham10000": (7200, 1800, 1015),
}

POLYP_TEST_FOLDERS = {
    "kvasir": "Kvasir",
    "cvc300": "CVC-300",
    "clinic": "CVC-ClinicDB",
    "colon": "CVC-ColonDB",
    "etis": "ETIS-LaribPolypDB",
    "test": "test",
}

_SPLIT_ALIASES = {
    "tr": "tr", "train": "tr",
    "vl": "vl", "val": "vl",
    "te": "te", "test": "te",
}

_IMG_EXTS = (".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff")


def _norm_split(split: str) -> str:
    split = _SPLIT_ALIASES.get(split, split)
    if split not in ("tr", "vl", "te"):
        raise ValueError(f"Unknown split '{split}'. Use tr/vl/te (or train/val/test).")
    return split


def _file_dataset(pairs, n_points, img_size, augment, aug_level):
    return PH2ContourDataset(
        pairs, n_points=n_points, img_size=img_size,
        augment=augment, aug_level=aug_level,
    )


def _array_dataset(images, masks, n_points, img_size, augment, aug_level):
    return ArrayContourDataset(
        images, masks, n_points=n_points, img_size=img_size,
        augment=augment, aug_level=aug_level,
    )


# ----- skin (preprocessed npy, published index ranges) ---------------------

def _load_skin_npy(skin_root, dataset, npy_size):
    d = DATASET_DIRS[dataset]
    npy_dir = os.path.join(skin_root, d, "np")
    x = os.path.join(npy_dir, f"X_tr_{npy_size}x{npy_size}.npy")
    y = os.path.join(npy_dir, f"Y_tr_{npy_size}x{npy_size}.npy")
    if not (os.path.isfile(x) and os.path.isfile(y)):
        raise FileNotFoundError(
            f"Missing preprocessed arrays for '{dataset}':\n  {x}\n  {y}"
        )
    return np.load(x), np.load(y)


def _skin_slices(dataset, n):
    tr, vl, te = DATASET_SPLITS[dataset]
    if tr + vl + te != n:
        raise ValueError(
            f"{dataset}: npy has {n} samples but split sums to {tr + vl + te}."
        )
    return {"tr": slice(0, tr), "vl": slice(tr, tr + vl), "te": slice(tr + vl, n)}


def _build_skin(skin_root, dataset, split, n_points, img_size, augment, aug_level, npy_size):
    X, Y = _load_skin_npy(skin_root, dataset, npy_size)
    sel = _skin_slices(dataset, len(X))[split]
    return _array_dataset(X[sel], Y[sel], n_points, img_size, augment, aug_level)


# ----- BUSI (ultrasound, binary) ------------------------------------------

def _read_stem_list(path):
    stems = []
    with open(path) as f:
        for line in f:
            stem = line.strip()
            if stem:
                stems.append(stem)
    return stems


def _busi_pairs(root, stems):
    img_dir = os.path.join(root, "images")
    msk_dir = os.path.join(root, "masks")
    pairs = []
    for stem in stems:
        img = os.path.join(img_dir, f"{stem}.png")
        msk = os.path.join(msk_dir, f"{stem}.png")
        if os.path.isfile(img) and os.path.isfile(msk):
            pairs.append({"id": stem, "img": img, "mask": msk})
    if not pairs:
        raise FileNotFoundError(f"No BUSI pairs under {root} for {len(stems)} ids.")
    return pairs


def _busi_splits(data_root):
    root = os.path.join(data_root, "US", "busi")
    train = _read_stem_list(os.path.join(root, "train.txt"))
    val_all = _read_stem_list(os.path.join(root, "val.txt"))
    # Official files are train/val only; split val in file order into val + test.
    mid = len(val_all) // 2
    return {
        "root": root,
        "tr": _busi_pairs(root, train),
        "vl": _busi_pairs(root, val_all[:mid]),
        "te": _busi_pairs(root, val_all[mid:]),
    }


def _build_busi(data_root, split, n_points, img_size, augment, aug_level):
    parts = _busi_splits(data_root)
    return _file_dataset(parts[split], n_points, img_size, augment, aug_level)


# ----- polyp / Kvasir -----------------------------------------------------

def _stem(path):
    return os.path.splitext(os.path.basename(path))[0]


def _pairs_by_stem(img_dir, msk_dir):
    if not (os.path.isdir(img_dir) and os.path.isdir(msk_dir)):
        raise FileNotFoundError(f"Expected image/mask dirs:\n  {img_dir}\n  {msk_dir}")
    imgs = {
        _stem(f): os.path.join(img_dir, f)
        for f in os.listdir(img_dir)
        if f.lower().endswith(_IMG_EXTS)
    }
    msks = {
        _stem(f): os.path.join(msk_dir, f)
        for f in os.listdir(msk_dir)
        if f.lower().endswith(_IMG_EXTS)
    }
    pairs = []
    for stem in sorted(set(imgs) & set(msks)):
        pairs.append({"id": stem, "img": imgs[stem], "mask": msks[stem]})
    if not pairs:
        raise RuntimeError(f"No matching image/mask pairs in {img_dir}")
    return pairs


def _polyp_train_val(data_root, seed=42, val_ratio=0.2):
    root = os.path.join(data_root, "Polyp", "TrainDataset")
    pairs = _pairs_by_stem(os.path.join(root, "images"), os.path.join(root, "masks"))
    rng = random.Random(seed)
    rng.shuffle(pairs)
    n_val = int(round(len(pairs) * val_ratio))
    return pairs[n_val:], pairs[:n_val]


def _polyp_test(data_root, folder):
    root = os.path.join(data_root, "Polyp", "TestDataset", folder)
    return _pairs_by_stem(os.path.join(root, "images"), os.path.join(root, "masks"))


def _resolve_polyp_test(name):
    key = name.lower().replace("_", "").replace("-", "")
    aliases = {
        "kvasir": "Kvasir", "kvasirseg": "Kvasir",
        "cvc300": "CVC-300", "cvc": "CVC-300",
        "cvcclinicdb": "CVC-ClinicDB", "clinic": "CVC-ClinicDB", "clinicdb": "CVC-ClinicDB",
        "cvccolondb": "CVC-ColonDB", "colon": "CVC-ColonDB", "colondb": "CVC-ColonDB",
        "etislaribpolypdb": "ETIS-LaribPolypDB", "etis": "ETIS-LaribPolypDB",
        "test": "test", "polyp test": "test",
    }
    if name in POLYP_TEST_FOLDERS.values():
        return name
    if key in POLYP_TEST_FOLDERS:
        return POLYP_TEST_FOLDERS[key]
    if key in aliases:
        return aliases[key]
    raise ValueError(
        f"Unknown polyp test set '{name}'. "
        f"Use one of: {sorted(set(POLYP_TEST_FOLDERS) | set(POLYP_TEST_FOLDERS.values()))}"
    )


def _build_polyp(data_root, split, n_points, img_size, augment, aug_level, polyp_test):
    if split == "te":
        pairs = _polyp_test(data_root, _resolve_polyp_test(polyp_test))
    else:
        train, val = _polyp_train_val(data_root)
        pairs = train if split == "tr" else val
    return _file_dataset(pairs, n_points, img_size, augment, aug_level)


# ----- ACDC / Synapse (MRI, binarized) ------------------------------------

def _resize_hw(img, size):
    h, w = size
    if img.shape[:2] == (h, w):
        return img
    interp = cv2.INTER_NEAREST if img.ndim == 2 and img.dtype != np.float32 else cv2.INTER_LINEAR
    if img.ndim == 2 and (img.dtype == np.uint8 or np.issubdtype(img.dtype, np.integer)):
        interp = cv2.INTER_NEAREST
    return cv2.resize(img, (w, h), interpolation=interp)


def _stack_rgb_mask(img2d, mask2d, size):
    img2d = _resize_hw(np.asarray(img2d), size)
    mask2d = _resize_hw(np.asarray(mask2d), size)
    rgb = _as_uint8_rgb(img2d)
    if rgb.ndim == 2:
        rgb = np.stack([rgb, rgb, rgb], axis=0)
    else:
        rgb = np.moveaxis(rgb, -1, 0)
    mask = (np.asarray(mask2d) > 0).astype(np.uint8) * 255
    return rgb, mask


@lru_cache(maxsize=8)
def _acdc_arrays(data_root, split, size):
    """Load ACDC 2D slices (test volumes are expanded). Cached per process."""
    base = os.path.join(data_root, "ACDC_2D")
    folder = {"tr": "train", "vl": "valid", "te": "test"}[split]
    list_path = os.path.join(base, "list_ACDC", f"{'valid' if split == 'vl' else folder}.txt")
    names = _read_stem_list(list_path)
    images, masks = [], []
    h, w = size
    for name in names:
        path = os.path.join(base, folder, name.strip())
        if not path.endswith(".npz"):
            path = path + ".npz" if not os.path.isfile(path) else path
        data = np.load(path)
        img, lab = data["img"], data["label"]
        if img.ndim == 2:
            img, lab = img[None], lab[None]
        for i in range(img.shape[0]):
            if np.max(lab[i]) <= 0:
                continue
            rgb, mask = _stack_rgb_mask(img[i], lab[i], (h, w))
            images.append(rgb)
            masks.append(mask)
    if not images:
        raise RuntimeError(f"ACDC {split}: no foreground slices in {base}/{folder}")
    return np.stack(images), np.stack(masks)


def _build_acdc(data_root, split, n_points, img_size, augment, aug_level):
    X, Y = _acdc_arrays(data_root, split, tuple(img_size))
    return _array_dataset(X, Y, n_points, img_size, augment, aug_level)


@lru_cache(maxsize=8)
def _synapse_arrays(data_root, split, size):
    base = os.path.join(data_root, "Synapse")
    h, w = size
    images, masks = [], []
    if split == "tr":
        list_path = os.path.join(base, "lists", "lists_Synapse", "train.txt")
        names = _read_stem_list(list_path)
        for name in names:
            path = os.path.join(base, "train_npz", name + ".npz")
            data = np.load(path)
            img, lab = data["image"], data["label"]
            if np.max(lab) <= 0:
                continue
            rgb, mask = _stack_rgb_mask(img, lab, (h, w))
            images.append(rgb)
            masks.append(mask)
    else:
        # No official 2D val; hold out the last 2 test volumes as val.
        list_path = os.path.join(base, "lists", "lists_Synapse", "test_vol.txt")
        names = _read_stem_list(list_path)
        val_names, test_names = names[-2:], names[:-2]
        chosen = val_names if split == "vl" else test_names
        try:
            import h5py
        except ImportError as exc:
            raise ImportError("Synapse test volumes require h5py") from exc
        for name in chosen:
            path = os.path.join(base, "test_vol_h5", f"{name}.npy.h5")
            with h5py.File(path, "r") as handle:
                img, lab = handle["image"][:], handle["label"][:]
            for i in range(img.shape[0]):
                if np.max(lab[i]) <= 0:
                    continue
                rgb, mask = _stack_rgb_mask(img[i], lab[i], (h, w))
                images.append(rgb)
                masks.append(mask)
    if not images:
        raise RuntimeError(f"Synapse {split}: no foreground slices found.")
    return np.stack(images), np.stack(masks)


def _build_synapse(data_root, split, n_points, img_size, augment, aug_level):
    X, Y = _synapse_arrays(data_root, split, tuple(img_size))
    return _array_dataset(X, Y, n_points, img_size, augment, aug_level)


# ----- public API ---------------------------------------------------------

def build_contour_dataset(
    skin_root,
    dataset,
    split,
    n_points=200,
    img_size=(224, 224),
    augment=False,
    aug_level="strong",
    npy_size=224,
    data_root="/hdd/datasets",
    polyp_test="kvasir",
):
    """Return a contour dataset for one split of a named dataset."""
    dataset = dataset.lower()
    split = _norm_split(split)
    if dataset not in DATASET_NAMES:
        raise ValueError(f"Unknown dataset '{dataset}'. Known: {list(DATASET_NAMES)}")

    if dataset in DATASET_DIRS:
        return _build_skin(
            skin_root, dataset, split, n_points, img_size, augment, aug_level, npy_size,
        )
    if dataset == "busi":
        return _build_busi(data_root, split, n_points, img_size, augment, aug_level)
    if dataset in ("polyp", "kvasir"):
        test_name = "Kvasir" if dataset == "kvasir" else polyp_test
        return _build_polyp(
            data_root, split, n_points, img_size, augment, aug_level, test_name,
        )
    if dataset == "acdc":
        return _build_acdc(data_root, split, n_points, img_size, augment, aug_level)
    if dataset == "synapse":
        return _build_synapse(data_root, split, n_points, img_size, augment, aug_level)
    raise ValueError(f"Unhandled dataset '{dataset}'.")


def split_counts(skin_root, dataset, npy_size=224, data_root="/hdd/datasets",
                 polyp_test="kvasir"):
    """(train, val, test) sizes — handy for logging / sanity checks."""
    dataset = dataset.lower()
    counts = {}
    for split, key in (("tr", "tr"), ("vl", "vl"), ("te", "te")):
        ds = build_contour_dataset(
            skin_root, dataset, split, npy_size=npy_size,
            data_root=data_root, polyp_test=polyp_test, augment=False,
        )
        counts[key] = len(ds)
    return counts
