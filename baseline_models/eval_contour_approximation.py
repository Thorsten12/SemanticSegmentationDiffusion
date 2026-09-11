"""
Vergleicht, wie gut eine Kontur-Approximation mit N Punkten die
Ground-Truth-Maske reproduziert -- OHNE Modell, rein als Approximations-
Fehleranalyse (GT-Maske -> N Konturpunkte -> zurueck-rasterisierte Maske ->
Dice/IoU gegen Original-GT-Maske).

Nutzt bewusst dieselbe Punkt-Extraktion wie im Training (ArrayContourDataset
aus ph2_dataset.py -- curvature_adaptive_sampling ueber _contour_points()),
damit die Punkte konsistent mit dem sind, was das Modell tatsaechlich als
Zielrepraesentation sieht.

Ausgabe:
    - contour_approximation_results.json  (Rohdaten: pro Dataset x N -> mean/std Dice/IoU)
    - contour_approximation_table.md      (Markdown-Tabelle, Datasets x N)

RASTERISIERUNG (geklaert anhand des tatsaechlichen Quellcodes von
ph2_dataset.py, Dokument 4):
    ArrayContourDataset/PH2ContourDataset besitzen KEINE points_to_mask()
    o.ae. Methode -- die Klasse rechnet nur in die andere Richtung (Maske ->
    Punkte via cv2.findContours + curvature_adaptive_sampling/
    uniform_sampling in _contour_points()). Die Rueck-Rasterisierung fuer
    diesen Vergleich wird daher bewusst FEST mit cv2.fillPoly implementiert
    (kein Fallback-Suchmechanismus mehr noetig).

    WICHTIG: __getitem__ liefert points_tensor bereits normalisiert in
    [-1, 1] (x, y) -- siehe ph2_dataset.py:
        points[:, 0] = points[:, 0] / (W - 1); points[:, 1] analog
        points = points * 2.0 - 1.0
    D.h. die Rueck-Transformation nach Pixelkoordinaten ist die Inverse
    davon: px = (norm + 1) / 2 * (W - 1)  (und entsprechend fuer y, H).
"""

import json
from pathlib import Path

import cv2
import numpy as np

# -----------------------------
# Konfiguration
# -----------------------------
SKIN_ROOT = "/loctmp/sit28238/SemanticSegmentationDiffusion/data/datasets"
DATASETS = ["ph2", "isic2017", "isic2018", "ham10000", "busi", "polyp", "tn3k"]
N_POINTS_LIST = [16, 32, 64, 128, 256]
IMG_SIZE = (224, 224)
NPY_SIZE = 224

OUT_JSON = Path("table/contour_approximation_results.json")
OUT_MD = Path("table/contour_approximation_table.md")
OUT_JSON.parent.mkdir(parents=True, exist_ok=True)

# tn3k/tg3k sind analog busi/polyp preprocessed: tr/vl/te liegen als
# SEPARATE npy-Dateien vor (kein Index-Slicing auf einen gemeinsamen
# "tr"-Pool). Im Referenz-Loader (Dokument 3) MUSS daher gelten:
#   INDEXED_DATASETS      = {"ph2", "isic2017", "isic2018", "ham10000"}
#   DIRECT_SPLIT_DATASETS = {"busi", "polyp", "tn3k", "tg3k"}
# Dokument 3 hatte tn3k faelschlich in INDEXED_DATASETS UND fehlend in
# DATASET_SPLITS eingetragen (-> KeyError in _slices). Falls das im
# Hauptmodul noch nicht korrigiert ist, wird hier defensiv nachjustiert,
# damit dieses Script trotzdem korrekt laeuft.
EXTRA_DIRECT_SPLIT_DATASETS = {"tn3k"}


# -----------------------------
# Import des Referenz-Loaders
# -----------------------------
# Passe den Modulpfad an, falls die Datei aus Dokument 3 anders heisst /
# an anderer Stelle liegt (z.B. `from src.data.skin.contour_loader import ...`).
try:
    from data import (
        DATASET_DIRS,
        DIRECT_SPLIT_DATASETS,
        INDEXED_DATASETS,
        _load_npy,
        _slices,
    )
except ImportError as e:
    raise ImportError(
        "Konnte den Referenz-Loader (Dokument 3) nicht importieren. Bitte "
        "Dateinamen/Pfad in eval_contour_approximation.py anpassen (Zeile "
        "mit 'from contour_dataset_loader import ...')."
    ) from e

from data import ArrayContourDataset  # noqa: E402

# Defensive Korrektur, falls das Hauptmodul den tn3k/tg3k-Split-Typ noch
# nicht korrekt eingetragen hat (siehe Kommentar oben bei EXTRA_DIRECT_SPLIT_DATASETS).
INDEXED_DATASETS = set(INDEXED_DATASETS) - EXTRA_DIRECT_SPLIT_DATASETS
DIRECT_SPLIT_DATASETS = set(DIRECT_SPLIT_DATASETS) | EXTRA_DIRECT_SPLIT_DATASETS


# -----------------------------
# Alle Splits eines Datasets als EIN (X, Y) Pool laden (tr+vl+te zusammen)
# -----------------------------
def load_full_dataset(skin_root, dataset, npy_size=NPY_SIZE):
    dataset = dataset.lower()

    if dataset in INDEXED_DATASETS:
        # INDEXED_DATASETS: eine "tr"-Pool-Datei enthaelt bereits alles,
        # die Split-Grenzen sind nur logische Slices darauf. Da wir hier
        # ohnehin tr+vl+te zusammen wollen, reicht das Laden des Pools.
        X, Y = _load_npy(skin_root, dataset, "tr", npy_size)
        return X, Y

    # DIRECT_SPLIT_DATASETS (busi, polyp, tn3k, tg3k): tr/vl/te liegen als
    # separate Dateien vor -> alle drei laden und konkatenieren.
    Xs, Ys = [], []
    for split in ("tr", "vl", "te"):
        x, y = _load_npy(skin_root, dataset, split, npy_size)
        Xs.append(x)
        Ys.append(y)
    X = np.concatenate(Xs, axis=0)
    Y = np.concatenate(Ys, axis=0)
    return X, Y


# -----------------------------
# Punkte -> Maske (Rasterisierung)
# -----------------------------
def points_norm_to_mask(points_norm, size):
    """Rasterisiert normalisierte Konturpunkte (wie von ArrayContourDataset
    zurueckgegeben: [-1, 1] in (x, y)) zu einer binaeren Pixelmaske via
    cv2.fillPoly.

    Inverse der Normalisierung aus ph2_dataset.py (__getitem__):
        points[:, 0] = points[:, 0] / (W - 1); dann *2-1  (analog fuer y, H)
    D.h. Rueckweg: px = (norm + 1) / 2 * (W - 1)
    """
    h, w = size
    pts = np.asarray(points_norm, dtype=np.float64).copy()

    pts[:, 0] = (pts[:, 0] + 1.0) / 2.0 * (w - 1)
    pts[:, 1] = (pts[:, 1] + 1.0) / 2.0 * (h - 1)

    pts = np.clip(pts, [0, 0], [w - 1, h - 1])
    pts = pts.reshape(-1, 1, 2).astype(np.int32)

    mask = np.zeros((h, w), dtype=np.uint8)
    cv2.fillPoly(mask, [pts], 255)
    return mask


def get_gt_points_dataset(dataset_name, n_points, X, Y):
    """Instanziiert ArrayContourDataset fuer gegebenes n_points; die
    Rasterisierung erfolgt separat & fest via points_norm_to_mask()."""
    return ArrayContourDataset(
        X, Y,
        n_points=n_points,
        img_size=IMG_SIZE,
        augment=False,          # WICHTIG: keine Augmentierung fuer reine Approx.-Analyse
        aug_level="strong",
        adaptive_sampling=True,
    )


# -----------------------------
# Metriken
# -----------------------------
def dice_iou(pred_mask, gt_mask):
    """pred_mask, gt_mask: (H, W) Arrays, Werte 0/255 oder 0/1."""
    p = (pred_mask > 0).astype(np.uint8)
    g = (gt_mask > 0).astype(np.uint8)

    inter = np.logical_and(p, g).sum()
    union = np.logical_or(p, g).sum()
    p_sum = p.sum()
    g_sum = g.sum()

    if p_sum + g_sum == 0:
        # beide leer -> per Konvention perfektes Match
        return 1.0, 1.0

    dice = (2.0 * inter) / (p_sum + g_sum + 1e-8)
    iou = inter / (union + 1e-8)
    return float(dice), float(iou)


# -----------------------------
# Extraktion von Punkten + GT-Maske aus einem ArrayContourDataset-Sample
# -----------------------------
def extract_points_and_gt_mask(sample):
    """__getitem__ von PH2ContourDataset/ArrayContourDataset liefert exakt
    (img_tensor, points_tensor, mask_tensor) -- siehe ph2_dataset.py.
        points_tensor: [N, 2], normalisiert in [-1, 1] (x, y)
        mask_tensor:   [1, H, W], Werte in {0, 1} (float)
    Wir nutzen mask_tensor als GT direkt aus dem Dataset (statt Y[i] aus dem
    rohen npy), da dies exakt die Maske ist, aus der auch die Punkte
    extrahiert wurden (inkl. Resize/Binarisierung des Datasets selbst)."""
    _, points, gt_mask = sample
    points = points.detach().cpu().numpy() if hasattr(points, "detach") else np.asarray(points)
    gt_mask = gt_mask.detach().cpu().numpy() if hasattr(gt_mask, "detach") else np.asarray(gt_mask)
    if gt_mask.ndim == 3:
        gt_mask = gt_mask[0]
    gt_mask = (gt_mask > 0).astype(np.uint8) * 255
    return points, gt_mask


# -----------------------------
# Hauptevaluierung
# -----------------------------
def evaluate_dataset(dataset_name, skin_root=SKIN_ROOT, n_points_list=N_POINTS_LIST):
    print(f"\n=== {dataset_name} ===")
    X, Y = load_full_dataset(skin_root, dataset_name)
    n_samples = len(X)
    print(f"{dataset_name}: {n_samples} samples (tr+vl+te kombiniert)")

    results = {}

    for n_points in n_points_list:
        ds = get_gt_points_dataset(dataset_name, n_points, X, Y)

        dice_scores = []
        iou_scores = []

        for i in range(len(ds)):
            sample = ds[i]
            points, gt_mask = extract_points_and_gt_mask(sample)

            pred_mask = points_norm_to_mask(points, IMG_SIZE)

            dice, iou = dice_iou(pred_mask, gt_mask)
            dice_scores.append(dice)
            iou_scores.append(iou)

        results[n_points] = {
            "dice_mean": float(np.mean(dice_scores)),
            "dice_std": float(np.std(dice_scores)),
            "iou_mean": float(np.mean(iou_scores)),
            "iou_std": float(np.std(iou_scores)),
            "n_samples": n_samples,
        }

        print(
            f"  N={n_points:>3}: Dice={results[n_points]['dice_mean']:.4f} "
            f"(+/-{results[n_points]['dice_std']:.4f})  "
            f"IoU={results[n_points]['iou_mean']:.4f} "
            f"(+/-{results[n_points]['iou_std']:.4f})"
        )

    return results


def main():
    all_results = {}
    for dataset_name in DATASETS:
        try:
            all_results[dataset_name] = evaluate_dataset(dataset_name)
        except FileNotFoundError as e:
            print(f"[SKIP] {dataset_name}: {e}")
            continue

    OUT_JSON.write_text(json.dumps(all_results, indent=2, ensure_ascii=False))
    print(f"\nJSON gespeichert: {OUT_JSON}")

    write_markdown_table(all_results, OUT_MD)
    print(f"Markdown-Tabelle gespeichert: {OUT_MD}")


def write_markdown_table(all_results, out_path):
    lines = []
    lines.append("# Contour-Approximation: GT vs. rueck-rasterisierte N-Points-Maske\n")

    for metric in ("dice", "iou"):
        lines.append(f"## {metric.upper()}\n")
        header = "| Dataset | " + " | ".join(f"N={n}" for n in N_POINTS_LIST) + " |"
        sep = "|---" * (len(N_POINTS_LIST) + 1) + "|"
        lines.append(header)
        lines.append(sep)

        for dataset_name, res in all_results.items():
            row = [dataset_name]
            for n_points in N_POINTS_LIST:
                if n_points in res:
                    mean = res[n_points][f"{metric}_mean"]
                    std = res[n_points][f"{metric}_std"]
                    row.append(f"{mean:.3f} ± {std:.3f}")
                else:
                    row.append("-")
            lines.append("| " + " | ".join(row) + " |")
        lines.append("")

    out_path.write_text("\n".join(lines))


if __name__ == "__main__":
    main()