"""Verify the "large diffuse region vs. small sharp region" ambiguity hypothesis.

Motivation: visual inspection of worst-Dice test samples suggests the model
is often not confused by smoothness (that problem is largely solved), but by
a genuine multi-candidate ambiguity: the image contains two plausible lesion
regions -- a LARGE, low-contrast one and a SMALL, high-contrast one -- and the
prediction ends up as some compromise between the two rather than committing
to either.

This script makes that hypothesis falsifiable/quantifiable instead of relying
on eyeballing panels:

1. For each test image, find candidate regions via multi-threshold Otsu-style
   segmentation on a saliency/contrast map, then keep the largest two disjoint
   connected components as "candidate A" (larger) and "candidate B" (smaller,
   if it exists and is sufficiently disjoint from A).
2. Classify each GT mask as matching candidate A, candidate B, or neither
   (this tells us, per image, which candidate was actually correct).
3. For the model's prediction, compute:
   - IoU against candidate A and candidate B separately (not just against GT).
   - A "compromise score": how much predicted mass sits in the region that
     is in NEITHER candidate A nor candidate B nor their agreement region --
     i.e. area that's plausible under a literal blend/average of A and B but
     not fully committed to either. High compromise score + a genuine A/B
     ambiguity in the image is the direct fingerprint of the hypothesis.
4. Correlate: does per-sample Dice drop specifically on the subset of images
   that have a genuine, well-separated two-candidate ambiguity (vs. images
   with only one dominant candidate)? Is the "compromise score" itself
   predictive of low Dice, beyond what image_contrast alone predicts?

Usage:
    python -m x0_prediction_backup.dual_candidate_ambiguity \
        --ckpt x0_prediction_backup/runs/.../best.pth \
        --dataset isic2018 --split test \
        --out_dir x0_prediction_backup/analysis_out/dual_candidate
"""

import argparse
import os

import numpy as np
import torch
import cv2
from scipy import stats
from torch.utils.data import DataLoader
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from .config import Config
from .data import build_contour_dataset
from .sample import load_checkpoint
from .diffusion import GaussianDiffusion
from .utils import dice_score, iou_score, points_to_mask


# ---------------------------------------------------------------------------
# Candidate-region extraction
# ---------------------------------------------------------------------------

# --- Neue Funktion: Kalibrierungsplot für commitment-Schwellen ---
def compromise_case_panel(records, out_dir):
    """Speichert die tatsächlichen compromise-Fälle (pred_commitment=='compromise')
    als eigenes, übersichtliches Panel -- getrennt von worst/best, da diese
    Gruppe typischerweise zu klein ist, um in den generischen Top-12-Listen
    zuverlässig aufzutauchen.
    """
    dual_records = [r for r in records if r["has_dual"]]
    compromise_cases = [r for r in dual_records if r["pred_commitment"] == "compromise"]

    print(f"\n--- Compromise-Fälle: n={len(compromise_cases)} ---")
    if not compromise_cases:
        print("Keine compromise-Fälle gefunden.")
        return

    for i, r in enumerate(compromise_cases):
        print(f"  [{i}] dice={r['dice']:.4f} disputed_frac={r['disputed_frac']:.3f} "
              f"small_recall={r['small_recall']:.3f} large_excess_recall={r['large_excess_recall']:.3f} "
              f"gt_commitment={r['gt_commitment']} size_ratio={r['size_ratio']:.3f}")

    save_grid(compromise_cases, os.path.join(out_dir, "compromise_cases_only.png"),
              f"Alle {len(compromise_cases)} pred_commitment='compromise' Fälle", cols=4)
    print(f"\nGespeichert: compromise_cases_only.png in {out_dir}")


def filtered_correlation(records, out_dir, small_recall_thresh=0.85):
    """Kontinuierliche Korrelation large_excess_recall vs. Dice, aber nur
    innerhalb der Teilmenge mit small_recall > small_recall_thresh -- d.h.
    nur Fälle, in denen das Modell die kleine Kandidatenregion bereits
    (fast) vollständig abdeckt. Das isoliert die Frage "wie viel zusätzliche
    Fläche der großen Region nimmt das Modell noch mit, und schadet das?"
    von der Frage "deckt es die kleine Region überhaupt ab?" -- die beiden
    waren im ursprünglichen 4-Klassen-Schema vermischt.
    """
    dual_records = [r for r in records if r["has_dual"]]
    subset = [r for r in dual_records if r["small_recall"] > small_recall_thresh]

    print(f"\n--- Gefilterte Korrelation (small_recall > {small_recall_thresh}) ---")
    print(f"n={len(subset)} von {len(dual_records)} dual-candidate Fällen")

    if len(subset) < 10:
        print("Zu wenige Fälle für eine belastbare Korrelation.")
        return

    large_excess = np.array([r["large_excess_recall"] for r in subset])
    dice = np.array([r["dice"] for r in subset])

    rho, p = stats.spearmanr(large_excess, dice)
    print(f"Spearman correlation(large_excess_recall, Dice): rho={rho:.3f}, p={p:.4g}")
    if p < 0.05:
        direction = "negativ" if rho < 0 else "positiv"
        print(f"-> Signifikant, {direction}e Korrelation. "
              f"{'Mehr zusätzliche Fläche aus der großen Region senkt den Dice.' if rho < 0 else 'Unerwartetes Vorzeichen -- genauer prüfen.'}")
    else:
        print("-> Keine signifikante Korrelation. Selbst unter kontrollierten Bedingungen "
              "(small_recall bereits hoch) hängt Dice nicht davon ab, wie viel zusätzliche "
              "Fläche der großen Region das Modell mitnimmt.")

    # Zusätzlich: Scatterplot zur Visualisierung
    fig, ax = plt.subplots(figsize=(7, 6))
    ax.scatter(large_excess, dice, alpha=0.5, s=25, color="steelblue")
    ax.set_xlabel("large_excess_recall")
    ax.set_ylabel("Dice")
    ax.set_title(f"large_excess_recall vs. Dice\n(small_recall > {small_recall_thresh}, n={len(subset)}, "
                 f"rho={rho:.3f}, p={p:.3g})")
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "filtered_correlation_scatter.png"), dpi=140)
    plt.close(fig)
    print(f"Gespeichert: filtered_correlation_scatter.png in {out_dir}")

def calibration_plot(records, out_dir):
    """Scatter von small_recall vs. large_excess_recall über alle
    dual-candidate Records, eingefärbt nach gt_commitment, mit den
    aktuell verwendeten Schwellenlinien aus compromise_score() als
    Referenz. Zeigt, ob die Schwellen (0.85 / 0.60 / 0.35 / 0.15)
    tatsächlich an Clustern in den Daten liegen oder willkürlich
    durch die reale Verteilung schneiden.
    """
    os.makedirs(out_dir, exist_ok=True)
    dual_records = [r for r in records if r["has_dual"]]
    if not dual_records:
        print("Keine dual-candidate Records vorhanden -- Kalibrierung übersprungen.")
        return

    small_recall = np.array([r["small_recall"] for r in dual_records])
    large_excess = np.array([r["large_excess_recall"] for r in dual_records])
    gt_comm = np.array([r["gt_commitment"] for r in dual_records])
    pred_comm = np.array([r["pred_commitment"] for r in dual_records])
    dice = np.array([r["dice"] for r in dual_records])

    color_map = {"small": "tab:blue", "large": "tab:orange",
                 "compromise": "tab:red", "neither": "tab:gray"}

    # --- Plot 1: Scatter small_recall vs. large_excess_recall, Farbe = GT ---
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    ax = axes[0]
    for label, color in color_map.items():
        idx = gt_comm == label
        if idx.sum() == 0:
            continue
        ax.scatter(small_recall[idx], large_excess[idx], c=color, label=f"gt={label} (n={idx.sum()})",
                   alpha=0.6, s=25)
    # aktuelle Schwellen als Referenzlinien
    ax.axvline(0.85, color="k", linestyle="--", linewidth=0.8)
    ax.axvline(0.35, color="k", linestyle=":", linewidth=0.8)
    ax.axhline(0.60, color="k", linestyle="--", linewidth=0.8)
    ax.axhline(0.25, color="k", linestyle=":", linewidth=0.8)
    ax.axhline(0.15, color="k", linestyle=":", linewidth=0.8)
    ax.set_xlabel("small_recall")
    ax.set_ylabel("large_excess_recall")
    ax.set_title("Eingefärbt nach GT-commitment\n(gestrichelt/gepunktet = aktuelle Schwellen)")
    ax.legend(fontsize=8)

    ax2 = axes[1]
    for label, color in color_map.items():
        idx = pred_comm == label
        if idx.sum() == 0:
            continue
        ax2.scatter(small_recall[idx], large_excess[idx], c=color, label=f"pred={label} (n={idx.sum()})",
                    alpha=0.6, s=25)
    ax2.axvline(0.85, color="k", linestyle="--", linewidth=0.8)
    ax2.axvline(0.35, color="k", linestyle=":", linewidth=0.8)
    ax2.axhline(0.60, color="k", linestyle="--", linewidth=0.8)
    ax2.axhline(0.25, color="k", linestyle=":", linewidth=0.8)
    ax2.axhline(0.15, color="k", linestyle=":", linewidth=0.8)
    ax2.set_xlabel("small_recall")
    ax2.set_ylabel("large_excess_recall")
    ax2.set_title("Eingefärbt nach Pred-commitment\n(gestrichelt/gepunktet = aktuelle Schwellen)")
    ax2.legend(fontsize=8)

    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "calibration_scatter.png"), dpi=140)
    plt.close(fig)

    # --- Plot 2: Marginal-Histogramme beider Größen ---
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    axes[0].hist(small_recall, bins=30, color="steelblue", alpha=0.8)
    axes[0].axvline(0.85, color="k", linestyle="--", label="high thresh (0.85)")
    axes[0].axvline(0.35, color="k", linestyle=":", label="low thresh (0.35)")
    axes[0].set_title("Verteilung: small_recall")
    axes[0].set_xlabel("small_recall")
    axes[0].legend(fontsize=8)

    axes[1].hist(large_excess, bins=30, color="darkorange", alpha=0.8)
    axes[1].axvline(0.60, color="k", linestyle="--", label="high thresh (0.60)")
    axes[1].axvline(0.25, color="k", linestyle=":", label="mid thresh (0.25)")
    axes[1].axvline(0.15, color="k", linestyle="-.", label="low thresh (0.15)")
    axes[1].set_title("Verteilung: large_excess_recall")
    axes[1].set_xlabel("large_excess_recall")
    axes[1].legend(fontsize=8)

    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "calibration_histograms.png"), dpi=140)
    plt.close(fig)

    # --- Konsolen-Ausgabe: wie viele Punkte liegen in "Grauzonen" ---
    # d.h. Bereiche, in denen kleine Schwellenverschiebungen das Label kippen
    near_high_sr = np.abs(small_recall - 0.85) < 0.05
    near_low_sr = np.abs(small_recall - 0.35) < 0.05
    near_high_ler = np.abs(large_excess - 0.60) < 0.05
    near_mid_ler = np.abs(large_excess - 0.25) < 0.05
    near_low_ler = np.abs(large_excess - 0.15) < 0.05

    print(f"\n--- Kalibrierungs-Diagnose (n={len(dual_records)} dual-candidate) ---")
    print(f"Nahe small_recall=0.85 (±0.05): {near_high_sr.sum()} Fälle")
    print(f"Nahe small_recall=0.35 (±0.05): {near_low_sr.sum()} Fälle")
    print(f"Nahe large_excess=0.60 (±0.05): {near_high_ler.sum()} Fälle")
    print(f"Nahe large_excess=0.25 (±0.05): {near_mid_ler.sum()} Fälle")
    print(f"Nahe large_excess=0.15 (±0.05): {near_low_ler.sum()} Fälle")

    # Dice-Vergleich: liegen "neither"-Fälle systematisch nah an Schwellen?
    neither_idx = pred_comm == "neither"
    if neither_idx.sum() > 0:
        print(f"\n'neither'-Fälle (n={neither_idx.sum()}): "
              f"mean small_recall={small_recall[neither_idx].mean():.3f}, "
              f"mean large_excess_recall={large_excess[neither_idx].mean():.3f}, "
              f"mean dice={dice[neither_idx].mean():.4f}")
        print("(Falls diese Werte nah an den Schwellen liegen -> Grenzfälle, die durch\n"
              " leichte Schwellenanpassung neu klassifiziert würden. Falls sie weit von\n"
              " ALLEN Schwellen entfernt liegen -> 'neither' ist ein echtes viertes\n"
              " Verhalten, keine Kalibrierungslücke.)")

    print(f"\nGespeichert: calibration_scatter.png, calibration_histograms.png in {out_dir}")

def _saliency_map(image_u8: np.ndarray) -> np.ndarray:
    """Simple, dependency-light saliency proxy: inverted, contrast-stretched
    grayscale intensity (lesions are typically darker than surrounding skin
    in dermoscopy images -- adjust the sign if this doesn't hold for your
    modality). Blurred lightly to suppress hair/noise before thresholding.
    """
    gray = cv2.cvtColor(image_u8, cv2.COLOR_RGB2GRAY)
    gray = cv2.GaussianBlur(gray, (5, 5), 0)
    inv = 255 - gray
    inv = cv2.normalize(inv, None, 0, 255, cv2.NORM_MINMAX)
    return inv.astype(np.uint8)


def _threshold_candidates(saliency: np.ndarray, low_q: float = 0.55, high_q: float = 0.85):
    """Two-threshold candidate extraction.

    low_q (permissive) threshold -> captures the LARGE, low-contrast
        candidate region (anything moderately more salient than background).
    high_q (strict) threshold -> captures the SMALL, high-contrast candidate
        region (only the most salient core).

    Returns (mask_large, mask_small) as uint8 {0,1} masks, each restricted to
    their single largest connected component (so noise specks don't count).
    """
    def _largest_component(mask):
        mask = mask.astype(np.uint8)
        n, labels, stats_, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
        if n <= 1:
            return np.zeros_like(mask)
        areas = stats_[1:, cv2.CC_STAT_AREA]
        largest = 1 + int(np.argmax(areas))
        return (labels == largest).astype(np.uint8)

    low_thresh = np.quantile(saliency, low_q)
    high_thresh = np.quantile(saliency, high_q)

    mask_large = _largest_component(saliency >= low_thresh)
    mask_small = _largest_component(saliency >= high_thresh)
    return mask_large, mask_small


def _iou(mask_a: np.ndarray, mask_b: np.ndarray) -> float:
    inter = np.logical_and(mask_a, mask_b).sum()
    union = np.logical_or(mask_a, mask_b).sum()
    return float(inter) / float(union) if union > 0 else 0.0


def find_dual_candidates(image_u8: np.ndarray, min_area_frac: float = 0.01,
                          max_containment: float = 0.90):
    """Identify whether an image has a genuine large-vs-small candidate
    ambiguity, as opposed to one candidate simply being a noisy subset of
    the other (which would not represent a real ambiguity).

    Returns a dict:
      has_dual: bool -- True if both candidates exist, are each large
          enough, and the small one is NOT almost entirely a trivial
          erosion of the large one at a near-identical area (i.e. they
          disagree about SIZE in a way that could plausibly correspond to
          two different real lesion-boundary interpretations, not just
          threshold jitter on the same edge).
      mask_large, mask_small: the two candidate masks.
      size_ratio: area(small) / area(large), for reporting.
    """
    h, w = image_u8.shape[:2]
    total_px = h * w
    saliency = _saliency_map(image_u8)
    mask_large, mask_small = _threshold_candidates(saliency)

    area_large = mask_large.sum()
    area_small = mask_small.sum()

    if area_large < min_area_frac * total_px or area_small < min_area_frac * total_px:
        return {"has_dual": False, "mask_large": mask_large, "mask_small": mask_small,
                "size_ratio": np.nan}

    size_ratio = area_small / max(area_large, 1)
    # Containment: what fraction of the small candidate's area lies inside
    # the large candidate. If this is very high AND size_ratio is close to
    # 1, the two thresholds just found the same region twice (not a real
    # ambiguity). We want size_ratio meaningfully < 1 (small is genuinely
    # smaller) while containment is high (they're nested, i.e. plausibly
    # "same lesion, disagreement about extent" rather than two unrelated
    # blobs, which would be a different failure mode entirely).
    containment = np.logical_and(mask_large, mask_small).sum() / max(area_small, 1)

    has_dual = (
        containment >= 0.7          # small sits inside large (same lesion, disputed extent)
        and size_ratio <= max_containment  # meaningfully smaller, not just a rounding difference
        and size_ratio >= 0.15      # not so tiny that it's just a bright artifact fleck
    )

    return {
        "has_dual": bool(has_dual),
        "mask_large": mask_large,
        "mask_small": mask_small,
        "size_ratio": float(size_ratio),
        "containment": float(containment),
    }


# ---------------------------------------------------------------------------
# Compromise score: does the prediction sit "between" the two candidates?
# ---------------------------------------------------------------------------

def compromise_score(pred_mask: np.ndarray, mask_large: np.ndarray, mask_small: np.ndarray) -> float:
    """Quantifies how much the prediction looks like a blend of the two
    candidates rather than a commitment to either.

    Defined as: fraction of predicted foreground area that lies in the
    "disputed ring" (inside mask_large but outside mask_small) -- i.e. area
    the model included that only the LARGE candidate would justify, while
    simultaneously not fully covering mask_small (checked separately via
    small_recall). A high disputed-ring fraction combined with incomplete
    small_recall is the direct signature of straddling both candidates
    instead of picking one.

    Returns a dict with:
      disputed_frac: fraction of pred area in the disputed ring (large \ small)
      small_recall: fraction of mask_small covered by pred
      large_excess_recall: fraction of (large \ small) covered by pred
      commitment: "small" | "large" | "compromise" | "neither" heuristic label
    """
    disputed_ring = np.logical_and(mask_large, np.logical_not(mask_small))
    pred_area = pred_mask.sum()

    disputed_frac = (
        np.logical_and(pred_mask, disputed_ring).sum() / max(pred_area, 1)
    )
    small_recall = (
        np.logical_and(pred_mask, mask_small).sum() / max(mask_small.sum(), 1)
    )
    large_excess_recall = (
        np.logical_and(pred_mask, disputed_ring).sum() / max(disputed_ring.sum(), 1)
    )

    # Heuristic commitment label, purely descriptive (used for grouping in
    # the report, not as a hard classifier).
    if small_recall >= 0.85 and large_excess_recall <= 0.25:
        commitment = "small"
    elif small_recall >= 0.85 and large_excess_recall >= 0.60:
        commitment = "large"
    elif 0.35 <= small_recall <= 0.85 and 0.15 <= large_excess_recall <= 0.60:
        commitment = "compromise"
    else:
        commitment = "neither"

    return {
        "disputed_frac": float(disputed_frac),
        "small_recall": float(small_recall),
        "large_excess_recall": float(large_excess_recall),
        "commitment": commitment,
    }


def gt_commitment(gt_mask: np.ndarray, mask_large: np.ndarray, mask_small: np.ndarray) -> str:
    """Which candidate does GT actually match? Same recall-based logic as
    compromise_score, applied to GT instead of the prediction -- tells us
    the GROUND-TRUTH answer to "was the small or large region correct here".
    """
    res = compromise_score(gt_mask, mask_large, mask_small)
    return res["commitment"]


# ---------------------------------------------------------------------------
# Data collection
# ---------------------------------------------------------------------------

def denorm_to_uint8(image_chw: np.ndarray) -> np.ndarray:
    img = np.transpose(image_chw, (1, 2, 0))
    img = (img + 1.0) / 2.0
    img = np.clip(img, 0.0, 1.0)
    if img.shape[-1] == 1:
        img = np.repeat(img, 3, axis=-1)
    return (img * 255).astype(np.uint8)


@torch.no_grad()
def collect_records(ckpt_path, dataset, split, device, max_samples=None):
    cfg = Config()
    device = torch.device(device)

    encoder, denoiser, snapper, proposal_head = load_checkpoint(ckpt_path, cfg, device)
    diffusion = GaussianDiffusion(cfg.timesteps, cfg.beta_start, cfg.beta_end, device=device)
    encoder.eval(); denoiser.eval(); snapper.eval()
    if proposal_head is not None:
        proposal_head.eval()

    split_key = "te" if split == "test" else "vl"
    ds = build_contour_dataset(cfg.skin_root, dataset, split_key, cfg.n_points,
                               cfg.img_size, augment=False, npy_size=cfg.npy_size)
    loader = DataLoader(ds, batch_size=8, shuffle=False)

    records = []
    snap_t_threshold = int(round(getattr(cfg, "snap_t_threshold_frac", 0.15) * cfg.timesteps))
    n_seen = 0

    for images, gt_points, gt_masks in loader:
        if max_samples is not None and n_seen >= max_samples:
            break
        images = images.to(device)
        raw = encoder.extract(images)
        proposal = proposal_head(raw) if proposal_head is not None else None
        cond_fn = lambda t_b: encoder.fuse(raw, t_b)
        shape = (images.shape[0], cfg.n_points, 2)

        pred_points = diffusion.ddim_sample(
            denoiser, cond_fn, shape,
            proposal=proposal, proposal_target=cfg.proposal_target,
            ddim_steps=cfg.ddim_steps, guidance_scale=cfg.guidance_scale, clamp=1.0,
            snapper=snapper if cfg.snap_mode in ("loop", "both") else None,
            snap_maps=raw if cfg.snap_mode in ("loop", "both") else None,
            snap_image=images if cfg.snap_mode in ("loop", "both") else None,
            snap_t_threshold=snap_t_threshold if cfg.snap_mode in ("loop", "both") else None,
            snap_every=getattr(cfg, "snap_every", 1),
            collect_every=0,
        )
        if cfg.snap_mode in ("post", "both"):
            pred_points = snapper(pred_points, raw, image=images, hard=True)

        images_cpu = images.cpu().numpy()
        for i in range(images.shape[0]):
            gt_mask = gt_masks[i].squeeze().cpu().numpy().astype(np.uint8)
            pred_mask = points_to_mask(pred_points[i], cfg.img_size)
            image_u8 = denorm_to_uint8(images_cpu[i])

            d = dice_score(pred_mask, gt_mask)
            j = iou_score(pred_mask, gt_mask)

            dual = find_dual_candidates(image_u8)
            rec = {
                "dice": d, "iou": j,
                "has_dual": dual["has_dual"],
                "size_ratio": dual["size_ratio"],
            }
            if dual["has_dual"]:
                comp = compromise_score(pred_mask, dual["mask_large"], dual["mask_small"])
                gt_comm = gt_commitment(gt_mask, dual["mask_large"], dual["mask_small"])
                rec.update({
                    "disputed_frac": comp["disputed_frac"],
                    "small_recall": comp["small_recall"],
                    "large_excess_recall": comp["large_excess_recall"],
                    "pred_commitment": comp["commitment"],
                    "gt_commitment": gt_comm,
                })
                rec["image_u8"] = image_u8
                rec["gt_mask"] = gt_mask
                rec["pred_mask"] = pred_mask
                rec["mask_large"] = dual["mask_large"]
                rec["mask_small"] = dual["mask_small"]
            records.append(rec)
            n_seen += 1
            if max_samples is not None and n_seen >= max_samples:
                break

    return records


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------

def make_panel(rec):
    """image | GT with both candidate outlines | pred with both candidate outlines."""
    img = rec["image_u8"]
    h, w = rec["gt_mask"].shape
    panel = np.zeros((h, w * 3 + 20, 3), dtype=np.uint8)
    panel[:, :w] = img

    def _contour(mask):
        cs, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        return max(cs, key=cv2.contourArea) if cs else None

    gt_panel = img.copy()
    c_large = _contour(rec["mask_large"])
    c_small = _contour(rec["mask_small"])
    if c_large is not None:
        cv2.drawContours(gt_panel, [c_large], -1, (255, 165, 0), 1)   # orange = large candidate
    if c_small is not None:
        cv2.drawContours(gt_panel, [c_small], -1, (0, 200, 255), 1)   # cyan = small candidate
    c_gt = _contour(rec["gt_mask"])
    if c_gt is not None:
        cv2.drawContours(gt_panel, [c_gt], -1, (0, 255, 0), 2)        # green = GT
    panel[:, w + 10:w * 2 + 10] = gt_panel

    pred_panel = img.copy()
    if c_large is not None:
        cv2.drawContours(pred_panel, [c_large], -1, (255, 165, 0), 1)
    if c_small is not None:
        cv2.drawContours(pred_panel, [c_small], -1, (0, 200, 255), 1)
    c_pred = _contour(rec["pred_mask"])
    if c_pred is not None:
        cv2.drawContours(pred_panel, [c_pred], -1, (255, 0, 255), 2)  # magenta = prediction
    panel[:, w * 2 + 20:] = pred_panel

    return panel


def save_grid(records, out_path, title, cols=4):
    n = len(records)
    if n == 0:
        return
    rows = int(np.ceil(n / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 5, rows * 2.4))
    axes = np.array(axes).reshape(-1)

    for ax_idx, rec in enumerate(records):
        panel = make_panel(rec)
        ax = axes[ax_idx]
        ax.imshow(panel)
        ax.set_title(
            f"dice={rec['dice']:.2f} disputed={rec['disputed_frac']:.2f}\n"
            f"pred={rec['pred_commitment']} gt={rec['gt_commitment']} "
            f"size_ratio={rec['size_ratio']:.2f}",
            fontsize=8,
        )
        ax.axis("off")

    for ax_idx in range(n, len(axes)):
        axes[ax_idx].axis("off")

    fig.suptitle(title + "\norange=large candidate, cyan=small candidate, green=GT, magenta=pred",
                 fontsize=11)
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    fig.savefig(out_path, dpi=140)
    plt.close(fig)


def report(records, out_dir):
    os.makedirs(out_dir, exist_ok=True)
    dice = np.array([r["dice"] for r in records])
    has_dual = np.array([r["has_dual"] for r in records])

    n = len(records)
    n_dual = int(has_dual.sum())
    print(f"N = {n} total, {n_dual} ({100*n_dual/n:.1f}%) flagged as having a genuine "
          f"large-vs-small candidate ambiguity.\n")

    dice_dual = dice[has_dual]
    dice_no_dual = dice[~has_dual]
    print(f"Mean Dice | dual-candidate images:     {dice_dual.mean():.4f} (n={len(dice_dual)})")
    print(f"Mean Dice | single-candidate images:   {dice_no_dual.mean():.4f} (n={len(dice_no_dual)})")
    if len(dice_dual) > 5 and len(dice_no_dual) > 5:
        t_stat, p_val = stats.ttest_ind(dice_dual, dice_no_dual, equal_var=False)
        print(f"Welch's t-test (dual vs. non-dual Dice): t={t_stat:.3f}, p={p_val:.4g}")
        print("(Low p, dual-candidate mean lower => the ambiguity subgroup is a real, "
              "statistically distinct source of degraded performance, not noise.)\n")

    dual_records = [r for r in records if r["has_dual"]]
    if dual_records:
        disputed = np.array([r["disputed_frac"] for r in dual_records])
        dual_dice = np.array([r["dice"] for r in dual_records])
        rho, p = stats.spearmanr(disputed, dual_dice)
        print(f"Within dual-candidate subset: correlation(disputed_frac, Dice) = "
              f"rho={rho:.3f}, p={p:.4g}")
        print("(Negative rho => the more the prediction straddles both candidates "
              "instead of committing, the worse the Dice -- direct evidence for the "
              "'compromise' failure mode, within images that actually have the "
              "ambiguity.)\n")

        commitments = [r["pred_commitment"] for r in dual_records]
        gt_commitments = [r["gt_commitment"] for r in dual_records]
        print("Prediction commitment breakdown (within dual-candidate images):")
        for label in ("small", "large", "compromise", "neither"):
            idx = [i for i, c in enumerate(commitments) if c == label]
            if idx:
                mean_dice = dual_dice[idx].mean()
                print(f"  pred_commitment={label:11s} n={len(idx):3d}  mean_dice={mean_dice:.4f}")

        print("\nGT commitment breakdown (which candidate was actually correct):")
        for label in ("small", "large", "compromise", "neither"):
            idx = [i for i, c in enumerate(gt_commitments) if c == label]
            if idx:
                print(f"  gt_commitment={label:11s}  n={len(idx):3d} "
                      f"({100*len(idx)/len(gt_commitments):.0f}%)")

        # Confusion-style cross-tab: does the model commit to the WRONG candidate?
        print("\nCross-tab: pred_commitment vs. gt_commitment (within dual-candidate images):")
        labels = ("small", "large", "compromise", "neither")
        header = "gt \\ pred".ljust(12) + "".join(l.ljust(12) for l in labels)
        print(header)
        for gt_l in labels:
            row = f"{gt_l:12s}"
            for pred_l in labels:
                cnt = sum(1 for c, g in zip(commitments, gt_commitments) if c == pred_l and g == gt_l)
                row += f"{cnt:<12d}"
            print(row)
        print(
            "\n(Read this as: rows = what GT actually needed, columns = what the model\n"
            "produced. Mass on the diagonal = correct commitment. Off-diagonal small/large\n"
            "cells = the model committed confidently to the WRONG candidate. High\n"
            "'compromise' column mass = the model hedges instead of committing, "
            "regardless\nof which candidate was correct.)"
        )

        worst = sorted(dual_records, key=lambda r: r["dice"])[:min(12, len(dual_records))]
        best = sorted(dual_records, key=lambda r: -r["dice"])[:min(12, len(dual_records))]
        save_grid(worst, os.path.join(out_dir, "dual_candidate_worst.png"),
                  "Worst-Dice dual-candidate samples")
        save_grid(best, os.path.join(out_dir, "dual_candidate_best.png"),
                  "Best-Dice dual-candidate samples")
        print(f"\nSaved dual_candidate_worst.png / dual_candidate_best.png to {out_dir}")
    else:
        print("No dual-candidate images were flagged -- consider loosening "
              "find_dual_candidates()'s thresholds (min_area_frac, containment, "
              "size_ratio bounds) if you visually know such cases exist in this split.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ckpt", type=str, required=True)
    parser.add_argument("--dataset", type=str, default="isic2018")
    parser.add_argument("--split", type=str, default="test")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--max_samples", type=int, default=None)
    parser.add_argument("--out_dir", type=str, default="analysis_out/dual_candidate")
    parser.add_argument("--calibrate", action="store_true",
                         help="Zusätzlich Kalibrierungsplots für die commitment-Schwellen erzeugen.")
    parser.add_argument("--check_compromise", action="store_true",
                         help="Zusätzlich die tatsächlichen compromise-Fälle isoliert anzeigen "
                              "und die gefilterte Korrelation large_excess_recall vs. Dice berechnen.")
    args = parser.parse_args()

    records = collect_records(args.ckpt, args.dataset, args.split, args.device,
                               max_samples=args.max_samples)
    report(records, args.out_dir)
    if args.calibrate:
        calibration_plot(records, args.out_dir)
    if args.check_compromise:
        compromise_case_panel(records, args.out_dir)
        filtered_correlation(records, args.out_dir)