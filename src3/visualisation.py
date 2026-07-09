"""
Visualisierung für das Contour-Diffusion-Modell.

Zeigt für die ersten `n_samples` Beispiele aus einem Dataset 3 Plots pro Sample:
    1) Bild + Ground-Truth-Maske (aus den echten Contour-Punkten gerastert)
    2) Bild + Predicted-Maske (aus den vom Diffusionsmodell gesampelten Punkten gerastert)
    3) Differenz/Überlappung beider Masken (TP / FN / FP eingefärbt)

Zusätzlich wird pro Sample Dice und IoU berechnet und im Plot-Titel angezeigt.

"""

import numpy as np
import cv2
import torch
import matplotlib.pyplot as plt


def denorm_points(points_norm: np.ndarray, img_size=(224, 224)) -> np.ndarray:
    """
    Wandelt normierte Contour-Punkte (x, y in [-1, 1]) zurück in Pixelkoordinaten.

    points_norm: (N, 2) Array, Reihenfolge (x, y), Wertebereich [-1, 1]
    img_size: (H, W)
    returns: (N, 2) Array in Pixelkoordinaten [0, W] bzw. [0, H]
    """
    h, w = img_size
    pts = points_norm.copy()
    pts[:, 0] = (pts[:, 0] + 1) / 2 * w
    pts[:, 1] = (pts[:, 1] + 1) / 2 * h
    return pts


def points_to_mask(points_norm: np.ndarray, img_size=(224, 224)) -> np.ndarray:
    """
    Rastert normierte Contour-Punkte (x, y in [-1, 1]) zu einer binären Maske.

    points_norm: (N, 2) Array, Reihenfolge (x, y), Wertebereich [-1, 1]
    img_size: (H, W)
    returns: (H, W) uint8 Maske mit Werten 0/1
    """
    h, w = img_size

    pts = denorm_points(points_norm, img_size)
    pts_int = pts.round().astype(np.int32).reshape(-1, 1, 2)  # (N, 1, 2), von cv2.fillPoly erwartet

    mask = np.zeros((h, w), dtype=np.uint8)
    cv2.fillPoly(mask, [pts_int], color=1)
    return mask


def denorm_image(img_tensor: torch.Tensor) -> np.ndarray:
    """
    img_tensor: (3, H, W), normalisiert mit mean=0.5, std=0.5 -> Wertebereich [-1, 1]
    returns: (H, W, 3) uint8 Array im Bereich [0, 255], fuer plt.imshow
    """
    img = img_tensor.detach().cpu().numpy()
    img = np.transpose(img, (1, 2, 0))       # (H, W, 3)
    img = img * 0.5 + 0.5                    # zurück auf [0, 1]
    img = np.clip(img, 0.0, 1.0)
    img = (img * 255).astype(np.uint8)
    return img


def dice_iou(pred_mask: np.ndarray, gt_mask: np.ndarray, eps: float = 1e-7):
    """
    pred_mask, gt_mask: (H, W) binaere Masken (0/1)
    returns: (dice, iou) als floats
    """
    pred = pred_mask.astype(bool)
    gt = gt_mask.astype(bool)

    intersection = np.logical_and(pred, gt).sum()
    union = np.logical_or(pred, gt).sum()
    pred_sum = pred.sum()
    gt_sum = gt.sum()

    dice = (2.0 * intersection + eps) / (pred_sum + gt_sum + eps)
    iou = (intersection + eps) / (union + eps)
    return float(dice), float(iou)


@torch.no_grad()
def _predict_mask_for_sample(denoiser, encoder, diffusion, img_tensor, device, n_points, img_size, guidance):
    """
    Interner Helper: sampelt Contour-Punkte für ein einzelnes Bild und rastert sie zu einer Maske.
    Wird sowohl von visualize_predictions als auch von evaluate_dice genutzt, damit beide
    exakt dieselbe Sampling-/Rasterisierungs-Logik verwenden.
    """
    images = img_tensor.unsqueeze(0).to(device)  # (1, 3, H, W)

    predicted_points = diffusion.sample(
        denoiser=denoiser,
        encoder=encoder,
        images=images,
        shape=(1, n_points, 2),
        device=device,
        cfg_scale=guidance,
    )  # (1, n_points, 2)

    pred_points_np = predicted_points.squeeze(0).detach().cpu().numpy()
    pred_mask = points_to_mask(pred_points_np, img_size=img_size)
    return pred_points_np, pred_mask


@torch.no_grad()
def evaluate_dice(
    denoiser,
    encoder,
    diffusion,
    dataset,
    device,
    n_points: int = 200,
    img_size=(224, 224),
    guidance: float = 2.5,
):
    """
    Berechnet den mittleren Dice-Score über ALLE Samples in `dataset`
    (nicht nur eine kleine Teilmenge wie bei visualize_predictions).

    Nutzt exakt dieselbe Sampling-/Rasterisierungs-Logik wie visualize_predictions,
    damit der hier berechnete Dice-Score mit dem in den Plots angezeigten vergleichbar ist.

    returns: (mean_dice, mean_iou) als floats
    """
    was_training = denoiser.training
    denoiser.eval()
    encoder.eval()

    dice_scores = []
    iou_scores = []

    for i in range(len(dataset)):
        img_tensor, gt_points_tensor, _ = dataset[i]
        gt_points_np = gt_points_tensor.numpy()
        gt_mask = points_to_mask(gt_points_np, img_size=img_size)

        _, pred_mask = _predict_mask_for_sample(
            denoiser=denoiser,
            encoder=encoder,
            diffusion=diffusion,
            img_tensor=img_tensor,
            device=device,
            n_points=n_points,
            img_size=img_size,
            guidance=guidance,
        )

        dice, iou = dice_iou(pred_mask, gt_mask)
        dice_scores.append(dice)
        iou_scores.append(iou)

    if was_training:
        denoiser.train()

    mean_dice = float(np.mean(dice_scores)) if dice_scores else 0.0
    mean_iou = float(np.mean(iou_scores)) if iou_scores else 0.0
    return mean_dice, mean_iou


@torch.no_grad()
def visualize_predictions(
    denoiser,
    encoder,
    diffusion,
    dataset,
    device,
    n_points: int = 200,
    img_size=(224, 224),
    n_samples: int = 4,
    out_path: str = "visualisation.png",
    guidance:float = 2.5
):
    """
    Erzeugt eine (n_samples x 3) Plot-Grid und speichert sie unter out_path.
    Nutzt die ersten n_samples Eintraege aus `dataset`.
    """
    denoiser.eval()
    encoder.eval()

    fig, axes = plt.subplots(n_samples, 3, figsize=(12, 4 * n_samples))
    if n_samples == 1:
        axes = axes[None, :]  # damit axes[i, j] auch bei n_samples=1 funktioniert

    for i in range(n_samples):
        img_tensor, gt_points_tensor, _ = dataset[i]

        pred_points_np, pred_mask = _predict_mask_for_sample(
            denoiser=denoiser,
            encoder=encoder,
            diffusion=diffusion,
            img_tensor=img_tensor,
            device=device,
            n_points=n_points,
            img_size=img_size,
            guidance=guidance,
        )

        gt_points_np = gt_points_tensor.numpy()
        gt_mask = points_to_mask(gt_points_np, img_size=img_size)

        print("pred points min/max:", pred_points_np.min(), pred_points_np.max())
        print("pred mask sum:", pred_mask.sum(), "gt mask sum:", gt_mask.sum())

        dice, iou = dice_iou(pred_mask, gt_mask)
        print(f"Sample {i} — Prediction\nDice: {dice:.3f} | IoU: {iou:.3f}")
        img_np = denorm_image(img_tensor)

        # Pixelkoordinaten für Scatter-Overlays
        gt_px = denorm_points(gt_points_np, img_size)
        pred_px = denorm_points(pred_points_np, img_size)

        # --- Plot 1: Bild + Ground Truth ---
        ax = axes[i, 0]
        ax.imshow(img_np)
        ax.imshow(gt_mask, alpha=0.4, cmap="Greens")
        ax.scatter(gt_px[:, 0], gt_px[:, 1], s=4, c="lime", alpha=0.6)
        ax.set_title(f"Sample {i} — Ground Truth")
        ax.axis("off")

        # --- Plot 2: Bild + Prediction ---
        ax = axes[i, 1]
        ax.imshow(img_np)
        ax.scatter(pred_px[:, 0], pred_px[:, 1], s=4, c="yellow", alpha=0.6)
        ax.set_title(f"Sample {i} — Prediction\nDice: {dice:.3f} | IoU: {iou:.3f}")
        ax.axis("off")

        # --- Plot 3: Unterschied / Overlap ---
        # TP: in beiden Masken (gruen), FN: nur GT (blau), FP: nur Prediction (rot)
        diff_rgb = np.zeros((*img_size, 3), dtype=np.uint8)
        tp = np.logical_and(gt_mask, pred_mask)
        fn = np.logical_and(gt_mask, np.logical_not(pred_mask))
        fp = np.logical_and(np.logical_not(gt_mask), pred_mask)

        diff_rgb[tp] = [0, 200, 0]     # gruen: richtig erkannt
        diff_rgb[fn] = [0, 0, 200]     # blau: von GT verpasst
        diff_rgb[fp] = [200, 0, 0]     # rot: faelschlich vorhergesagt

        ax = axes[i, 2]
        ax.imshow(img_np)
        ax.imshow(diff_rgb, alpha=0.5)
        ax.set_title(f"Sample {i} — Diff (grün=TP, blau=FN, rot=FP)")
        ax.axis("off")

    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close(fig)

    denoiser.train()
    print(f"Visualisierung gespeichert unter: {out_path}")