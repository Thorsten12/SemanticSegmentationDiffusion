"""
sample.py

Lädt einen trainierten Checkpoint (z.B. best.pth oder checkpoint_epoch_X.pt)
und visualisiert Predictions auf einem Datensatz.

Ausgewählt werden:
  - die ersten 4 Samples des Datasets (Index 0..3, unabhängig vom Score)
  - die 4 Samples mit dem SCHLECHTESTEN Dice-Score im gesamten Set

Nutzt für Sampling/Rasterisierung dieselben Funktionen wie visualisation.py
(_predict_mask_for_sample, points_to_mask, denorm_image, denorm_points, dice_iou),
damit die Ergebnisse 1:1 vergleichbar mit dem Training sind.
"""

import argparse
import json
import os

import numpy as np
import torch
import matplotlib.pyplot as plt

from .models import build_conditioner, ContourDenoiser
from .data import ArrayContourDataset
from .diffusion import GaussianDiffusion
from .data_split import make_splits
from .visualisation import (
    _predict_mask_for_sample,
    points_to_mask,
    denorm_image,
    denorm_points,
    dice_iou,
)


def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ("yes", "true", "t", "y", "1"):
        return True
    elif v.lower() in ("no", "false", "f", "n", "0"):
        return False
    raise argparse.ArgumentTypeError(f"Boolean value expected, got: {v!r}")


def build_models_from_cfg(cfg):
    """Baut Encoder + Denoiser exakt wie in train.py, aber aus einem geladenen config.json."""
    class Cfg:
        pass
    args = Cfg()
    for k, v in cfg.items():
        setattr(args, k, v)

    encoder = build_conditioner(cfg=args)
    denoiser = ContourDenoiser(
        n_point=args.n_points,
        hidden_dim=args.hidden_dim,
        coord_fourier_bands=args.coord_fourier_bands,
        feature_channels=encoder.feature_channels,
    )
    return encoder, denoiser, args


@torch.no_grad()
def score_all_samples(denoiser, encoder, diffusion, dataset, device, n_points, img_size, guidance):
    """Berechnet für jedes Sample im Dataset (dice, iou) und gibt eine Liste zurück."""
    denoiser.eval()
    encoder.eval()

    results = []  # Liste von dicts: idx, dice, iou
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
        results.append({"idx": i, "dice": dice, "iou": iou})
        print(f"[{i+1}/{len(dataset)}] Dice: {dice:.4f} | IoU: {iou:.4f}")

    return results


@torch.no_grad()
def plot_selected(denoiser, encoder, diffusion, dataset, device, n_points, img_size,
                   guidance, selected, out_path):
    """
    selected: Liste von (idx, label) Tupeln, z.B. [(0, "first"), (17, "worst"), ...]
    label wird nur für die Zeilenbeschriftung verwendet ("erste" vs. "schlechteste").
    """
    denoiser.eval()
    encoder.eval()

    n_rows = len(selected)
    fig, axes = plt.subplots(n_rows, 3, figsize=(12, 4 * n_rows))
    if n_rows == 1:
        axes = axes[None, :]

    for row, (idx, label) in enumerate(selected):
        img_tensor, gt_points_tensor, _ = dataset[idx]

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
        dice, iou = dice_iou(pred_mask, gt_mask)

        img_np = denorm_image(img_tensor)
        gt_px = denorm_points(gt_points_np, img_size)
        pred_px = denorm_points(pred_points_np, img_size)

        # --- Plot 1: Ground Truth ---
        ax = axes[row, 0]
        ax.imshow(img_np)
        ax.imshow(gt_mask, alpha=0.4, cmap="Greens")
        ax.scatter(gt_px[:, 0], gt_px[:, 1], s=4, c="lime", alpha=0.6)
        ax.set_title(f"[{label}] idx={idx} — Ground Truth")
        ax.axis("off")

        # --- Plot 2: Prediction ---
        ax = axes[row, 1]
        ax.imshow(img_np)
        ax.scatter(pred_px[:, 0], pred_px[:, 1], s=4, c="yellow", alpha=0.6)
        ax.set_title(f"idx={idx} — Prediction\nDice: {dice:.3f} | IoU: {iou:.3f}")
        ax.axis("off")

        # --- Plot 3: Diff ---
        diff_rgb = np.zeros((*img_size, 3), dtype=np.uint8)
        tp = np.logical_and(gt_mask, pred_mask)
        fn = np.logical_and(gt_mask, np.logical_not(pred_mask))
        fp = np.logical_and(np.logical_not(gt_mask), pred_mask)
        diff_rgb[tp] = [0, 200, 0]
        diff_rgb[fn] = [0, 0, 200]
        diff_rgb[fp] = [200, 0, 0]

        ax = axes[row, 2]
        ax.imshow(img_np)
        ax.imshow(diff_rgb, alpha=0.5)
        ax.set_title(f"idx={idx} — Diff (grün=TP, blau=FN, rot=FP)")
        ax.axis("off")

    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"Visualisierung gespeichert unter: {out_path}")


def main():
    parser = argparse.ArgumentParser(description="Sampling / Visualisierung eines trainierten Checkpoints.")

    parser.add_argument("--checkpoint", type=str, required=True,
                         help="Pfad zu best.pth oder checkpoint_epoch_X.pt")
    parser.add_argument("--config", type=str, default=None,
                         help="Pfad zu config.json aus dem Trainingsrun. "
                              "Default: config.json im selben Ordner wie --checkpoint")
    parser.add_argument("--data", type=str, default=None,
                         help="Ordner mit X_tr_224x224.npy / Y_tr_224x224.npy (dein GESAMTES Set, "
                              "genau wie in train.py). Default: args.data aus config.json")
    parser.add_argument("--split", type=str, default="test", choices=["train", "val", "test"],
                         help="Welcher der drei rekonstruierten Splits ausgewertet wird. "
                              "Wird via make_splits() mit denselben random_state/val_size/test_size "
                              "wie beim Training aus dem vollen Set neu erzeugt.")
    parser.add_argument("--out_dir", type=str, default="./sample_out")
    parser.add_argument("--use_ema", type=str2bool, default=True,
                         help="EMA-Gewichte statt Roh-Gewichte für Inferenz nutzen")
    parser.add_argument("--guidance", type=float, default=None,
                         help="Default: guidance aus config.json")
    parser.add_argument("--n_worst", type=int, default=4)
    parser.add_argument("--n_first", type=int, default=4)

    args = parser.parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # --- Config laden (Modellarchitektur muss exakt wie beim Training sein) ---
    config_path = args.config or os.path.join(os.path.dirname(args.checkpoint), "config.json")
    with open(config_path, "r") as f:
        cfg = json.load(f)

    guidance = args.guidance if args.guidance is not None else cfg["guidance"]
    data_dir = args.data or cfg["data"]

    os.makedirs(args.out_dir, exist_ok=True)

    # --- Modelle bauen & Checkpoint laden ---
    encoder, denoiser, model_args = build_models_from_cfg(cfg)
    encoder.to(device)
    denoiser.to(device)

    ckpt = torch.load(args.checkpoint, map_location=device)

    if args.use_ema:
        # ema_state_dict enthält sowohl das Online- als auch das EMA-Modell (ema-pytorch Format).
        # Wir laden nur die EMA-Gewichte in unseren "nackten" denoiser.
        ema_sd = ckpt["ema_state_dict"]
        ema_weights = {
            k.replace("ema_model.", ""): v
            for k, v in ema_sd.items()
            if k.startswith("ema_model.")
        }
        missing, unexpected = denoiser.load_state_dict(ema_weights, strict=False)
        print(f"[EMA] geladen | missing={len(missing)} unexpected={len(unexpected)}")
    else:
        denoiser.load_state_dict(ckpt["denoiser_state_dict"])
        print("[Raw] denoiser_state_dict geladen")

    denoiser.eval()
    encoder.eval()

    print(f"Checkpoint von Epoch {ckpt.get('epoch', '?')} geladen "
          f"(val_dice={ckpt.get('val_mean_dice', '?')})")

    # --- Volles Set laden (dieselben Dateien wie in train.py) ---
    X_path = os.path.join(data_dir, "X_tr_224x224.npy")
    Y_path = os.path.join(data_dir, "Y_tr_224x224.npy")
    print(f"Lade X von: {os.path.abspath(X_path)}")
    print(f"Lade Y von: {os.path.abspath(Y_path)}")

    X_full = np.load(X_path)
    Y_full = np.load(Y_path)

    # --- Denselben Split wie beim Training reproduzieren ---
    # random_state/val_size/test_size kommen aus config.json, damit sample.py
    # garantiert denselben Split sieht wie train.py -- niemals hardcoden.
    random_state = cfg.get("random_state", 42)
    val_size = cfg.get("val_size", 0.2)
    test_size = cfg.get("test_size", 0.2)

    (X_train, Y_train), (X_val, Y_val), (X_test, Y_test) = make_splits(
        X_full, Y_full, random_state=random_state, val_size=val_size, test_size=test_size,
    )

    split_map = {
        "train": (X_train, Y_train),
        "val": (X_val, Y_val),
        "test": (X_test, Y_test),
    }
    X, Y = split_map[args.split]
    print(f"Nutze Split '{args.split}' mit {len(X)} Samples "
          f"(random_state={random_state}, val_size={val_size}, test_size={test_size})")

    dataset = ArrayContourDataset(
        images=X, masks=Y, n_points=model_args.n_points, img_size=(224, 224)
    )

    diffusion = GaussianDiffusion(
        timesteps=model_args.timesteps,
        beta_start=model_args.beta_start,
        beta_end=model_args.beta_end,
        device=device,
        use_uncertainty_weighting=False,  # für reines Sampling irrelevant
    )

    # --- Alle Samples bewerten ---
    results = score_all_samples(
        denoiser=denoiser, encoder=encoder, diffusion=diffusion, dataset=dataset,
        device=device, n_points=model_args.n_points, img_size=(224, 224), guidance=guidance,
    )

    mean_dice = float(np.mean([r["dice"] for r in results]))
    mean_iou = float(np.mean([r["iou"] for r in results]))
    print(f"\nGesamt über {len(results)} Samples: Mean Dice={mean_dice:.4f} | Mean IoU={mean_iou:.4f}")

    with open(os.path.join(args.out_dir, "scores.json"), "w") as f:
        json.dump(results, f, indent=2)

    # --- Erste n_first + schlechteste n_worst auswählen ---
    n_first = min(args.n_first, len(results))
    first_selected = [(results[i]["idx"], "erste") for i in range(n_first)]

    sorted_by_dice = sorted(results, key=lambda r: r["dice"])
    n_worst = min(args.n_worst, len(sorted_by_dice))
    worst_selected = [(r["idx"], "schlechteste") for r in sorted_by_dice[:n_worst]]

    selected = first_selected + worst_selected

    plot_selected(
        denoiser=denoiser, encoder=encoder, diffusion=diffusion, dataset=dataset,
        device=device, n_points=model_args.n_points, img_size=(224, 224), guidance=guidance,
        selected=selected, out_path=os.path.join(args.out_dir, "sample_first_and_worst.png"),
    )


if __name__ == "__main__":
    main()