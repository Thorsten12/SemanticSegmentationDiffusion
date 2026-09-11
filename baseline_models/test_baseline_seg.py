"""Evaluate a trained baseline segmenter (UNet-decoder or simple-head) on the
TEST split and report Dice/IoU, mirroring how you'd verify P2SDiff.

    python -m test_baseline_seg --dataset ph2 --skin_root /path/to/PH2 \
        --checkpoint ./runs/baseline_unet/best.pth --model unet \
        --out_dir ./runs/baseline_unet/test_eval
"""

import argparse
import json
import os

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

from .baseline_seg import (
    BaselineUNetSegmenter,
    SimpleHeadSegmenter,
    dice_iou_score,
    compute_model_stats,
)


@torch.no_grad()
def save_prediction_grid(model, loader, device, path, n_images=4, thresh=0.5):
    """Same 4-column layout as training-time visualization: Input / GT / Pred / Overlay."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    model.eval()
    images, _points, masks = next(iter(loader))
    images = images[:n_images].to(device)
    masks = masks[:n_images].float()
    if masks.dim() == 3:
        masks = masks.unsqueeze(1)

    logits = model(images)
    if logits.shape[-2:] != masks.shape[-2:]:
        logits = F.interpolate(logits, size=masks.shape[-2:], mode="bilinear", align_corners=False)
    probs = torch.sigmoid(logits).cpu()
    preds = (probs > thresh).float()

    imgs_np = (images.cpu() * 0.5 + 0.5).clamp(0, 1).permute(0, 2, 3, 1).numpy()
    masks_np = masks.cpu().squeeze(1).numpy()
    preds_np = preds.squeeze(1).numpy()

    n = imgs_np.shape[0]
    fig, axes = plt.subplots(n, 4, figsize=(12, 3 * n))
    if n == 1:
        axes = axes[None, :]
    for i in range(n):
        axes[i, 0].imshow(imgs_np[i]); axes[i, 0].set_title("Input" if i == 0 else ""); axes[i, 0].axis("off")
        axes[i, 1].imshow(masks_np[i], cmap="gray"); axes[i, 1].set_title("GT mask" if i == 0 else ""); axes[i, 1].axis("off")
        axes[i, 2].imshow(preds_np[i], cmap="gray"); axes[i, 2].set_title("Prediction" if i == 0 else ""); axes[i, 2].axis("off")

        overlay = imgs_np[i].copy()
        gt_layer = np.zeros_like(overlay); gt_layer[..., 1] = masks_np[i]
        pred_layer = np.zeros_like(overlay); pred_layer[..., 0] = preds_np[i]
        overlay = overlay * (1 - 0.5 * masks_np[i][..., None]) + gt_layer * 0.5
        overlay = overlay * (1 - 0.5 * preds_np[i][..., None]) + pred_layer * 0.5
        overlay = overlay.clip(0, 1)
        axes[i, 3].imshow(overlay); axes[i, 3].set_title("GT (green) / Pred (red)" if i == 0 else ""); axes[i, 3].axis("off")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description="Test-set evaluation for baseline segmenters")
    parser.add_argument("--dataset", choices=["ph2", "isic2017", "isic2018", "ham10000"], required=True)
    parser.add_argument("--skin_root", type=str, required=True)
    parser.add_argument("--checkpoint", type=str, required=True,
                         help="Path to a best.pth / last.pth saved by baseline_seg.py")
    parser.add_argument("--config", type=str, default=None,
                         help="Optional path to the config.json saved next to the checkpoint "
                              "by baseline_seg.py. If given, --model/--backbone/--decoder_dim/"
                              "--img_size are auto-filled from it (CLI values still override).")
    parser.add_argument("--model", choices=["unet", "simple_head"], default=None,
                         help="Must match the architecture the checkpoint was trained with. "
                              "Required unless --config is given.")
    parser.add_argument("--out_dir", type=str, default="./runs/baseline_test")
    parser.add_argument("--img_size", type=int, default=None)
    parser.add_argument("--backbone", type=str, default=None)
    parser.add_argument("--decoder_dim", type=int, default=None)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--thresh", type=float, default=0.5)
    parser.add_argument("--device", type=str, default="cuda")
    args = parser.parse_args()

    # ----- merge with training config.json, if given: CLI args always win -----
    if args.config is not None:
        with open(args.config) as f:
            train_cfg = json.load(f)
        if args.model is None:
            args.model = train_cfg.get("model")
        if args.img_size is None:
            args.img_size = train_cfg.get("img_size", 256)
        if args.backbone is None:
            args.backbone = train_cfg.get("backbone", "convnext_tiny")
        if args.decoder_dim is None:
            args.decoder_dim = train_cfg.get("decoder_dim", 64)
        print(f"Loaded training config from {args.config}: "
              f"model={args.model}, backbone={args.backbone}, "
              f"decoder_dim={args.decoder_dim}, img_size={args.img_size}")

    # fall back to defaults for anything still unset (no --config given)
    args.img_size = args.img_size or 256
    args.backbone = args.backbone or "convnext_tiny"
    args.decoder_dim = args.decoder_dim or 64
    if args.model is None:
        raise ValueError("Either --model or --config (pointing to the training run's "
                          "config.json) must be given so the correct architecture is built.")

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    os.makedirs(args.out_dir, exist_ok=True)

    # ----- data: TEST split -----
    from .data import build_contour_dataset
    img_size = (args.img_size, args.img_size)
    test_ds = build_contour_dataset(args.skin_root, args.dataset, "te", n_points=1,
                                     img_size=img_size, augment=False)
    test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False,
                              num_workers=args.num_workers)
    print(f"Test set: {len(test_ds)} samples")

    # ----- model (architecture must match checkpoint) -----
    if args.model == "unet":
        model = BaselineUNetSegmenter(
            backbone_name=args.backbone, pretrained=False,  # weights come from checkpoint
            freeze_backbone=False, decoder_dim=args.decoder_dim, num_classes=1,
        ).to(device)
    else:
        model = SimpleHeadSegmenter(
            backbone_name=args.backbone, pretrained=False,
            freeze_backbone=False, head_dim=args.decoder_dim, num_classes=1,
        ).to(device)

    ckpt = torch.load(args.checkpoint, map_location=device)
    state_dict = ckpt["model"] if isinstance(ckpt, dict) and "model" in ckpt else ckpt
    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    if missing:
        print(f"[WARN] Missing keys when loading checkpoint: {missing}")
    if unexpected:
        print(f"[WARN] Unexpected keys when loading checkpoint: {unexpected}")
    if isinstance(ckpt, dict) and "epoch" in ckpt:
        print(f"Loaded checkpoint: {args.checkpoint} (trained epoch {ckpt['epoch']}, "
              f"val Dice {ckpt.get('val_dice', float('nan')):.4f})")
    else:
        print(f"Loaded checkpoint: {args.checkpoint}")
    model.eval()

    # ----- full test-set evaluation -----
    dices, ious = [], []
    with torch.no_grad():
        for images, _points, masks in tqdm(test_loader, desc="Testing"):
            images = images.to(device)
            masks = masks.to(device).float()
            if masks.dim() == 3:
                masks = masks.unsqueeze(1)
            logits = model(images)
            if logits.shape[-2:] != masks.shape[-2:]:
                logits = F.interpolate(logits, size=masks.shape[-2:], mode="bilinear", align_corners=False)
            d, i = dice_iou_score(logits, masks, thresh=args.thresh)
            dices.append(d); ious.append(i)

    test_dice = float(np.mean(dices))
    test_iou = float(np.mean(ious))
    test_dice_std = float(np.std(dices))
    test_iou_std = float(np.std(ious))

    print(f"\n=== Test results ({args.model}) ===")
    print(f"Dice: {test_dice:.4f} +/- {test_dice_std:.4f}")
    print(f"IoU:  {test_iou:.4f} +/- {test_iou_std:.4f}")

    results = {
        "model": args.model,
        "checkpoint": args.checkpoint,
        "n_test": len(test_ds),
        "dice_mean": test_dice, "dice_std": test_dice_std,
        "iou_mean": test_iou, "iou_std": test_iou_std,
    }

    print("Computing FLOPs / latency / memory stats...")
    model_stats = compute_model_stats(model, device, img_size=args.img_size)
    results.update(model_stats)

    with open(os.path.join(args.out_dir, "test_results.json"), "w") as f:
        json.dump(results, f, indent=2)

    # qualitative check on first batch, same 4-column layout as training viz
    save_prediction_grid(model, test_loader, device,
                          path=os.path.join(args.out_dir, "test_predictions.png"),
                          thresh=args.thresh)

    print(f"Saved results + visualization to {args.out_dir}")


if __name__ == "__main__":
    main()