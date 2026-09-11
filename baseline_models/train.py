"""Baseline segmentation model for comparison against P2SDiff.

Uses the SAME pretrained backbone as the diffusion model's conditioner
(convnext / resnet / swin / pvt / vmamba), paired with a swappable decoder
head from baseline_models.decoders (unet, shallow_conv_head, linear_mlp,
...). No diffusion, no contour points, no timestep conditioning -- a
standard encoder-decoder segmentation baseline.

Decoder selection is via --decoder (preferred) or the legacy --model flag
(kept as an alias so existing sweep scripts / already-trained checkpoints
under table/<encoder>/baseline_<name>/... keep working unchanged):

    --model unet          == --decoder unet
    --model simple_head   == --decoder shallow_conv_head

    python -m baseline_seg --dataset isic2018 --epochs 300 --batch_size 8 --encoder convnext --backbone convnext_tiny --decoder unet
    python -m baseline_seg --dataset busi --epochs 300 --batch_size 8 --encoder pvt --pvt_variant pvt_v2_b2 --decoder linear_mlp
    python -m baseline_seg --dataset polyp_kvasir --epochs 300 --batch_size 8 --encoder resnet --decoder unet

Dataset protocol notes (see dataset_loader.py for full detail):
  - ph2 is a standalone, unseen ZERO-SHOT TEST SET only -- it has no tr/vl
    and cannot be passed to --dataset directly (main() refuses it with a
    clear error). It is only ever used as an *additional* evaluation at the
    end of an isic2018 run (see the PH2 zero-shot eval block in main()),
    never for training and never for early stopping/checkpoint selection.
  - busi is a seeded 70/10/20 tr/vl/te split.
  - "polyp" no longer exists as a single dataset name -- it is replaced by
    5 separate dataset keys (polyp_clinicdb, polyp_kvasir, polyp_colondb,
    polyp_etis, polyp_cvc300), each with its own tr/vl/te (tr/vl are the
    SAME shared PraNet train pool; only te differs per test set).
  - tn3k is a seeded 80/20 tr/vl split (from the official trainval pool)
    plus the official test split.
"""

import argparse
import json
import os
import random

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

from .decoders import build_decoder, DECODER_REGISTRY, DECODER_ALIASES


# --------------------------------------------------------------------------- #
# Encoder factory -- builds the chosen conditioner with a plain
# argparse.Namespace, so this file has no dependency on a project-wide cfg
# object. All conditioners expose the same interface used below:
#   .extract(image) -> list of feature maps, finest scale first
#   .feature_channels -> list of per-scale channel counts
# --------------------------------------------------------------------------- #
def build_encoder(args):
    """Instantiate the chosen pretrained encoder ('convnext', 'resnet', 'swin', 'pvt', 'vmamba')."""
    from .models import ConvNeXtConditioner, ResNetConditioner, SwinConditioner, PVTConditioner, VMambaConditioner  # adjust import to your package layout

    if args.encoder == "convnext":
        return ConvNeXtConditioner(
            backbone=args.backbone,
            pretrained=True,
            freeze=args.freeze_backbone,
            stem_dim=0,  # no fourier stem needed for this baseline
        )
    elif args.encoder == "resnet":
        return ResNetConditioner(
            pretrained=True,
            freeze=args.freeze_backbone,
            stem_dim=0,
        )
    elif args.encoder == "swin":
        return SwinConditioner(
            pretrained=True,
            freeze=args.freeze_backbone,
            stem_dim=0,
        )
    elif args.encoder == "pvt":
        pretrained_path = args.pvt_pretrained_path or (
            "/loctmp/sit28238/SemanticSegmentationDiffusion/pretrained/pvt_v2_b2.pth"
        )
        return PVTConditioner(
            variant=args.pvt_variant,
            pretrained_path=pretrained_path,
            freeze=args.freeze_backbone,
            stem_dim=0,
        )
    elif args.encoder == "vmamba":
        return VMambaConditioner(
            pretrained=True,
            freeze=args.freeze_backbone,
            stem_dim=0,
            checkpoint_path="/loctmp/sit28238/SemanticSegmentationDiffusion/pretrained/vssm1_tiny_0230s_ckpt_epoch_264.pth",
        )
    raise ValueError(f"Unknown --encoder '{args.encoder}' (expected 'convnext', 'resnet', 'swin', 'vmamba', or 'pvt').")


# --------------------------------------------------------------------------- #
# Full baseline model: pretrained backbone -> swappable decoder -> logits.
# Replaces the old BaselineUNetSegmenter / SimpleHeadSegmenter pair with a
# single class parameterized by --decoder, via decoders.build_decoder().
# --------------------------------------------------------------------------- #
class BaselineSegmenter(nn.Module):
    """Standard encoder-decoder segmentation model, no diffusion.

    Reuses build_encoder() as the backbone wrapper so the ImageNet
    preprocessing / freeze logic / checkpoint loading is identical to what
    P2SDiff's encoder does. The decoder is looked up from DECODER_REGISTRY
    by name (see baseline_models/decoders/), so any decoder implementing
    the PyramidDecoder contract (feats -> logits) can be dropped in without
    touching this class or the training loop.
    """

    def __init__(self, args, decoder_dim=64, num_classes=1, out_size=None):
        super().__init__()
        self.backbone_wrapper = build_encoder(args)
        enc_ch = self.backbone_wrapper.feature_channels  # [c4, c8, c16, c32] finest->coarsest
        assert len(enc_ch) == 4, (
            f"Expected a 4-scale pyramid (stride 4/8/16/32) from the encoder, got "
            f"{len(enc_ch)} scales ({enc_ch}). This baseline is only wired for "
            f"4-scale encoders."
        )

        self.decoder = build_decoder(args.decoder, enc_ch, decoder_dim=decoder_dim, num_classes=num_classes)
        self.out_size = out_size  # optional (H, W) to force-resize logits

    def forward(self, image):
        feats = self.backbone_wrapper.extract(image)  # [s4, s8, s16, s32]
        
        feats = [f.contiguous() for f in feats]
        
        logits = self.decoder(feats)
        return logits


# --------------------------------------------------------------------------- #
# Losses (BCE + soft Dice, the standard combo for binary seg baselines)
# --------------------------------------------------------------------------- #
def dice_loss(logits, target, eps=1e-6):
    probs = torch.sigmoid(logits)
    probs = probs.flatten(1)
    target = target.flatten(1)
    inter = (probs * target).sum(-1)
    union = probs.sum(-1) + target.sum(-1)
    dice = (2 * inter + eps) / (union + eps)
    return 1 - dice.mean()


def bce_dice_loss(logits, target, dice_weight=1.0, bce_weight=1.0):
    bce = F.binary_cross_entropy_with_logits(logits, target)
    dl = dice_loss(logits, target)
    return bce_weight * bce + dice_weight * dl, {"bce": bce.detach(), "dice": dl.detach()}


@torch.no_grad()
def dice_iou_score(logits, target, thresh=0.5, eps=1e-6):
    probs = (torch.sigmoid(logits) > thresh).float()
    probs = probs.flatten(1)
    t = target.flatten(1)
    inter = (probs * t).sum(-1)
    union = probs.sum(-1) + t.sum(-1)
    dice = (2 * inter + eps) / (union + eps)
    iou = (inter + eps) / (union - inter + eps)
    return dice.mean().item(), iou.mean().item()


# --------------------------------------------------------------------------- #
# Minimal training loop, mirrors your train.py structure/logging so runs are
# easy to compare side by side.
# --------------------------------------------------------------------------- #
@torch.no_grad()
def visualize_predictions(model, val_loader, device, path, n_images=4, thresh=0.5):
    """Save a grid: rows = first n_images val samples, cols = [input, GT mask, pred mask]."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    model.eval()
    images, _points, masks = next(iter(val_loader))
    images = images[:n_images].to(device)
    masks = masks[:n_images].float()
    if masks.dim() == 3:
        masks = masks.unsqueeze(1)

    logits = model(images)
    if logits.shape[-2:] != masks.shape[-2:]:
        logits = F.interpolate(logits, size=masks.shape[-2:], mode="bilinear", align_corners=False)
    probs = torch.sigmoid(logits).cpu()
    preds = (probs > thresh).float()

    imgs_np = (images.cpu() * 0.5 + 0.5).clamp(0, 1).permute(0, 2, 3, 1).numpy()  # [-1,1] -> [0,1]
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
        gt_layer = np.zeros_like(overlay)
        gt_layer[..., 1] = masks_np[i]  # green channel = GT
        pred_layer = np.zeros_like(overlay)
        pred_layer[..., 0] = preds_np[i]  # red channel = prediction
        overlay = overlay * (1 - 0.5 * masks_np[i][..., None]) + gt_layer * 0.5
        overlay = overlay * (1 - 0.5 * preds_np[i][..., None]) + pred_layer * 0.5
        overlay = overlay.clip(0, 1)
        axes[i, 3].imshow(overlay); axes[i, 3].set_title("GT (green) / Pred (red)" if i == 0 else ""); axes[i, 3].axis("off")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    model.train()


@torch.no_grad()
def evaluate_split(model, loader, device):
    """Runs a full eval pass over `loader` and returns (mean_dice, mean_iou).
    Used for the periodic val-eval during training, the final held-out
    test-set eval, and the additional PH2 zero-shot eval (see main())."""
    model.eval()
    dices, ious = [], []
    for images, _points, masks in loader:
        images = images.to(device)
        masks = masks.to(device).float()
        if masks.dim() == 3:
            masks = masks.unsqueeze(1)
        logits = model(images)
        if logits.shape[-2:] != masks.shape[-2:]:
            logits = F.interpolate(logits, size=masks.shape[-2:], mode="bilinear", align_corners=False)
        d, i = dice_iou_score(logits, masks)
        dices.append(d); ious.append(i)
    model.train()
    mean_dice = sum(dices) / max(1, len(dices))
    mean_iou = sum(ious) / max(1, len(ious))
    return mean_dice, mean_iou


def compute_model_stats(model, device, img_size, n_warmup=5, n_timed=20):
    """Collects everything useful for comparing architectures head-to-head:
    param counts, FLOPs (if fvcore or thop is installed), inference latency,
    and peak GPU memory during a forward pass. Never raises -- if a library
    is missing or a measurement fails, that field is set to None so the run
    still completes and the rest of the stats are saved."""
    stats = {}

    n_total = sum(p.numel() for p in model.parameters())
    n_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    stats["total_params"] = n_total
    stats["trainable_params"] = n_trainable
    stats["total_params_M"] = round(n_total / 1e6, 3)
    stats["trainable_params_M"] = round(n_trainable / 1e6, 3)

    dummy = torch.randn(1, 3, img_size, img_size, device=device)
    model.eval()

    # ----- FLOPs -----
    stats["flops"] = None
    stats["flops_G"] = None
    stats["flops_source"] = None
    try:
        from fvcore.nn import FlopCountAnalysis
        with torch.no_grad():
            flop_counter = FlopCountAnalysis(model, dummy)
            flop_counter.unsupported_ops_warnings(False)  # expected (mul/add/gelu/silu etc. uncounted)
            flops = flop_counter.total()
        stats["flops"] = int(flops)
        stats["flops_G"] = round(flops / 1e9, 3)
        stats["flops_source"] = "fvcore"
    except ImportError:
        try:
            from thop import profile
            with torch.no_grad():
                macs, _ = profile(model, inputs=(dummy,), verbose=False)
            stats["flops"] = int(macs * 2)
            stats["flops_G"] = round(macs * 2 / 1e9, 3)
            stats["flops_source"] = "thop (MACs*2)"
        except ImportError:
            stats["flops_source"] = "unavailable (pip install fvcore or thop for this)"
    except Exception as e:
        stats["flops_source"] = f"error: {e}"

    # ----- Inference latency (GPU-synchronized wall-clock) -----
    stats["latency_ms_per_image"] = None
    try:
        import time
        with torch.no_grad():
            for _ in range(n_warmup):
                model(dummy)
            if device.type == "cuda":
                torch.cuda.synchronize()
            start = time.perf_counter()
            for _ in range(n_timed):
                model(dummy)
            if device.type == "cuda":
                torch.cuda.synchronize()
            elapsed = time.perf_counter() - start
        stats["latency_ms_per_image"] = round((elapsed / n_timed) * 1000, 3)
    except Exception as e:
        stats["latency_error"] = str(e)

    # ----- Peak GPU memory for a single forward pass -----
    stats["peak_memory_MB"] = None
    if device.type == "cuda":
        try:
            torch.cuda.reset_peak_memory_stats(device)
            with torch.no_grad():
                model(dummy)
            torch.cuda.synchronize()
            stats["peak_memory_MB"] = round(torch.cuda.max_memory_allocated(device) / 1e6, 2)
        except Exception as e:
            stats["peak_memory_error"] = str(e)

    model.train()
    return stats


def set_seed(seed):
    """Seeds python/numpy/torch RNGs for this run. Deliberately NOT setting
    torch.backends.cudnn.deterministic=True (that would make convolutions
    much slower); a plain seed is enough to make weight init, data shuffling,
    and augmentation reproducible/comparable across seeds, with only minor
    residual nondeterminism from cuDNN's algorithm selection -- acceptable
    for the mean +/- std comparison this sweep is used for."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _resolve_decoder(args):
    """Handles the --model/--decoder alias: --model (legacy) wins only if
    --decoder was left at its default and --model was explicitly given."""
    if args.decoder is not None:
        return args.decoder
    if args.model is not None:
        return DECODER_ALIASES.get(args.model, args.model)
    return "unet"  # historical default


# Datasets that can actually be trained on via main()'s tr/vl/te path.
# "ph2" is intentionally excluded -- see the guard in main().
TRAINABLE_DATASETS = [
    "isic2017", "isic2018", "ham10000",
    "busi", "tn3k",
    "polyp_clinicdb", "polyp_kvasir", "polyp_colondb", "polyp_etis", "polyp_cvc300",
]


def main():
    parser = argparse.ArgumentParser(description="Baseline segmenter (no diffusion), swappable decoder")
    parser.add_argument("--dataset", choices=TRAINABLE_DATASETS + ["ph2"], required=True,
                        help="Dataset to train on. NOTE: 'ph2' is a zero-shot-only test set "
                             "(no tr/vl) and cannot be trained on directly -- passing it here "
                             "will raise a clear error. Train on 'isic2018' instead; ph2 is "
                             "automatically evaluated as an extra zero-shot test at the end of "
                             "an isic2018 run.")
    parser.add_argument("--skin_root", type=str, default="/loctmp/sit28238/SemanticSegmentationDiffusion/data/datasets")

    # ----- decoder selection (new) + legacy alias -----
    parser.add_argument("--decoder", type=str, default=None,
                        choices=list(DECODER_REGISTRY.keys()),
                        help=f"Decoder head to use. Available: {sorted(DECODER_REGISTRY.keys())}")
    parser.add_argument("--model", type=str, default=None, choices=["unet", "linear_mlp", "semantic_fpn", "upernet", # Group A
                                                                    "cascade", "g-cascade", "emcad"],)               # Group B

    # ----- encoder selection -----
    parser.add_argument("--encoder", choices=["convnext", "pvt", "resnet", "swin", "vmamba"], default="convnext",
                        help="Which pretrained pyramid backbone to use as the encoder.")
    parser.add_argument("--backbone", type=str, default="convnext_tiny",
                        choices=["convnext_tiny", "convnext_small", "convnext_base"],
                        help="Only used when --encoder convnext.")
    parser.add_argument("--pvt_variant", type=str, default="pvt_v2_b2",
                        help="Only used when --encoder pvt (e.g. pvt_v2_b0 ... pvt_v2_b5).")
    parser.add_argument("--pvt_pretrained_path", type=str, default=None,
                        help="Only used when --encoder pvt. Defaults to the project's "
                             "pretrained/pvt_v2_b2.pth if not given.")

    parser.add_argument("--out_dir", type=str, default="baseline_models/runs/baseline_unet")
    parser.add_argument("--seed", type=int, default=0,
                         help="Random seed for weight init / data shuffling / augmentation. "
                              "Run the same (backbone, dataset, decoder) combo over several seeds "
                              "and report mean +/- std to get a statistically meaningful comparison "
                              "instead of a single noisy number.")
    parser.add_argument("--epochs", type=int, default=300,
                         help="Maximum number of epochs (upper bound; early stopping may end sooner). "
                              "Ignored if --max_steps is given.")
    parser.add_argument("--max_steps", type=int, default=8000,
                         help="Maximum number of gradient-update steps (recommended for cross-dataset "
                              "comparison, since dataset sizes differ a lot). Overrides --epochs: "
                              "epochs = ceil(max_steps / steps_per_epoch). Default 8000. Pass 0 to "
                              "disable and fall back to --epochs instead.")
    parser.add_argument("--patience", type=str, default="auto",
                         help="Early-stopping patience in epochs-without-val-improvement. "
                              "Pass an int, or 'auto' to use max(min_patience, epochs // 10). "
                              "Pass 0 to disable early stopping.")
    parser.add_argument("--min_patience", type=int, default=15,
                         help="Floor for --patience auto when epochs is small (e.g. HAM10000).")
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--img_size", type=int, default=256)
    parser.add_argument("--freeze_backbone", action="store_true")
    parser.add_argument("--decoder_dim", type=int, default=64)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--aug_level", type=str, default="light", choices=["none", "light", "strong"])
    parser.add_argument("--skip_ph2_zeroshot", action="store_true",
                        help="Skip the additional PH2 zero-shot evaluation that otherwise runs "
                             "automatically at the end of an --dataset isic2018 run.")
    args = parser.parse_args()

    args.decoder = _resolve_decoder(args)
    # keep args.model populated too (for config.json / summary.json readability
    # and for anything downstream that still reads it)
    args.model = args.model or args.decoder

    # ----- PH2 guard -----
    # PH2 has no tr/vl by design (see dataset_loader.py: TEST_ONLY_DATASETS).
    # It must never be trained on and never used for early stopping/checkpoint
    # selection -- that would leak "unseen-ness" into model selection and
    # undermine the zero-shot claim. Fail loudly and immediately, before any
    # data loading or training happens.
    if args.dataset == "ph2":
        raise ValueError(
            "--dataset ph2 is not trainable: ph2 has no 'tr'/'vl' split by design -- "
            "it's a standalone, unseen zero-shot test set (protocol: train on isic2018, "
            "evaluate zero-shot on ph2). Train with --dataset isic2018 instead; ph2 is "
            "then automatically evaluated as an extra zero-shot test at the end of that "
            "run (see the PH2 zero-shot eval block below, disable with --skip_ph2_zeroshot)."
        )

    set_seed(args.seed)

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    os.makedirs(args.out_dir, exist_ok=True)

    with open(os.path.join(args.out_dir, "config.json"), "w") as f:
        json.dump(vars(args), f, indent=2, default=str)

    # ----- data -----
    from .data import build_contour_dataset  # adjust import to your package layout
    img_size = (args.img_size, args.img_size)
    train_ds = build_contour_dataset(args.skin_root, args.dataset, "tr", n_points=1,
                                      img_size=img_size, augment=True)
    val_ds = build_contour_dataset(args.skin_root, args.dataset, "vl", n_points=1,
                                    img_size=img_size, augment=False)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                               num_workers=args.num_workers, drop_last=True, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False,
                             num_workers=args.num_workers)

    steps_per_epoch = len(train_loader)
    if args.max_steps and args.max_steps > 0:
        import math
        args.epochs = max(1, math.ceil(args.max_steps / steps_per_epoch))
        print(f"--max_steps {args.max_steps} given -> steps_per_epoch={steps_per_epoch} "
              f"-> epochs set to {args.epochs} (last epoch may overshoot max_steps slightly, "
              f"since the loop is still epoch-granular).")
    else:
        args.max_steps = None
        print(f"--max_steps disabled -> using --epochs {args.epochs} directly. "
              f"steps_per_epoch={steps_per_epoch} -> total budget = "
              f"{steps_per_epoch * args.epochs} steps.")

    # ----- model -----
    model = BaselineSegmenter(
        args, decoder_dim=args.decoder_dim, num_classes=1,
    ).to(device)

    if args.freeze_backbone:
        for p in model.backbone_wrapper.backbone.parameters():
            p.requires_grad_(False)

    trainable = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(trainable, lr=args.lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    n_train = sum(p.numel() for p in trainable)
    n_total = sum(p.numel() for p in model.parameters())
    backbone_desc = args.backbone if args.encoder == "convnext" else args.pvt_variant
    print(f"decoder={args.decoder} | encoder {args.encoder} ({backbone_desc}, frozen={args.freeze_backbone}) | "
          f"trainable {n_train/1e6:.2f}M / total {n_total/1e6:.2f}M | device {device} | seed {args.seed}")

    if args.patience == "auto":
        patience = max(args.min_patience, args.epochs // 10)
    else:
        patience = int(args.patience)
    print(f"Early stopping: patience={patience} "
          f"({'disabled' if patience <= 0 else f'auto = max({args.min_patience}, {args.epochs}//10)' if args.patience == 'auto' else 'fixed'})")

    config_full = vars(args).copy()
    config_full.update({
        "n_train_samples": len(train_ds),
        "n_val_samples": len(val_ds),
        "trainable_params": n_train,
        "total_params": n_total,
        "device_used": str(device),
        "resolved_patience": patience,
    })
    with open(os.path.join(args.out_dir, "config.json"), "w") as f:
        json.dump(config_full, f, indent=2, default=str)

    history = {"epoch": [], "loss": [], "val_epoch": [], "val_dice": [], "val_iou": [], "global_step": []}
    best_val = -1.0
    epochs_no_improve = 0
    stopped_early_at = None
    global_step = 0

    for epoch in range(args.epochs):
        model.train()
        running = 0.0
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{args.epochs}")
        for images, _points, masks in pbar:
            images = images.to(device, non_blocking=True)
            masks = masks.to(device, non_blocking=True).float()
            if masks.dim() == 3:
                masks = masks.unsqueeze(1)  # [B,1,H,W]

            optimizer.zero_grad(set_to_none=True)
            logits = model(images)
            if logits.shape[-2:] != masks.shape[-2:]:
                masks = F.interpolate(
                    masks.float(),
                    size=logits.shape[-2:],
                    mode="nearest",
                )
            loss, parts = bce_dice_loss(logits, masks)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            global_step += 1

            running += loss.item()
            pbar.set_postfix({"loss": f"{loss.item():.4f}",
                               "bce": f"{parts['bce'].item():.4f}",
                               "dice": f"{parts['dice'].item():.4f}",
                               "step": global_step})
        scheduler.step()

        avg = running / max(1, len(train_loader))
        history["epoch"].append(epoch + 1)
        history["loss"].append(avg)
        history["global_step"].append(global_step)
        print(f"Epoch {epoch+1} | train loss {avg:.4f}")

        # NOTE: this val eval (and everything derived from it -- best.pth
        # selection, early stopping) always runs against args.dataset's OWN
        # val split. For isic2018 that's isic2018's held-out val (259
        # samples) -- never ph2. PH2 is intentionally kept out of this loop
        # entirely; see the PH2 zero-shot eval block after training below.
        val_dice, val_iou = evaluate_split(model, val_loader, device)
        history["val_epoch"].append(epoch + 1)
        history["val_dice"].append(val_dice)
        history["val_iou"].append(val_iou)
        print(f"Epoch {epoch+1} | val Dice {val_dice:.4f} | val IoU {val_iou:.4f}")

        if val_dice > best_val:
            best_val = val_dice
            epochs_no_improve = 0
            torch.save({"model": model.state_dict(), "epoch": epoch + 1,
                        "val_dice": val_dice, "val_iou": val_iou},
                       os.path.join(args.out_dir, "best.pth"))
            print(f"  -> new best (Dice {val_dice:.4f}) -> best.pth")
        else:
            epochs_no_improve += 1

        with open(os.path.join(args.out_dir, "history.json"), "w") as f:
            json.dump(history, f, indent=2)

        if (epoch + 1) % 20 == 0:
            _plot_history(history, os.path.join(args.out_dir, "loss_curve.png"))
            visualize_predictions(
                model, val_loader, device,
                path=os.path.join(args.out_dir, f"val_preds_epoch{epoch+1}.png"),
            )

        if patience > 0 and epochs_no_improve >= patience:
            stopped_early_at = epoch + 1
            print(f"Early stopping: no val Dice improvement for {patience} epochs "
                  f"(stopped at epoch {stopped_early_at}, best was {best_val:.4f})")
            break

    _plot_history(history, os.path.join(args.out_dir, "loss_curve.png"))
    print(f"Done. Best val Dice {best_val:.4f}. Artifacts in {args.out_dir}")

    # ----- final TEST-set evaluation, using the best checkpoint (by val Dice) -----
    print("Loading best.pth for final test-set evaluation...")
    test_dice, test_iou = None, None
    test_ds = None
    best_ckpt_path = os.path.join(args.out_dir, "best.pth")
    if os.path.isfile(best_ckpt_path):
        ckpt = torch.load(best_ckpt_path, map_location=device)
        model.load_state_dict(ckpt["model"])

        test_ds = build_contour_dataset(args.skin_root, args.dataset, "te", n_points=1,
                                         img_size=img_size, augment=False)
        test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False,
                                  num_workers=args.num_workers)
        test_dice, test_iou = evaluate_split(model, test_loader, device)
        print(f"Test Dice {test_dice:.4f} | Test IoU {test_iou:.4f} "
              f"(best.pth from epoch {ckpt.get('epoch')}, n_test={len(test_ds)})")
    else:
        print(f"  -> WARNING: no best.pth found at {best_ckpt_path}, skipping test eval "
              f"(this should only happen if training crashed before any val improvement).")

    # ----- additional PH2 zero-shot evaluation (isic2018 runs only) -----
    # Protocol: "train on ISIC 2018, evaluate zero-shot on PH2". PH2 is used
    # here ONLY as a final, additional, unseen test set alongside isic2018's
    # own held-out test set above -- it never participated in training or in
    # best.pth/early-stopping selection (that happened on isic2018's own val
    # split, see the training loop above).
    ph2_dice, ph2_iou, n_ph2 = None, None, None
    if args.dataset == "isic2018" and not args.skip_ph2_zeroshot:
        if os.path.isfile(best_ckpt_path):
            print("Running additional zero-shot evaluation on PH2 (unseen)...")
            ph2_ds = build_contour_dataset(args.skin_root, "ph2", "te", n_points=1,
                                            img_size=img_size, augment=False)
            ph2_loader = DataLoader(ph2_ds, batch_size=args.batch_size, shuffle=False,
                                     num_workers=args.num_workers)
            ph2_dice, ph2_iou = evaluate_split(model, ph2_loader, device)
            n_ph2 = len(ph2_ds)
            print(f"PH2 zero-shot Dice {ph2_dice:.4f} | IoU {ph2_iou:.4f} (n={n_ph2})")
        else:
            print("  -> WARNING: no best.pth found, skipping PH2 zero-shot eval too.")

    print("Computing FLOPs / latency / memory stats...")
    model_stats = compute_model_stats(model, device, img_size=args.img_size)

    best_epoch_idx = history["val_dice"].index(max(history["val_dice"])) if history["val_dice"] else None
    summary = {
        "decoder": args.decoder,
        "model": args.model,  # kept for backward compatibility with old collect_table.py scripts
        "encoder": args.encoder,
        "backbone": args.backbone if args.encoder == "convnext" else args.pvt_variant,
        "dataset": args.dataset,
        "seed": args.seed,
        "img_size": args.img_size,
        "max_epochs": args.epochs,
        "epochs_trained": stopped_early_at or (history["epoch"][-1] if history["epoch"] else 0),
        "steps_per_epoch": steps_per_epoch,
        "global_steps_trained": global_step,
        "max_steps_requested": args.max_steps,
        "stopped_early": stopped_early_at is not None,
        "patience": patience,
        "best_val_dice": best_val,
        "best_val_iou": history["val_iou"][best_epoch_idx] if best_epoch_idx is not None else None,
        "best_val_epoch": history["val_epoch"][best_epoch_idx] if best_epoch_idx is not None else None,
        "best_val_step": (history["global_step"][history["epoch"].index(history["val_epoch"][best_epoch_idx])]
                           if best_epoch_idx is not None else None),
        "test_dice": test_dice,
        "test_iou": test_iou,
        "n_test_samples": len(test_ds) if test_ds is not None else None,
        "ph2_zeroshot_dice": ph2_dice,
        "ph2_zeroshot_iou": ph2_iou,
        "n_ph2_zeroshot_samples": n_ph2,
        "final_train_loss": history["loss"][-1] if history["loss"] else None,
        **model_stats,
    }
    with open(os.path.join(args.out_dir, "summary.json"), "w") as f:
        json.dump(summary, f, indent=2)
    print(f"Saved comparison summary to {os.path.join(args.out_dir, 'summary.json')}")
    print(json.dumps(summary, indent=2))


def _plot_history(history, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax1 = plt.subplots(figsize=(10, 5))
    ax1.plot(history["epoch"], history["loss"], "k-", label="train loss")
    ax1.set_xlabel("epoch"); ax1.set_ylabel("loss"); ax1.grid(alpha=0.3)
    if history["val_dice"]:
        ax2 = ax1.twinx()
        ax2.plot(history["val_epoch"], history["val_dice"], "g.-", label="val Dice")
        ax2.plot(history["val_epoch"], history["val_iou"], "b.--", label="val IoU")
        ax2.set_ylabel("val Dice / IoU")
        lines1, labels1 = ax1.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax1.legend(lines1 + lines2, labels1 + labels2, loc="upper right")
    if history.get("global_step"):
        ax3 = ax1.twiny()
        ax3.set_xlim(ax1.get_xlim())
        step_ticks_idx = list(range(0, len(history["epoch"]), max(1, len(history["epoch"]) // 6)))
        ax3.set_xticks([history["epoch"][i] for i in step_ticks_idx])
        ax3.set_xticklabels([f"{history['global_step'][i]:,}" for i in step_ticks_idx])
        ax3.set_xlabel("global step")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


if __name__ == "__main__":
    main()
