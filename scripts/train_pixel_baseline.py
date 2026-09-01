#!/usr/bin/env python3
"""Pixel-segmentation baselines on the published P2SDiff splits.

Trains a from-scratch U-Net and/or ImageNet-pretrained PVT-v2-b2 with a
standard FPN decoder. Same PH2 80/20/100 npy split, same light augmentation,
BCE+Dice. This is the fair pixel ceiling to compare against contour diffusion.

    python -m scripts.train_pixel_baseline --model unet --dataset ph2
    python -m scripts.train_pixel_baseline --model pvt --dataset ph2
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.data import build_contour_dataset
from src.models.feature_unet import FeatureUNet
from src.models.pvtv2 import pvt_v2_b2
from src.utils.rasterize import dice_score, iou_score


def dice_loss(logits, masks, eps=1e-6):
    pred = torch.sigmoid(logits)
    inter = (pred * masks).sum(dim=(2, 3))
    denom = pred.sum(dim=(2, 3)) + masks.sum(dim=(2, 3))
    return (1.0 - (2.0 * inter + eps) / (denom + eps)).mean()


class PixelUNet(nn.Module):
    """From-scratch U-Net encoder + lightweight FPN head (pixel logits)."""

    def __init__(self, start_dim=64, dim_mults=(1, 2, 4, 8)):
        super().__init__()
        self.backbone = FeatureUNet(
            in_channels=3, start_dim=start_dim, dim_mults=dim_mults,
        )
        hidden = start_dim
        chs = self.backbone.feature_channels
        self.laterals = nn.ModuleList(nn.Conv2d(c, hidden, 1) for c in chs)
        self.head = nn.Conv2d(hidden, 1, 1)

    def forward(self, x):
        feats = self.backbone(x)  # finest-first pyramid
        p = self.laterals[-1](feats[-1])
        for lat, f in zip(reversed(self.laterals[:-1]), reversed(feats[:-1])):
            p = F.interpolate(p, size=f.shape[-2:], mode="bilinear", align_corners=True)
            p = p + lat(f)
        p = F.interpolate(p, size=x.shape[-2:], mode="bilinear", align_corners=True)
        return self.head(p)


class PVTPixelSeg(nn.Module):
    """PVT-v2-b2 + top-down FPN decoder (pixel logits at input resolution)."""

    def __init__(self, weights, hidden=64):
        super().__init__()
        self.backbone = pvt_v2_b2()
        sd = torch.load(weights, map_location="cpu")
        sd = {k: v for k, v in sd.items() if not k.startswith("head.")}
        missing, unexpected = self.backbone.load_state_dict(sd, strict=False)
        print(f"PVT load: missing={len(missing)} unexpected={len(unexpected)}")
        channels = [64, 128, 320, 512]
        self.laterals = nn.ModuleList(nn.Conv2d(c, hidden, 1) for c in channels)
        self.smooth = nn.ModuleList(
            nn.Sequential(
                nn.Conv2d(hidden, hidden, 3, padding=1),
                nn.GroupNorm(8, hidden),
                nn.GELU(),
            )
            for _ in channels
        )
        self.head = nn.Sequential(
            nn.Conv2d(hidden, hidden, 3, padding=1),
            nn.GELU(),
            nn.Conv2d(hidden, 1, 1),
        )
        mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
        self.register_buffer("mean", mean)
        self.register_buffer("std", std)

    def forward(self, x):
        x = (x * 0.5 + 0.5 - self.mean) / self.std
        feats = self.backbone(x)  # finest-first, stride 4..32
        p = self.laterals[-1](feats[-1])
        p = self.smooth[-1](p)
        for lat, sm, f in zip(reversed(self.laterals[:-1]), reversed(self.smooth[:-1]), reversed(feats[:-1])):
            p = F.interpolate(p, size=f.shape[-2:], mode="bilinear", align_corners=True)
            p = sm(p + lat(f))
        p = F.interpolate(p, size=x.shape[-2:], mode="bilinear", align_corners=True)
        return self.head(p)


def build_loaders(args):
    common = dict(
        skin_root=args.skin_root, dataset=args.dataset, n_points=32,
        img_size=(args.img_size, args.img_size), npy_size=args.img_size,
        data_root=args.data_root, aug_level="light",
    )
    train = build_contour_dataset(**common, split="tr", augment=True)
    val = build_contour_dataset(**common, split="vl", augment=False)
    test = build_contour_dataset(**common, split="te", augment=False)
    kw = dict(num_workers=4, pin_memory=True)
    return (
        DataLoader(train, batch_size=args.batch_size, shuffle=True, drop_last=True, **kw),
        DataLoader(val, batch_size=args.batch_size, shuffle=False, **kw),
        DataLoader(test, batch_size=args.batch_size, shuffle=False, **kw),
        len(train), len(val), len(test),
    )


@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    dices, ious = [], []
    for images, _, masks in loader:
        images = images.to(device)
        logits = model(images)
        pred = (torch.sigmoid(logits) > 0.5).float().cpu().numpy()
        gt = (masks > 0.5).float().numpy()
        for p, g in zip(pred, gt):
            dices.append(dice_score(p[0], g[0]))
            ious.append(iou_score(p[0], g[0]))
    return float(np.mean(dices)), float(np.mean(ious))


def param_groups(model, args):
    if args.model != "pvt":
        return model.parameters()
    backbone, head = [], []
    for n, p in model.named_parameters():
        (backbone if n.startswith("backbone.") else head).append(p)
    return [
        {"params": backbone, "lr": args.backbone_lr},
        {"params": head, "lr": args.lr},
    ]


def train_one(args):
    os.makedirs(args.out_dir, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    train_loader, val_loader, test_loader, n_tr, n_vl, n_te = build_loaders(args)
    print(f"{args.dataset} split: {n_tr}/{n_vl}/{n_te}  model={args.model}")

    if args.model == "unet":
        model = PixelUNet(start_dim=64, dim_mults=(1, 2, 4, 8)).to(device)
    else:
        model = PVTPixelSeg(args.pvt_weights).to(device)
    n_params = sum(p.numel() for p in model.parameters()) / 1e6
    print(f"params: {n_params:.2f}M")

    opt = torch.optim.AdamW(param_groups(model, args), lr=args.lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs, eta_min=1e-6)
    scaler = torch.cuda.amp.GradScaler(enabled=device.type == "cuda")

    best_val, best_path = -1.0, os.path.join(args.out_dir, "best.pth")
    history = []

    for epoch in range(1, args.epochs + 1):
        model.train()
        losses = []
        for images, _, masks in tqdm(train_loader, desc=f"ep {epoch}/{args.epochs}", leave=False):
            images = images.to(device)
            masks = masks.to(device)
            opt.zero_grad(set_to_none=True)
            with torch.cuda.amp.autocast(enabled=device.type == "cuda"):
                logits = model(images)
                loss = F.binary_cross_entropy_with_logits(logits, masks) + dice_loss(logits, masks)
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(opt)
            scaler.update()
            losses.append(loss.item())
        sched.step()

        row = {"epoch": epoch, "loss": float(np.mean(losses))}
        if epoch % args.eval_every == 0 or epoch == args.epochs:
            val_dice, val_iou = evaluate(model, val_loader, device)
            row.update(val_dice=val_dice, val_iou=val_iou)
            print(f"epoch {epoch:4d}  loss {row['loss']:.4f}  val Dice {val_dice:.4f}  IoU {val_iou:.4f}")
            if val_dice > best_val:
                best_val = val_dice
                torch.save({"model": model.state_dict(), "epoch": epoch, "val_dice": val_dice}, best_path)
                print(f"  -> new best val Dice {best_val:.4f}")
        history.append(row)

    ckpt = torch.load(best_path, map_location=device)
    model.load_state_dict(ckpt["model"])
    test_dice, test_iou = evaluate(model, test_loader, device)
    summary = {
        "model": args.model,
        "dataset": args.dataset,
        "params_m": n_params,
        "best_val_dice": best_val,
        "best_epoch": ckpt["epoch"],
        "test_dice": test_dice,
        "test_iou": test_iou,
        "epochs": args.epochs,
        "split": [n_tr, n_vl, n_te],
    }
    print(json.dumps(summary, indent=2))
    with open(os.path.join(args.out_dir, "summary.json"), "w") as f:
        json.dump({"summary": summary, "history": history}, f, indent=2)
    return summary


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", choices=["unet", "pvt"], required=True)
    p.add_argument("--dataset", default="ph2")
    p.add_argument("--skin_root", default="/hdd/datasets/Skin")
    p.add_argument("--data_root", default="/hdd/datasets")
    p.add_argument("--img_size", type=int, default=224)
    p.add_argument("--epochs", type=int, default=200)
    p.add_argument("--batch_size", type=int, default=8)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--backbone_lr", type=float, default=1e-5)
    p.add_argument("--eval_every", type=int, default=10)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--pvt_weights", default="pretrained_pth/pvt/pvt_v2_b2.pth")
    p.add_argument("--out_dir", default=None)
    args = p.parse_args()
    if args.out_dir is None:
        args.out_dir = f"src/runs/pixel_{args.model}_{args.dataset}"
    train_one(args)


if __name__ == "__main__":
    main()
