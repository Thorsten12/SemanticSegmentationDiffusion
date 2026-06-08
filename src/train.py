"""Train the P2SDiff boundary-point diffusion model on PH2.

Run from the repository root:
    python -m src.train --epochs 300 --batch_size 8

Saves EMA checkpoints + a loss curve + periodic validation visualizations under
`cfg.out_dir`.
"""

import argparse
import json
import os

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

from .config import Config
from .data import build_contour_dataset, split_counts
from .diffusion import GaussianDiffusion
from .models import ContourDenoiser, build_conditioner
from .utils import EMA


def build_models(cfg: Config, device):
    encoder = build_conditioner(cfg).to(device)
    denoiser = ContourDenoiser(
        n_points=cfg.n_points,
        hidden_dim=cfg.hidden_dim,
        num_layers=cfg.n_transformer_layers,
        num_heads=cfg.n_heads,
        scale_channels=encoder.feature_channels,
        proj_dim=cfg.cond_channels,
        coord_fourier_bands=cfg.coord_fourier_bands,
    ).to(device)
    return encoder, denoiser


def main():
    parser = argparse.ArgumentParser(description="Train P2SDiff on PH2")
    # Only expose the knobs people tweak most; everything else lives in Config.
    parser.add_argument("--dataset", choices=["ph2", "isic2017", "isic2018", "ham10000"])
    parser.add_argument("--skin_root", type=str)
    parser.add_argument("--out_dir", type=str)
    parser.add_argument("--epochs", type=int)
    parser.add_argument("--batch_size", type=int)
    parser.add_argument("--lr", type=float)
    parser.add_argument("--n_points", type=int)
    parser.add_argument("--encoder", choices=["convnext", "pvt", "unet"])
    parser.add_argument("--backbone", type=str)
    parser.add_argument("--freeze_backbone", action="store_true", default=None,
                        help="freeze the pretrained backbone (train only fusion + denoiser)")
    parser.add_argument("--backbone_lr", type=float)
    parser.add_argument("--coord_fourier_bands", type=int)
    parser.add_argument("--pos_grid_bands", type=int)
    parser.add_argument("--device", type=str)
    parser.add_argument("--num_workers", type=int)
    parser.add_argument("--eval_every", type=int)
    parser.add_argument("--aug_level", choices=["none", "light", "strong"])
    parser.add_argument("--guidance_scale", type=float)
    parser.add_argument("--ddim_steps", type=int)
    parser.add_argument("--lambda_dice", type=float)
    parser.add_argument("--snr_gamma", type=float)
    parser.add_argument("--no_amp", action="store_true")
    args = parser.parse_args()

    cfg = Config.from_args(args)
    if args.no_amp:
        cfg.amp = False

    torch.manual_seed(cfg.seed)
    device = torch.device(cfg.device if torch.cuda.is_available() else "cpu")
    os.makedirs(cfg.out_dir, exist_ok=True)
    with open(os.path.join(cfg.out_dir, "config.json"), "w") as f:
        json.dump({k: getattr(cfg, k) for k in cfg.__dataclass_fields__}, f, indent=2, default=str)

    # ----- data (published index split, read from preprocessed npy) -----
    counts = split_counts(cfg.skin_root, cfg.dataset, cfg.npy_size)
    print(f"Dataset {cfg.dataset} | split -> train {counts['tr']} | "
          f"val {counts['vl']} | test {counts['te']}")
    train_ds = build_contour_dataset(cfg.skin_root, cfg.dataset, "tr", cfg.n_points,
                                     cfg.img_size, augment=cfg.augment,
                                     aug_level=cfg.aug_level, npy_size=cfg.npy_size)
    val_ds = build_contour_dataset(cfg.skin_root, cfg.dataset, "vl", cfg.n_points,
                                   cfg.img_size, augment=False, npy_size=cfg.npy_size)
    persist = cfg.num_workers > 0
    train_loader = DataLoader(train_ds, batch_size=cfg.batch_size, shuffle=True,
                              num_workers=cfg.num_workers, drop_last=True, pin_memory=True,
                              persistent_workers=persist)
    val_loader = DataLoader(val_ds, batch_size=cfg.batch_size, shuffle=False,
                            num_workers=cfg.num_workers, persistent_workers=persist)

    # ----- models / diffusion / optim -----
    encoder, denoiser = build_models(cfg, device)
    diffusion = GaussianDiffusion(cfg.timesteps, cfg.beta_start, cfg.beta_end, device=device)
    ema = EMA([encoder, denoiser], decay=cfg.ema_decay)

    # Discriminative LR: a low LR for the pretrained backbone (when fine-tuning),
    # the normal LR for the freshly-initialized fusion + denoiser.
    backbone = getattr(encoder, "backbone", None)
    backbone_ids = {id(p) for p in backbone.parameters()} if backbone is not None else set()
    backbone_params, head_params = [], []
    for p in encoder.parameters():
        if not p.requires_grad:
            continue
        (backbone_params if id(p) in backbone_ids else head_params).append(p)
    head_params += list(denoiser.parameters())

    param_groups = [{"params": head_params, "lr": cfg.lr}]
    if backbone_params:
        param_groups.append({"params": backbone_params, "lr": cfg.backbone_lr})
    optimizer = torch.optim.AdamW(param_groups, lr=cfg.lr, weight_decay=cfg.weight_decay)
    trainable = head_params + backbone_params
    use_amp = cfg.amp and device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

    n_train = sum(p.numel() for p in trainable)
    n_total = sum(p.numel() for p in encoder.parameters()) + sum(p.numel() for p in denoiser.parameters())
    bb_state = "frozen" if not backbone_params else f"fine-tune @ lr {cfg.backbone_lr:g}"
    print(f"Encoder: {cfg.encoder} (backbone {bb_state}) | trainable {n_train/1e6:.2f}M / "
          f"total {n_total/1e6:.2f}M | device {device} | amp {use_amp}")

    history = {"epoch": [], "loss": [], "val_epoch": [], "val_dice": [], "val_iou": []}
    best_val = -1.0

    for epoch in range(cfg.epochs):
        encoder.train(); denoiser.train()
        running = 0.0
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{cfg.epochs}")
        for images, points, masks in pbar:
            images = images.to(device, non_blocking=True)
            points = points.to(device, non_blocking=True)
            masks = masks.to(device, non_blocking=True)
            b = images.shape[0]

            optimizer.zero_grad(set_to_none=True)
            t = torch.randint(0, cfg.timesteps, (b,), device=device).long()

            with torch.amp.autocast("cuda", enabled=use_amp):
                noisy = diffusion.q_sample(points, t)
                raw = encoder.extract(images)              # backbone (once)
                cond_maps = encoder.fuse(raw, t)           # time-conditioned fusion
                # Classifier-free guidance: drop the condition per-sample (not
                # per-batch), so the unconditional branch sees varied examples.
                keep = (torch.rand(b, device=device) >= cfg.cfg_dropout)
                keep = keep.to(cond_maps[0].dtype).view(b, 1, 1, 1)
                cond_maps = [m * keep for m in cond_maps]
                pred_x0 = denoiser(noisy, t, cond_maps)
                pred_x0 = torch.clamp(pred_x0, -cfg.x0_clamp, cfg.x0_clamp)
                loss, parts = diffusion.training_losses(
                    pred_x0, points, t, masks=masks,
                    lambda_uniformity=cfg.lambda_uniformity,
                    lambda_dice=cfg.lambda_dice, snr_gamma=cfg.snr_gamma)

            scaler.scale(loss).backward()
            if cfg.grad_clip:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(trainable, cfg.grad_clip)
            scaler.step(optimizer)
            scaler.update()
            ema.update([encoder, denoiser])

            running += loss.item()
            pbar.set_postfix({"loss": f"{loss.item():.4f}",
                              "x0": f"{parts['loss_x0'].item():.4f}",
                              "dice": f"{parts['loss_dice'].item():.4f}",
                              "unif": f"{parts['loss_uniformity'].item():.4f}"})

        avg = running / max(1, len(train_loader))
        history["epoch"].append(epoch + 1)
        history["loss"].append(avg)
        print(f"Epoch {epoch+1} | train loss {avg:.4f}")

        # ----- periodic validation with EMA weights -----
        do_eval = (epoch + 1) % cfg.eval_every == 0 or (epoch + 1) == cfg.epochs
        if do_eval:
            from .sample import evaluate  # local import avoids a cycle at module load
            ema_encoder, ema_denoiser = ema.modules
            dice, iou = evaluate(ema_encoder, ema_denoiser, diffusion, val_loader, cfg, device,
                                 viz_path=os.path.join(cfg.out_dir, f"val_epoch{epoch+1}.png"))
            history["val_epoch"].append(epoch + 1)
            history["val_dice"].append(dice)
            history["val_iou"].append(iou)
            print(f"Epoch {epoch+1} | val Dice {dice:.4f} | val IoU {iou:.4f}")

            if dice > best_val:
                best_val = dice
                torch.save({
                    "encoder": ema_encoder.state_dict(),
                    "denoiser": ema_denoiser.state_dict(),
                    "epoch": epoch + 1,
                    "val_dice": dice, "val_iou": iou,
                    "config": {k: getattr(cfg, k) for k in cfg.__dataclass_fields__},
                }, os.path.join(cfg.out_dir, "best.pth"))
                print(f"  -> new best (Dice {dice:.4f}) -> best.pth")

        # Always keep the latest EMA checkpoint.
        ema_encoder, ema_denoiser = ema.modules
        torch.save({"encoder": ema_encoder.state_dict(), "denoiser": ema_denoiser.state_dict(),
                    "epoch": epoch + 1}, os.path.join(cfg.out_dir, "last.pth"))

        with open(os.path.join(cfg.out_dir, "history.json"), "w") as f:
            json.dump(history, f, indent=2)

    _plot_history(history, os.path.join(cfg.out_dir, "loss_curve.png"))
    print(f"Done. Best val Dice {best_val:.4f}. Artifacts in {cfg.out_dir}")


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
        ax2.set_ylabel("val Dice")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


if __name__ == "__main__":
    main()
