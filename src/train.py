"""Train the P2SDiff boundary-point diffusion model.

Run from the repository root:
    python -m src.train --dataset isic2017 --epochs 100 --batch_size 16

Saves EMA checkpoints + a loss curve + periodic validation visualizations under
`cfg.out_dir`.
"""

import argparse
import json
import math
import os
import traceback

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

from .config import Config
from .data import build_contour_dataset, split_counts
from .diffusion import GaussianDiffusion
from .models import ContourDenoiser, build_conditioner
from .utils import EMA


def str2bool(v):
    if isinstance(v, bool):
        return v
    v = str(v).lower()
    if v in ("1", "true", "t", "yes", "y"):
        return True
    if v in ("0", "false", "f", "no", "n"):
        return False
    raise argparse.ArgumentTypeError(f"Boolean expected, got {v!r}")


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


def _load_init_checkpoint(path, encoder, denoiser, device):
    """Load matching EMA weights for transfer; skip mismatched keys."""
    ckpt = torch.load(path, map_location=device, weights_only=False)
    for name, module in (("encoder", encoder), ("denoiser", denoiser)):
        if name not in ckpt:
            print(f"  warn: no '{name}' in {path}")
            continue
        missing, unexpected = module.load_state_dict(ckpt[name], strict=False)
        print(f"  loaded {name}: missing={len(missing)} unexpected={len(unexpected)}")


class WarmupCosineScheduler:
    """Linear warmup then cosine decay to min_lr (per-epoch steps)."""

    def __init__(self, optimizer, warmup_epochs, total_epochs, min_lr):
        self.optimizer = optimizer
        self.warmup_epochs = max(0, int(warmup_epochs))
        self.total_epochs = max(1, int(total_epochs))
        self.min_lr = float(min_lr)
        self.base_lrs = [g["lr"] for g in optimizer.param_groups]
        self.last_epoch = 0

    def step(self, epoch=None):
        if epoch is None:
            epoch = self.last_epoch + 1
        self.last_epoch = epoch
        for group, base in zip(self.optimizer.param_groups, self.base_lrs):
            if epoch <= self.warmup_epochs and self.warmup_epochs > 0:
                lr = base * epoch / self.warmup_epochs
            else:
                t = epoch - self.warmup_epochs
                t_max = max(1, self.total_epochs - self.warmup_epochs)
                cos = 0.5 * (1.0 + math.cos(math.pi * min(t, t_max) / t_max))
                lr = self.min_lr + (base - self.min_lr) * cos
            group["lr"] = lr

    def get_last_lr(self):
        return [g["lr"] for g in self.optimizer.param_groups]


def main():
    parser = argparse.ArgumentParser(description="Train P2SDiff")
    parser.add_argument("--dataset", choices=["ph2", "isic2017", "isic2018", "ham10000"])
    parser.add_argument("--skin_root", type=str)
    parser.add_argument("--out_dir", type=str)
    parser.add_argument("--epochs", type=int)
    parser.add_argument("--batch_size", type=int)
    parser.add_argument("--lr", type=float)
    parser.add_argument("--weight_decay", type=float)
    parser.add_argument("--n_points", type=int)
    parser.add_argument("--encoder", choices=["convnext", "pvt", "unet"])
    parser.add_argument("--backbone", type=str)
    parser.add_argument("--freeze_backbone", type=str2bool, default=None)
    parser.add_argument("--backbone_lr", type=float)
    parser.add_argument("--stem_dim", type=int)
    parser.add_argument("--coord_fourier_bands", type=int)
    parser.add_argument("--pos_grid_bands", type=int)
    parser.add_argument("--device", type=str)
    parser.add_argument("--num_workers", type=int)
    parser.add_argument("--eval_every", type=int)
    parser.add_argument("--aug_level", choices=["none", "light", "strong"])
    parser.add_argument("--guidance_scale", type=float)
    parser.add_argument("--ddim_steps", type=int)
    parser.add_argument("--lambda_dice", type=float)
    parser.add_argument("--lambda_uniformity", type=float)
    parser.add_argument("--soft_dice_size", type=int)
    parser.add_argument("--snr_gamma", type=float)
    parser.add_argument("--scheduler", choices=["none", "cosine"])
    parser.add_argument("--warmup_epochs", type=int)
    parser.add_argument("--min_lr", type=float)
    parser.add_argument("--init_checkpoint", type=str)
    parser.add_argument("--seed", type=int)
    parser.add_argument("--no_amp", action="store_true")
    args = parser.parse_args()

    cfg = Config.from_args(args)
    if args.no_amp:
        cfg.amp = False

    torch.manual_seed(cfg.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(cfg.seed)
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
    if cfg.init_checkpoint:
        print(f"Warm-start from {cfg.init_checkpoint}")
        _load_init_checkpoint(cfg.init_checkpoint, encoder, denoiser, device)

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
    scheduler = None
    if cfg.scheduler == "cosine":
        scheduler = WarmupCosineScheduler(
            optimizer, warmup_epochs=cfg.warmup_epochs,
            total_epochs=cfg.epochs, min_lr=cfg.min_lr,
        )
    trainable = head_params + backbone_params
    use_amp = cfg.amp and device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

    n_train = sum(p.numel() for p in trainable)
    n_total = sum(p.numel() for p in encoder.parameters()) + sum(p.numel() for p in denoiser.parameters())
    bb_state = "frozen" if not backbone_params else f"fine-tune @ lr {cfg.backbone_lr:g}"
    print(f"Encoder: {cfg.encoder}/{cfg.backbone} (backbone {bb_state}) | "
          f"trainable {n_train/1e6:.2f}M / total {n_total/1e6:.2f}M | "
          f"device {device} | amp {use_amp} | sched {cfg.scheduler}")

    history = {"epoch": [], "loss": [], "lr": [], "val_epoch": [], "val_dice": [], "val_iou": []}
    best_val = -1.0

    for epoch in range(cfg.epochs):
        if scheduler is not None:
            scheduler.step(epoch + 1)
        cur_lr = optimizer.param_groups[0]["lr"]

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
                    lambda_dice=cfg.lambda_dice, snr_gamma=cfg.snr_gamma,
                    soft_dice_size=cfg.soft_dice_size)

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
                              "unif": f"{parts['loss_uniformity'].item():.4f}",
                              "lr": f"{cur_lr:.2e}"})

        avg = running / max(1, len(train_loader))
        history["epoch"].append(epoch + 1)
        history["loss"].append(avg)
        history["lr"].append(cur_lr)
        print(f"Epoch {epoch+1} | train loss {avg:.4f} | lr {cur_lr:.2e}")

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
                    "epoch": epoch + 1,
                    "config": {k: getattr(cfg, k) for k in cfg.__dataclass_fields__},
                    }, os.path.join(cfg.out_dir, "last.pth"))

        with open(os.path.join(cfg.out_dir, "history.json"), "w") as f:
            json.dump(history, f, indent=2)

    _plot_history(history, os.path.join(cfg.out_dir, "loss_curve.png"))
    # Final test eval on best checkpoint if present.
    best_path = os.path.join(cfg.out_dir, "best.pth")
    test_metrics = {}
    if os.path.isfile(best_path):
        try:
            from .sample import evaluate, load_checkpoint
            test_ds = build_contour_dataset(cfg.skin_root, cfg.dataset, "te", cfg.n_points,
                                            cfg.img_size, augment=False, npy_size=cfg.npy_size)
            test_loader = DataLoader(test_ds, batch_size=cfg.batch_size, shuffle=False,
                                     num_workers=cfg.num_workers)
            cfg_eval = Config()
            ckpt = torch.load(best_path, map_location=device, weights_only=False)
            for k, v in (ckpt.get("config") or {}).items():
                if hasattr(cfg_eval, k):
                    setattr(cfg_eval, k, v)
            enc_t, den_t = load_checkpoint(best_path, cfg_eval, device)
            test_dice, test_iou = evaluate(
                enc_t, den_t, diffusion, test_loader, cfg_eval, device,
                viz_path=os.path.join(cfg.out_dir, "test_grid.png"),
            )
            test_metrics = {"test_dice": test_dice, "test_iou": test_iou}
            with open(os.path.join(cfg.out_dir, "test_metrics.json"), "w") as f:
                json.dump(test_metrics, f, indent=2)
            print(f"Test Dice {test_dice:.4f} | Test IoU {test_iou:.4f}")
        except Exception:
            print("Test eval failed:")
            traceback.print_exc()

    summary = {
        "out_dir": cfg.out_dir,
        "dataset": cfg.dataset,
        "best_val_dice": best_val,
        **test_metrics,
        "epochs": cfg.epochs,
        "lr": cfg.lr,
        "weight_decay": cfg.weight_decay,
        "scheduler": cfg.scheduler,
        "backbone": cfg.backbone,
        "guidance_scale": cfg.guidance_scale,
        "aug_level": cfg.aug_level,
        "freeze_backbone": cfg.freeze_backbone,
    }
    with open(os.path.join(cfg.out_dir, "summary.json"), "w") as f:
        json.dump(summary, f, indent=2)
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
