import argparse
import time
import torch
import os
import json

from torch.utils.data import DataLoader
import numpy as np
from sklearn.model_selection import train_test_split
import torch.nn.functional as F

from ema_pytorch import EMA

from .models import build_conditioner, ContourDenoiser
from .data import ArrayContourDataset
from .diffusion import GaussianDiffusion

from .visualisation import visualize_predictions, evaluate_dice
from .data_split import make_splits

def build_models(args):
    encoder = build_conditioner(cfg=args)
    denoiser = ContourDenoiser(
        n_point=args.n_points,
        hidden_dim=args.hidden_dim,
        coord_fourier_bands=args.coord_fourier_bands,
        feature_channels=encoder.feature_channels,
    )
    return encoder, denoiser

def count_params(params):
    return sum(p.numel() for p in params if p.requires_grad)

def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ("yes", "true", "t", "y", "1"):
        return True
    elif v.lower() in ("no", "false", "f", "n", "0"):
        return False
    else:
        raise argparse.ArgumentTypeError(f"Boolean value expected, got: {v!r}")

def main():
    parser = argparse.ArgumentParser(description="Train a semantic segmentation model with diffusion.")

    parser.add_argument("--n_points", type=int, default=200)
    parser.add_argument("--hidden_dim", type=int, default=128)
    parser.add_argument("--coord_fourier_bands", type=int, default=12)

    parser.add_argument("--backbone", type=str, default="convnext_tiny")
    parser.add_argument("--encoder", type=str, default="convnext")
    parser.add_argument("--freeze", type=str2bool, default=True)

    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--learning_rate", type=float, default=1e-4)
    parser.add_argument("--num_epochs", type=int, default=800)
    parser.add_argument("--out_dir", type=str, default="./new_runs/output")
    parser.add_argument("--data", type=str, default="./data/datasets/PH2/np")

    parser.add_argument("--timesteps", type=int, default=1000)
    parser.add_argument("--beta_start", type=float, default=1e-4)
    parser.add_argument("--beta_end", type=float, default=0.02)
    parser.add_argument("--guidance", type=float, default=2.5)
    parser.add_argument("--boundary_gate_ratio", type=float, default=0.4)

    # --- EMA ---
    parser.add_argument("--ema_beta", type=float, default=0.9999)
    parser.add_argument("--ema_update_after_step", type=int, default=100)
    parser.add_argument("--ema_update_every", type=int, default=10)
    parser.add_argument("--use_ema_for_vis", type=str2bool, default=True)

    # --- Loss --
    parser.add_argument("--lambda_uniformity_max", type=float, default=0.1)
    parser.add_argument("--lamda_mse", type=float, default=5.0)
    parser.add_argument("--lamda_boundary_max", type=float, default=2.0)
    parser.add_argument("--lamda_dice_max", type=float, default=5.0)
    parser.add_argument("--use_uncertainty_weighting", type=str2bool, default=True)

    # --- LastVit/DinoV3 ---
    parser.add_argument("--layers", type=int, nargs="+", default=[2, 5, 7, 11])
    parser.add_argument("--dino_model_name", type=str, default="dinov3_vits16")

    # --- Data ---
    parser.add_argument("--random_state", type=int, default=42)
    parser.add_argument("--val_size", type=float, default=0.2)
    parser.add_argument("--test_size", type=float, default=0.2)

    # --- FineTunen ---
    parser.add_argument("--init_checkpoint", type=str, default=None)
    parser.add_argument("--init_from_ema", type=str2bool, default=True)

    # --- MECA ---
    parser.add_argument("--use_meca", type=str2bool, default=True)

    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    os.makedirs(args.out_dir, exist_ok=True)
    with open(os.path.join(args.out_dir, "config.json"), "w") as f:
        json.dump(vars(args), f, indent=2, default=str)

    X_path = os.path.join(args.data, "X_tr_224x224.npy")
    Y_path = os.path.join(args.data, "Y_tr_224x224.npy")

    print(f"Lade X von: {os.path.abspath(X_path)}")
    print(f"Lade Y von: {os.path.abspath(Y_path)}")

    X_full = np.load(X_path)
    Y_full = np.load(Y_path)

    (X_train, Y_train), (X_val, Y_val), (X_test, Y_test) = make_splits(
        X_full, Y_full,
        random_state=args.random_state,
        val_size=args.val_size,
        test_size=args.test_size,
    )

    train_dataset = ArrayContourDataset(
        images=X_train, masks=Y_train, n_points=args.n_points, img_size=(224, 224)
    )
    val_dataset = ArrayContourDataset(
        images=X_val, masks=Y_val, n_points=args.n_points, img_size=(224, 224)
    )

    train_loader = DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers
    )
    val_loader = DataLoader(
        val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers
    )

    encoder, denoiser = build_models(args)
    encoder.to(device)
    denoiser.to(device)

    if args.init_checkpoint is not None:
        ckpt = torch.load(args.init_checkpoint, map_location=device)
        if args.init_from_ema:
            ema_sd = ckpt["ema_state_dict"]
            weights = {k.replace("ema_model.", ""): v for k, v in ema_sd.items()
                       if k.startswith("ema_model.")}
        else:
            weights = ckpt["denoiser_state_dict"]
        missing, unexpected = denoiser.load_state_dict(weights, strict=False)
        print(f"[Finetune-Init] Denoiser-Gewichte geladen aus {args.init_checkpoint} "
              f"| missing={len(missing)} unexpected={len(unexpected)}")

    # --- EMA-Wrapper um den Denoiser ---
    # ema.ema_model ist die geglättete Kopie, die für Visualisierung/Inference genutzt wird.
    ema = EMA(
        denoiser,
        beta=args.ema_beta,
        update_after_step=0 if args.init_checkpoint is not None else args.ema_update_after_step,
        update_every=args.ema_update_every,
    ).to(device)

    diffusion = GaussianDiffusion(
        timesteps=args.timesteps, beta_start=args.beta_start, beta_end=args.beta_end, device=device,
        use_uncertainty_weighting=args.use_uncertainty_weighting,
    )

    # --- Sigma-Parameter nur einsammeln, wenn Uncertainty Weighting aktiv ist ---
    # (bei use_uncertainty_weighting=False sind diese None und dürfen weder an
    # den Optimizer noch in count_params gehen)
    if args.use_uncertainty_weighting:
        sigma_params = [
            diffusion.log_sigma_uniformity, diffusion.log_sigma_boundary, diffusion.log_sigma_dice,
        ]
    else:
        sigma_params = []

    # --- Trainierbare Parameter ---
    # Bei freeze=True bleibt das Backbone selbst eingefroren, aber MECA (Channel Attention)
    # und die Fusion (_PerPointFusion, Zeit-Gewichtung der Skalenstufen) sollen weiterhin
    # trainiert werden -- beide sind eigene, kleine Zusatzmodule, keine Backbone-Gewichte.
    if args.freeze:
        encoder.eval()
        meca_params = list(encoder.mecas.parameters()) if encoder.mecas is not None else []
        fusion_params = list(encoder.fusion.parameters())
        trainable_params = list(denoiser.parameters()) + meca_params + fusion_params + sigma_params
    else:
        trainable_params = (
            list(encoder.parameters()) + list(denoiser.parameters()) + sigma_params
        )

    optimizer = torch.optim.AdamW(trainable_params, lr=args.learning_rate)
    num_trainable = sum(p.numel() for p in trainable_params if p.requires_grad)

    print(f"Denoiser Parameter: {count_params(denoiser.parameters()):,}")
    if not args.freeze:
        print(f"Encoder Parameter: {count_params(encoder.parameters()):,}")
    else:
        if encoder.mecas is not None:
            print(f"MECA Parameter: {count_params(meca_params):,}")
        else:
            print("MECA: deaktiviert")
        print(f"Fusion Parameter: {count_params(fusion_params):,}")

    use_cuda_timing = device.type == "cuda"

    # --- Bestes Modell über den kompletten Val-Set-Dice-Score tracken ---
    best_dice = -1.0
    best_dice_epoch = -1

    for epoch in range(args.num_epochs):
        denoiser.train()

        total_loss = 0.0
        loss_uniformity = 0.0
        loss_mse = 0.0
        loss_boundary = 0.0
        loss_dice = 0.0

        sigma_uniformity = 0.0
        sigma_boundary   = 0.0
        sigma_dice       = 0.0
        n_active         = 0.0

        # --- Zeitmessung: Epoche gesamt + isoliert training_loss ---
        epoch_start = time.perf_counter()
        loss_time_accum = 0.0  # Sekunden, nur der training_loss-Call

        for batch_idx, (images, points, masks) in enumerate(train_loader):
            images = images.to(device)
            points = points.to(device)
            masks = masks.to(device)

            optimizer.zero_grad()

            b_size = images.shape[0]
            t = torch.randint(0, args.timesteps, (b_size,), device=device).long()

            # extract() schützt das Backbone bereits intern per no_grad() (self.freeze),
            # MECA läuft dort bewusst AUSSERHALB von no_grad() -> bleibt trainierbar.
            # fuse() (=_PerPointFusion) soll ebenfalls immer trainierbar sein.
            # Kein äußeres set_grad_enabled(not args.freeze) mehr -- das hätte MECA
            # und die Fusion mit-eingefroren, obwohl sie im Optimizer stehen.
            feats = encoder.extract(images)
            cond = encoder.fuse(feats, t)

            noise = torch.randn_like(points)
            noise = torch.clamp(noise, -3.0, 3.0)

            x_t = diffusion.q_sample(x_0=points, timestep=t, noise=noise)

            drop_mask = torch.rand(b_size, device=device) < 0.10
            predicted_noise = denoiser(x=x_t, t=t, cond=cond, drop_mask=drop_mask)
            
            if use_cuda_timing:
                torch.cuda.synchronize()
            loss_call_start = time.perf_counter()

            # use_uncertainty_weighting steuert die Mechanik jetzt zentral im
            # Konstruktor von GaussianDiffusion (self.use_uncertainty_weighting) -
            # training_loss() selbst nimmt dieses Flag nicht entgegen.
            loss, loss_dict = diffusion.training_loss(
                predicted_noise=predicted_noise,
                true_noise=noise,
                x_t=x_t,
                x_0=points,
                t=t,
                masks=masks,
                lambda_uniformity_max=args.lambda_uniformity_max,
                lamda_mse=args.lamda_mse,
                lamda_boundary_max=args.lamda_boundary_max,
                lamda_dice_max=args.lamda_dice_max,
                boundary_gate_ratio=args.boundary_gate_ratio,
            )

            if use_cuda_timing:
                torch.cuda.synchronize()
            loss_time_accum += time.perf_counter() - loss_call_start

            loss.backward()
            optimizer.step()

            # --- EMA-Update nach jedem Optimizer-Step ---
            ema.update()

            total_loss += loss.item()
            loss_uniformity += loss_dict.get("loss_uniformity").item()
            loss_mse += loss_dict.get("loss_mse").item()
            loss_boundary += loss_dict.get("loss_boundary").item()
            loss_dice += loss_dict.get("loss_dice").item()

            # sigma_* sind bei use_uncertainty_weighting=False None -> auf 0
            # zurückfallen statt .item() auf None aufzurufen
            sig_uni = loss_dict.get("sigma_uniformity")
            sig_bnd = loss_dict.get("sigma_boundary")
            sig_dic = loss_dict.get("sigma_dice")
            sigma_uniformity += sig_uni.item() if sig_uni is not None else 0.0
            sigma_boundary   += sig_bnd.item() if sig_bnd is not None else 0.0
            sigma_dice       += sig_dic.item() if sig_dic is not None else 0.0
            n_active         += loss_dict.get("n_active_boundary_dice")

        if use_cuda_timing:
            torch.cuda.synchronize()
        epoch_time = time.perf_counter() - epoch_start

        avg_loss            = total_loss / len(train_loader)
        avg_loss_uniformity = loss_uniformity / len(train_loader)
        avg_loss_mse        = loss_mse / len(train_loader)
        avg_loss_boundary   = loss_boundary / len(train_loader)
        avg_loss_dice       = loss_dice / len(train_loader)

        avg_sigma_uniformity = sigma_uniformity / len(train_loader)
        avg_sigma_boundary   = sigma_boundary / len(train_loader)
        avg_sigma_dice       = sigma_dice / len(train_loader)

        avg_n_active         = n_active / len(train_loader)

        loss_time_pct = 100.0 * loss_time_accum / epoch_time if epoch_time > 0 else 0.0

        print(f"Epoch [{epoch+1}/{args.num_epochs}] | Training Loss: {avg_loss:.6f} | "
              f"Uniformity: {avg_loss_uniformity:.6f} | MSE: {avg_loss_mse:.6f} | "
              f"Boundary: {avg_loss_boundary:.6f} | Dice: {avg_loss_dice:.6f} | "
              f"σ_uni: {avg_sigma_uniformity:.4f} | σ_bnd: {avg_sigma_boundary:.4f} | σ_dice: {avg_sigma_dice:.4f} "
              f"n_active: {avg_n_active:.4f} | "
              f"epoch_time: {epoch_time:.2f}s | loss_fn_time: {loss_time_accum:.2f}s ({loss_time_pct:.1f}%)")

        if (epoch + 1) % 20 == 0:
            vis_denoiser = ema.ema_model if args.use_ema_for_vis else denoiser
            visualize_predictions(
                denoiser=vis_denoiser,
                encoder=encoder,
                diffusion=diffusion,
                dataset=val_dataset,
                device=device,
                n_points=args.n_points,
                img_size=(224, 224),
                n_samples=4,
                out_path=os.path.join(args.out_dir, f"vis_epoch_{epoch+1}.png"),
                guidance=args.guidance,
            )

            # --- 20er-Checkpoint speichern (inkl. EMA-Wrapper, wie von ema-pytorch empfohlen) ---
            checkpoint_payload = {
                "epoch": epoch + 1,
                "denoiser_state_dict": denoiser.state_dict(),
                "ema_state_dict": ema.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
            }
            torch.save(
                checkpoint_payload,
                os.path.join(args.out_dir, f"checkpoint_epoch_{epoch+1}.pt"),
            )

            # --- Dice-Score über das GESAMTE Validierungsset (nicht nur die Plot-Samples) ---
            mean_dice, mean_iou = evaluate_dice(
                denoiser=vis_denoiser,
                encoder=encoder,
                diffusion=diffusion,
                dataset=val_dataset,
                device=device,
                n_points=args.n_points,
                img_size=(224, 224),
                guidance=args.guidance,
            )
            print(f"[Eval @ Epoch {epoch+1}] Mean Dice (val, n={len(val_dataset)}): "
                  f"{mean_dice:.4f} | Mean IoU: {mean_iou:.4f}")

            if mean_dice > best_dice:
                best_dice = mean_dice
                best_dice_epoch = epoch + 1
                best_payload = dict(checkpoint_payload)
                best_payload["val_mean_dice"] = mean_dice
                best_payload["val_mean_iou"] = mean_iou
                torch.save(best_payload, os.path.join(args.out_dir, "best.pth"))
                print(f"[Best] Neuer bester Dice-Score: {best_dice:.4f} (Epoch {best_dice_epoch}) "
                      f"-> gespeichert als best.pth")

if __name__ == "__main__":
    main()