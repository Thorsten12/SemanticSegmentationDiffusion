#!/usr/bin/env python3
"""Vergleichs-Report fuer die drei ContourDenoiser-Sampler-Strategien
("input" / "deformable" / "attention"): Parameteranzahl, Forward-Pass-
Timing (CPU + optional GPU), und funktionale Sanity-Checks.

Nicht Teil der Trainings-Pipeline -- ein eigenstaendiges Diagnose-Skript,
das VOR einem echten Trainingslauf schnell beantwortet:
  1. Wie viele zusaetzliche Parameter kostet welche Variante?
  2. Wie viel langsamer ist ein Forward-Pass ggue. der Baseline?
  3. Reproduziert die "Nirgends deformen/attenden"-Konfiguration exakt die
     Baseline-Ausgabe (numerisch, bei fixem Seed und deaktiviertem Dropout)?
  4. Ist der Zero-Init tatsaechlich zero (Guidance-Beitrag der neuen Koepfe
     ist am Init vernachlaessigbar / die Fusionsgewichte sind uniform)?

Nutzung (im Projekt-Root, mit installiertem torch + x0_prediction_V6-Paket):
    python report_sampler_comparison.py \
        --n_points 100 --hidden_dim 128 --scale_channels 64 128 256 384 768 \
        --map_resolutions 224 56 28 14 7 --batch_size 8 \
        --deform_n_samples 4 --deform_min_scale_res 28 \
        --device cpu

Falls dein Repo die Klassen unter einem anderen Pfad hat, passe den Import
unten an (`from x0_prediction_V6.denoiser import ...`).
"""

import argparse
import os
import sys
import time
from contextlib import contextmanager

import torch

# Robuster Import: funktioniert sowohl wenn dieses Skript im Projekt-Root
# liegt (Paket x0_prediction_V6 daneben) als auch wenn es DIREKT in
# x0_prediction_V6/ liegt (dann muss das Elternverzeichnis auf sys.path,
# damit "x0_prediction_V6" als Package auffindbar ist).
_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_PARENT_DIR = os.path.dirname(_THIS_DIR)
for _p in (_THIS_DIR, _PARENT_DIR):
    if _p not in sys.path:
        sys.path.insert(0, _p)

try:
    from x0_prediction_V6.models import ContourDenoiser
except ModuleNotFoundError:
    # Fallback: Skript liegt selbst in x0_prediction_V6/, Package-Praefix
    # entfaellt dann.
    from .models import ContourDenoiser


# ---------------------------------------------------------------------------
# Hilfsfunktionen
# ---------------------------------------------------------------------------

def count_params(model: torch.nn.Module) -> int:
    return sum(p.numel() for p in model.parameters())


def count_sampler_params(model: ContourDenoiser) -> int:
    return sum(p.numel() for p in model.sampler.parameters())


@contextmanager
def timer():
    t0 = time.perf_counter()
    yield lambda: time.perf_counter() - t0


def build_model(sampler_type, args):
    return ContourDenoiser(
        n_points=args.n_points,
        hidden_dim=args.hidden_dim,
        num_layers=args.num_layers,
        num_heads=args.num_heads,
        attn_window=args.transformer_attn_window,
        scale_channels=tuple(args.scale_channels),
        proj_dim=args.proj_dim,
        coord_fourier_bands=args.coord_fourier_bands,
        timesteps=args.timesteps,
        residual_target="zero",
        sampler_type=sampler_type,
        deform_n_samples=args.deform_n_samples,
        deform_min_scale_res=args.deform_min_scale_res,
        deform_radius_cells=args.deform_radius_cells,
        attn_sampler_window=args.attn_sampler_window,
        attn_sampler_heads=args.attn_sampler_heads,
    )


def make_dummy_batch(args, device):
    b, n = args.batch_size, args.n_points
    points = torch.rand(b, n, 2, device=device) * 2 - 1  # [-1, 1]
    t = torch.randint(0, args.timesteps, (b,), device=device)
    maps = [
        torch.randn(b, c, r, r, device=device)
        for c, r in zip(args.scale_channels, args.map_resolutions)
    ]
    return points, t, maps


def bench_forward(model, points, t, maps, n_warmup=3, n_iters=10, device="cpu"):
    model.eval()
    with torch.no_grad():
        for _ in range(n_warmup):
            model(points, t, maps)
        if device == "cuda":
            torch.cuda.synchronize()
        with timer() as elapsed:
            for _ in range(n_iters):
                model(points, t, maps)
            if device == "cuda":
                torch.cuda.synchronize()
        total = elapsed()
    return total / n_iters


# ---------------------------------------------------------------------------
# Sanity Checks
# ---------------------------------------------------------------------------

def sanity_check_nowhere_matches_baseline(args, device):
    """Deformable/Attention mit deform_min_scale_res so hoch, dass NIRGENDS
    deformiert/attend-et wird, muss numerisch exakt die Baseline
    reproduzieren (gleicher Seed, eval-Modus, gleiche Eingaben)."""
    torch.manual_seed(0)
    baseline = build_model("input", args)

    args_nowhere = argparse.Namespace(**vars(args))
    args_nowhere.deform_min_scale_res = 10**6  # nirgends deformen/attenden

    torch.manual_seed(0)
    deform_nowhere = build_model("deformable", args_nowhere)
    torch.manual_seed(0)
    attn_nowhere = build_model("attention", args_nowhere)

    # Gemeinsame Gewichte fuer die geteilten Teile (time_mlp, coord_mlp, etc.)
    # koennen bei unabhaengiger Initialisierung leicht abweichen (RNG-Ziehungen
    # in anderer Reihenfolge, da die Sampler-Klassen unterschiedlich viele
    # Parameter VOR den gemeinsamen Layern ziehen). Fuer einen fairen
    # Vergleich kopieren wir stattdessen NUR den Sampler-Teil 1:1 vom
    # jeweils jetzt jungfraeulich gebauten Modell in eine Kopie der Baseline
    # mit exakt denselben Nicht-Sampler-Gewichten.
    def clone_with_sampler(base_model, donor_model):
        import copy
        m = copy.deepcopy(base_model)
        m.sampler = donor_model.sampler
        return m

    deform_nowhere_aligned = clone_with_sampler(baseline, deform_nowhere)
    attn_nowhere_aligned = clone_with_sampler(baseline, attn_nowhere)

    points, t, maps = make_dummy_batch(args, device)
    baseline.eval(); deform_nowhere_aligned.eval(); attn_nowhere_aligned.eval()

    with torch.no_grad():
        out_base = baseline(points, t, maps)
        out_deform = deform_nowhere_aligned(points, t, maps)
        out_attn = attn_nowhere_aligned(points, t, maps)

    diff_deform = (out_base - out_deform).abs().max().item()
    diff_attn = (out_base - out_attn).abs().max().item()

    return diff_deform, diff_attn


def sanity_check_zero_init_uniform_fusion(args, device):
    """Bei Zero-Init sollten:
      - Deformable: alle gelernten Offsets == 0 (nur aligned Sample zaehlt effektiv,
        Fusionsgewichte uniform durch Zero-Init der weight_heads).
      - Attention: Query-Projektion == 0 -> Attention-Logits == 0 -> Softmax
        exakt uniform ueber die K Kandidaten.
    Dies verifiziert direkt die Gewichte, nicht nur den End-zu-End-Output."""
    torch.manual_seed(0)
    deform = build_model("deformable", args)
    torch.manual_seed(0)
    attn = build_model("attention", args)

    # Deformable: offset_heads Gewichte/Bias muessen exakt 0 sein.
    max_offset_weight = max(
        oh.weight.abs().max().item() for oh in deform.sampler.offset_heads
    )
    max_offset_bias = max(
        oh.bias.abs().max().item() for oh in deform.sampler.offset_heads
    )

    # Attention: Query-Projektionen muessen exakt 0 sein -> Logits ueberall 0.
    max_query_weight = max(
        qh.weight.abs().max().item() for qh in attn.sampler.query_heads
    )
    max_query_bias = max(
        qh.bias.abs().max().item() for qh in attn.sampler.query_heads
    )

    return {
        "deform_max_offset_weight": max_offset_weight,
        "deform_max_offset_bias": max_offset_bias,
        "attn_max_query_weight": max_query_weight,
        "attn_max_query_bias": max_query_bias,
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--n_points", type=int, default=100)
    p.add_argument("--hidden_dim", type=int, default=128)
    p.add_argument("--num_layers", type=int, default=4)
    p.add_argument("--num_heads", type=int, default=4)
    p.add_argument("--transformer_attn_window", type=int, default=7)
    p.add_argument("--proj_dim", type=int, default=64)
    p.add_argument("--coord_fourier_bands", type=int, default=6)
    p.add_argument("--timesteps", type=int, default=1000)
    p.add_argument("--scale_channels", type=int, nargs="+", default=[64, 128, 256, 384, 768])
    p.add_argument("--map_resolutions", type=int, nargs="+", default=[224, 56, 28, 14, 7])
    p.add_argument("--batch_size", type=int, default=8)
    p.add_argument("--deform_n_samples", type=int, default=4)
    p.add_argument("--deform_min_scale_res", type=int, default=28)
    p.add_argument("--deform_radius_cells", type=float, default=2.5)
    p.add_argument("--attn_sampler_window", type=int, default=5)
    p.add_argument("--attn_sampler_heads", type=int, default=4)
    p.add_argument("--n_warmup", type=int, default=3)
    p.add_argument("--n_iters", type=int, default=10)
    p.add_argument("--device", type=str, default="cpu", choices=["cpu", "cuda"])
    args = p.parse_args()

    assert len(args.scale_channels) == len(args.map_resolutions), \
        "scale_channels und map_resolutions muessen gleich lang sein"

    device = args.device if (args.device == "cpu" or torch.cuda.is_available()) else "cpu"
    if args.device == "cuda" and device == "cpu":
        print("WARNUNG: CUDA angefragt aber nicht verfuegbar -> falle auf CPU zurueck.")

    print("=" * 70)
    print("1) PARAMETERANZAHL")
    print("=" * 70)
    rows = []
    for sampler_type in ("input", "deformable", "attention"):
        torch.manual_seed(0)
        model = build_model(sampler_type, args).to(device)
        total = count_params(model)
        sampler_only = count_sampler_params(model)
        rows.append((sampler_type, total, sampler_only))

    base_total = rows[0][1]
    print(f"{'sampler_type':<14}{'total params':>16}{'sampler params':>18}{'delta vs input':>18}")
    for name, total, sampler_only in rows:
        delta = total - base_total
        print(f"{name:<14}{total:>16,}{sampler_only:>18,}{delta:>+18,}")

    print()
    print("=" * 70)
    print("2) FORWARD-PASS TIMING (Sekunden/Iteration, Mittel ueber "
          f"{args.n_iters} Iterationen nach {args.n_warmup} Warmup)")
    print("=" * 70)
    timing_rows = []
    for sampler_type in ("input", "deformable", "attention"):
        torch.manual_seed(0)
        model = build_model(sampler_type, args).to(device)
        points, t, maps = make_dummy_batch(args, device)
        avg_s = bench_forward(model, points, t, maps,
                               n_warmup=args.n_warmup, n_iters=args.n_iters, device=device)
        timing_rows.append((sampler_type, avg_s))

    base_time = timing_rows[0][1]
    print(f"{'sampler_type':<14}{'sec/iter':>14}{'x vs input':>14}")
    for name, avg_s in timing_rows:
        print(f"{name:<14}{avg_s:>14.5f}{avg_s / base_time:>13.2f}x")

    print()
    print("=" * 70)
    print("3) SANITY CHECK: 'nirgends deformen/attenden' == Baseline?")
    print("=" * 70)
    diff_deform, diff_attn = sanity_check_nowhere_matches_baseline(args, device)
    tol = 1e-5
    ok_deform = diff_deform < tol
    ok_attn = diff_attn < tol
    print(f"deformable (min_scale_res=1e6) max|diff| vs baseline: {diff_deform:.3e} "
          f"-> {'PASS' if ok_deform else 'FAIL'}")
    print(f"attention  (min_scale_res=1e6) max|diff| vs baseline: {diff_attn:.3e} "
          f"-> {'PASS' if ok_attn else 'FAIL'}")
    if not (ok_deform and ok_attn):
        print("WARNUNG: mindestens ein Sampler reproduziert die Baseline im "
              "'nirgends'-Grenzfall nicht exakt -- Code-Pfad pruefen, bevor "
              "die Ablation als vertrauenswuerdig gilt.")

    print()
    print("=" * 70)
    print("4) SANITY CHECK: Zero-Init tatsaechlich zero / uniforme Fusion")
    print("=" * 70)
    zi = sanity_check_zero_init_uniform_fusion(args, device)
    for k, v in zi.items():
        status = "PASS" if v == 0.0 else "FAIL"
        print(f"{k:<32}{v:>12.3e}  -> {status}")

    print()
    print("Hinweis: Timing hier ist EIN Forward-Pass durch den vollen "
          "ContourDenoiser (inkl. Transformer), nicht nur den Sampler isoliert.")
    print("Fuer isoliertes Sampler-Timing: --n_points/--batch_size erhoehen "
          "und obige Tabelle 1/2 relativ zueinander lesen (Transformer-Anteil "
          "ist ueber alle drei Varianten identisch, die Differenz kommt "
          "ausschliesslich aus dem Sampler).")


if __name__ == "__main__":
    main()