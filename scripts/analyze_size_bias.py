"""Analyze Dice versus lesion size and border contact from evaluation CSV.

Example:
    python scripts/analyze_size_bias.py \
      src/runs/ph2_v5_fourier_residual/test_case_metrics.csv
"""

import argparse
import csv
import math
from pathlib import Path


def mean(values):
    return sum(values) / len(values)


def pearson(x, y):
    if len(x) < 2:
        return float("nan")
    mx, my = mean(x), mean(y)
    dx, dy = [v - mx for v in x], [v - my for v in y]
    denom = math.sqrt(sum(v * v for v in dx) * sum(v * v for v in dy))
    return sum(a * b for a, b in zip(dx, dy)) / denom if denom else float("nan")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("metrics", type=Path)
    parser.add_argument("--out", type=Path, default=None)
    parser.add_argument("--no_plot", action="store_true")
    args = parser.parse_args()

    with args.metrics.open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        raise ValueError(f"No cases in {args.metrics}")

    area = [float(r["gt_area_fraction"]) for r in rows]
    border = [bool(int(r["gt_touches_border"])) for r in rows]
    dice = [float(r["dice"]) for r in rows]
    proposal = [float(r["proposal_dice"]) for r in rows]
    coarse = [float(r["coarse_dice"]) for r in rows]

    corr = pearson(area, dice)
    print(f"cases: {len(rows)}")
    print(f"Pearson(area, final Dice): {corr:.4f}")
    for name, wanted in (("non-border", False), ("border-touch", True)):
        sel = [i for i, value in enumerate(border) if value == wanted]
        if sel:
            print(
                f"{name}: n={len(sel)} | proposal={mean([proposal[i] for i in sel]):.4f} "
                f"| coarse={mean([coarse[i] for i in sel]):.4f} "
                f"| final={mean([dice[i] for i in sel]):.4f}"
            )

    if args.no_plot:
        return

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    out = args.out or args.metrics.with_name(args.metrics.stem + "_area_bias.png")
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    colors = ["tab:red" if value else "tab:blue" for value in border]
    axes[0].scatter(area, dice, c=colors, alpha=0.8, edgecolors="none")
    if len(rows) > 1 and max(area) > min(area):
        mx, my = mean(area), mean(dice)
        slope = sum((x - mx) * (y - my) for x, y in zip(area, dice)) / sum(
            (x - mx) ** 2 for x in area
        )
        intercept = my - slope * mx
        line_x = [min(area), max(area)]
        axes[0].plot(line_x, [slope * x + intercept for x in line_x],
                     "k--", linewidth=1)
    axes[0].set(xlabel="GT lesion area / image area", ylabel="Final Dice",
                title=f"Size bias (r={corr:.3f})")
    axes[0].grid(alpha=0.25)

    axes[1].scatter(proposal, dice, c=colors, alpha=0.8, edgecolors="none")
    axes[1].plot([0, 1], [0, 1], "k--", linewidth=1)
    axes[1].set(xlim=(0, 1), ylim=(0, 1), xlabel="Proposal Dice",
                ylabel="Final Dice", title="Proposal quality vs final result")
    axes[1].grid(alpha=0.25)
    fig.suptitle("Blue: non-border | Red: border-touch")
    fig.tight_layout()
    fig.savefig(out, dpi=160)
    plt.close(fig)
    print(f"plot: {out}")


if __name__ == "__main__":
    main()
