"""Analyze saved RadioDiff samples and create report-ready visual panels.

The script intentionally reads only the local RadioMapSeer archive and saved
PNG outputs. It does not require the model checkpoint, so it can be used for
post-hoc verification on a machine without a GPU.
"""
from __future__ import annotations

import argparse
import csv
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from skimage.metrics import peak_signal_noise_ratio, structural_similarity


def read_gray(path: Path, divisor: float) -> np.ndarray:
    return np.asarray(Image.open(path).convert("L"), dtype=np.float32) / divisor


def metrics(pred: np.ndarray, gt: np.ndarray) -> dict[str, float]:
    mse = float(np.mean((pred - gt) ** 2))
    return {
        "nmse": mse / float(np.mean(gt ** 2) + 1e-12),
        "rmse": float(np.sqrt(mse)),
        "ssim": float(structural_similarity(gt, pred, data_range=1.0)),
        "psnr": float(peak_signal_noise_ratio(gt, pred, data_range=1.0)),
    }


def save_panel(path: Path, panels: list[tuple[str, np.ndarray, str]], cols: int = 4) -> None:
    rows = int(np.ceil(len(panels) / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(3.1 * cols, 3.25 * rows))
    axes = np.atleast_1d(axes).ravel()
    for ax, (title, image, cmap) in zip(axes, panels):
        ax.imshow(np.clip(image, 0, 1), cmap=cmap, vmin=0, vmax=1)
        ax.set_title(title, fontsize=10)
        ax.axis("off")
    for ax in axes[len(panels):]:
        ax.axis("off")
    fig.tight_layout(pad=1.0)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--results", type=Path, default=Path("results"))
    parser.add_argument("--data", type=Path, default=Path("RadioDiff/data/archive"))
    parser.add_argument("--assets", type=Path, default=Path("report_assets"))
    args = parser.parse_args()

    gt_dir = args.data / "gain" / "carsDPM"
    rows: list[dict[str, object]] = []
    experiments = [p for p in sorted(args.results.iterdir()) if p.is_dir()]
    for exp in experiments:
        for pred_path in sorted(exp.glob("*.png")):
            gt_path = gt_dir / pred_path.name
            if not gt_path.exists():
                continue
            pred = read_gray(pred_path, 255.0)
            gt = read_gray(gt_path, 256.0)
            row = {"experiment": exp.name, "sample": pred_path.name}
            row.update(metrics(pred, gt))
            rows.append(row)

    csv_path = args.results / "per_sample_metrics.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["experiment", "sample", "nmse", "rmse", "ssim", "psnr"])
        writer.writeheader()
        writer.writerows(rows)

    # The best/worst examples are selected from the 5-step baseline by NMSE.
    base = [r for r in rows if r["experiment"] == "baseline_steps_5"]
    base.sort(key=lambda r: float(r["nmse"]))
    if base:
        best = str(base[0]["sample"])
        worst = str(base[-1]["sample"])
        for label, sample in [("best", best), ("worst", worst)]:
            pred = read_gray(args.results / "baseline_steps_5" / sample, 255.0)
            gt = read_gray(gt_dir / sample, 256.0)
            error = np.abs(pred - gt)
            save_panel(
                args.assets / f"qualitative_{label}.png",
                [("Ground truth", gt, "viridis"), ("Prediction", pred, "viridis"), ("Absolute error", error, "magma")],
                cols=3,
            )

        sample = worst
        panels = [("Ground truth", read_gray(gt_dir / sample, 256.0), "viridis")]
        for exp in ["baseline_steps_5", "baseline_steps_20", "baseline_steps_50", "seed_123"]:
            p = args.results / exp / sample
            if p.exists():
                panels.append((exp.replace("baseline_steps_", "steps=").replace("seed_123", "seed=123"), read_gray(p, 255.0), "viridis"))
        save_panel(args.assets / "steps_comparison.png", panels, cols=5)

    # Input channels for the best case: buildings, transmitter, and cars.
    if base:
        sample = best
        scene = sample.split("_")[0]
        building = read_gray(args.data / "png" / "buildings_complete" / f"{scene}.png", 255.0)
        tx = read_gray(args.data / "png" / "antennas" / sample, 255.0)
        cars = read_gray(args.data / "png" / "cars" / f"{scene}.png", 255.0)
        save_panel(args.assets / "input_condition.png", [("Buildings", building, "gray"), ("Transmitter", tx, "gray"), ("Cars", cars, "gray")], cols=3)

    print(f"saved {csv_path}")
    print(f"created report assets in {args.assets}")


if __name__ == "__main__":
    main()
