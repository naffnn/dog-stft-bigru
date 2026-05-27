from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

REPRO_DIR = Path(__file__).resolve().parents[2]


def load_axis(path: Path, n_bins: int) -> np.ndarray:
    if path.exists():
        freqs = pd.read_csv(path)["frequency_hz"].to_numpy(dtype=float)
        if len(freqs) == n_bins:
            return freqs
    return np.linspace(0, 8000, n_bins)


def plot_heatmap(arr: np.ndarray, out_path: Path, title: str, freqs: np.ndarray, vmin: float, vmax: float) -> None:
    fig, ax = plt.subplots(figsize=(8, 4.8))
    extent = [0, 4.0, float(freqs.min()), float(freqs.max())]
    im = ax.imshow(arr, origin="lower", aspect="auto", cmap="magma", extent=extent, vmin=vmin, vmax=vmax)
    ax.set_title(title)
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Frequency (Hz)")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    fig.savefig(out_path, dpi=300)
    plt.close(fig)


def plot_overlay(stft: np.ndarray | None, heatmap: np.ndarray, out_path: Path, title: str, freqs: np.ndarray, vmin: float, vmax: float) -> None:
    fig, ax = plt.subplots(figsize=(8, 4.8))
    extent = [0, 4.0, float(freqs.min()), float(freqs.max())]
    if stft is not None:
        ax.imshow(stft, origin="lower", aspect="auto", cmap="gray_r", extent=extent)
    im = ax.imshow(heatmap, origin="lower", aspect="auto", cmap="magma", alpha=0.55, extent=extent, vmin=vmin, vmax=vmax)
    ax.set_title(title)
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Frequency (Hz)")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    fig.savefig(out_path, dpi=300)
    plt.close(fig)


def run(args: argparse.Namespace) -> None:
    heatmap_paths = sorted(args.heatmap_dir.rglob("*_heatmap.npy"))
    if not heatmap_paths:
        raise FileNotFoundError(f"No heatmap npy files found under {args.heatmap_dir}")
    values = np.concatenate([np.load(path).reshape(-1) for path in heatmap_paths])
    vmin = float(np.percentile(values, args.percentile_low))
    vmax = float(np.percentile(values, args.percentile_high))
    if vmax <= vmin:
        vmax = float(values.max())

    args.output_dir.mkdir(parents=True, exist_ok=True)
    reports_dir = args.output_dir.parent / "reports"
    reports_dir.mkdir(parents=True, exist_ok=True)
    first = np.load(heatmap_paths[0])
    freqs = load_axis(args.frequency_axis_csv, first.shape[0])

    for heatmap_path in heatmap_paths:
        rel = heatmap_path.relative_to(args.heatmap_dir)
        out_subdir = args.output_dir / rel.parent
        out_subdir.mkdir(parents=True, exist_ok=True)
        heatmap = np.load(heatmap_path)
        stft_path = Path(str(heatmap_path).replace("_heatmap.npy", "_stft.npy"))
        stft = np.load(stft_path) if stft_path.exists() else None
        stem = heatmap_path.stem.replace("_heatmap", "")
        plot_heatmap(heatmap, out_subdir / f"{stem}_unified_heatmap.png", stem, freqs, vmin, vmax)
        plot_overlay(stft, heatmap, out_subdir / f"{stem}_unified_overlay.png", stem, freqs, vmin, vmax)

    config = {
        "vmin": vmin,
        "vmax": vmax,
        "percentile_low": args.percentile_low,
        "percentile_high": args.percentile_high,
        "num_samples": len(heatmap_paths),
    }
    (reports_dir / "unified_colorbar_config.json").write_text(json.dumps(config, ensure_ascii=False, indent=2), encoding="utf-8")
    summary_md = reports_dir / "explain_summary.md"
    if summary_md.exists():
        with summary_md.open("a", encoding="utf-8") as f:
            f.write("\n## Unified Colorbar\n\n")
            f.write(f"- vmin: `{vmin}`\n")
            f.write(f"- vmax: `{vmax}`\n")
            f.write(f"- percentile_low: `{args.percentile_low}`\n")
            f.write(f"- percentile_high: `{args.percentile_high}`\n")
            f.write(f"- num_samples: `{len(heatmap_paths)}`\n")
            f.write("- config: `outputs/reports/unified_colorbar_config.json`\n")
    print(f"Saved unified heatmaps to: {args.output_dir}")
    print(f"Saved config: {reports_dir / 'unified_colorbar_config.json'}")


def parse_args() -> argparse.Namespace:
    base = REPRO_DIR / "explain_analysis" / "outputs"
    parser = argparse.ArgumentParser(description="Redraw saliency heatmaps with a unified colorbar.")
    parser.add_argument("--heatmap_dir", type=Path, default=base / "heatmaps_raw")
    parser.add_argument("--output_dir", type=Path, default=base / "heatmaps_unified")
    parser.add_argument("--frequency_axis_csv", type=Path, default=base / "reports" / "frequency_axis.csv")
    parser.add_argument("--percentile_low", type=float, default=1.0)
    parser.add_argument("--percentile_high", type=float, default=99.0)
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
