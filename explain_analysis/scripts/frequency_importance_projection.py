from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

REPRO_DIR = Path(__file__).resolve().parents[2]
LABELS = ["angry", "anxious", "happy", "lonely", "sad"]
COLORS = {"angry": "#d62728", "anxious": "#9467bd", "happy": "#2ca02c", "lonely": "#1f77b4", "sad": "#7f7f7f"}


def curve_from_heatmap(path: Path, n_freq: int) -> np.ndarray:
    arr = np.load(path)
    if arr.shape[0] == n_freq:
        return arr.mean(axis=1)
    if arr.shape[1] == n_freq:
        return arr.mean(axis=0)
    raise RuntimeError(f"Cannot infer frequency axis for {path}: shape={arr.shape}, n_freq={n_freq}")


def run(args: argparse.Namespace) -> None:
    args.output_dir.mkdir(parents=True, exist_ok=True)
    index_df = pd.read_csv(args.index_csv)
    freq_df = pd.read_csv(args.frequency_axis_csv)
    freqs = freq_df["frequency_hz"].to_numpy(dtype=float)
    rows = []
    for row in index_df.itertuples(index=False):
        curve = curve_from_heatmap(Path(row.heatmap_npy_path), len(freqs))
        for freq, value in zip(freqs, curve):
            rows.append({"sample_id": row.sample_id, "emotion": row.true_label, "frequency_hz": freq, "importance": float(value)})
    long_df = pd.DataFrame(rows)
    summary = (
        long_df.groupby(["emotion", "frequency_hz"])["importance"]
        .agg(mean_importance="mean", std_importance="std", n_samples="count")
        .reset_index()
    )
    summary["std_importance"] = summary["std_importance"].fillna(0.0)
    out_csv = args.output_dir / "frequency_importance_by_class.csv"
    summary.to_csv(out_csv, index=False, encoding="utf-8-sig")

    fig, ax = plt.subplots(figsize=(9, 5))
    for label in LABELS:
        label_df = summary[summary["emotion"] == label]
        if label_df.empty:
            continue
        ax.plot(label_df["frequency_hz"], label_df["mean_importance"], label=label, color=COLORS[label])
    ax.set_title("Frequency Importance by Class")
    ax.set_xlabel("Frequency (Hz)")
    ax.set_ylabel("Mean Importance")
    ax.legend()
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(args.output_dir / "frequency_importance_curves.png", dpi=300)
    plt.close(fig)
    print(f"Saved frequency projection: {out_csv}")


def parse_args() -> argparse.Namespace:
    base = REPRO_DIR / "explain_analysis" / "outputs"
    parser = argparse.ArgumentParser(description="Project saliency heatmaps to frequency importance curves.")
    parser.add_argument("--index_csv", type=Path, default=base / "reports" / "heatmap_index.csv")
    parser.add_argument("--frequency_axis_csv", type=Path, default=base / "reports" / "frequency_axis.csv")
    parser.add_argument("--output_dir", type=Path, default=base / "frequency_projection")
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
