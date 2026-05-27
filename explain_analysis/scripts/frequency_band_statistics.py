from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

REPRO_DIR = Path(__file__).resolve().parents[2]
LABELS = ["angry", "anxious", "happy", "lonely", "sad"]
BANDS = {"low": (0.0, 1000.0), "mid": (1000.0, 3000.0), "high": (3000.0, 8000.0)}
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
    sample_rows = []
    for row in index_df.itertuples(index=False):
        curve = curve_from_heatmap(Path(row.heatmap_npy_path), len(freqs))
        for band, (lo, hi) in BANDS.items():
            mask = (freqs >= lo) & (freqs < hi if hi < freqs.max() else freqs <= hi)
            sample_rows.append(
                {
                    "sample_id": row.sample_id,
                    "emotion": row.true_label,
                    "band": band,
                    "band_low_hz": lo,
                    "band_high_hz": hi,
                    "mean_importance": float(curve[mask].mean()) if mask.any() else 0.0,
                }
            )
    sample_df = pd.DataFrame(sample_rows)
    summary = (
        sample_df.groupby(["emotion", "band", "band_low_hz", "band_high_hz"])["mean_importance"]
        .agg(mean_importance="mean", std_importance="std", n_samples="count")
        .reset_index()
    )
    summary["std_importance"] = summary["std_importance"].fillna(0.0)
    summary.to_csv(args.output_dir / "frequency_band_importance.csv", index=False, encoding="utf-8-sig")

    fig, ax = plt.subplots(figsize=(9, 5))
    x = np.arange(len(BANDS))
    width = 0.14
    for i, label in enumerate(LABELS):
        vals = [summary[(summary["emotion"] == label) & (summary["band"] == band)]["mean_importance"].mean() for band in BANDS]
        ax.bar(x + (i - 2) * width, vals, width=width, label=label, color=COLORS[label])
    ax.set_xticks(x)
    ax.set_xticklabels(list(BANDS.keys()))
    ax.set_ylabel("Mean Importance")
    ax.set_title("Frequency Band Importance")
    ax.legend()
    fig.tight_layout()
    fig.savefig(args.output_dir / "frequency_band_barplot.png", dpi=300)
    plt.close(fig)

    pivot = summary.pivot(index="emotion", columns="band", values="mean_importance").reindex(index=LABELS, columns=list(BANDS.keys()))
    fig, ax = plt.subplots(figsize=(6, 4.5))
    im = ax.imshow(pivot.to_numpy(), cmap="magma", aspect="auto")
    ax.set_xticks(range(len(BANDS)))
    ax.set_xticklabels(list(BANDS.keys()))
    ax.set_yticks(range(len(LABELS)))
    ax.set_yticklabels(LABELS)
    ax.set_title("Frequency Band Importance Heatmap")
    for i in range(len(LABELS)):
        for j in range(len(BANDS)):
            ax.text(j, i, f"{pivot.iloc[i, j]:.3g}", ha="center", va="center", color="white")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    fig.savefig(args.output_dir / "frequency_band_heatmap.png", dpi=300)
    plt.close(fig)

    reports_dir = args.output_dir.parent / "reports"
    summary_md = reports_dir / "explain_summary.md"
    if summary_md.exists():
        with summary_md.open("a", encoding="utf-8") as f:
            f.write("\n## Frequency Importance Projection\n\n")
            f.write("- `outputs/frequency_projection/frequency_importance_curves.png`\n\n")
            f.write("## Low/Mid/High Band Statistics\n\n")
            f.write("- `outputs/band_statistics/frequency_band_barplot.png`\n")
            f.write("- `outputs/band_statistics/frequency_band_heatmap.png`\n")
    print(f"Saved band statistics to: {args.output_dir}")


def parse_args() -> argparse.Namespace:
    base = REPRO_DIR / "explain_analysis" / "outputs"
    parser = argparse.ArgumentParser(description="Summarize saliency importance in low/mid/high frequency bands.")
    parser.add_argument("--index_csv", type=Path, default=base / "reports" / "heatmap_index.csv")
    parser.add_argument("--frequency_axis_csv", type=Path, default=base / "reports" / "frequency_axis.csv")
    parser.add_argument("--output_dir", type=Path, default=base / "band_statistics")
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
