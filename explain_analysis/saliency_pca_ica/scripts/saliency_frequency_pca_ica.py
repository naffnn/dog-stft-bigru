from __future__ import annotations

import argparse
import json
import sys
import warnings
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.decomposition import FastICA, PCA
from sklearn.exceptions import ConvergenceWarning
from sklearn.preprocessing import StandardScaler

SCRIPT_PATH = Path(__file__).resolve()
EXPLAIN_DIR = SCRIPT_PATH.parents[2]
REPRO_DIR = EXPLAIN_DIR.parent
LABELS = ["angry", "anxious", "happy", "lonely", "sad"]
COLORS = {"angry": "#d62728", "anxious": "#9467bd", "happy": "#2ca02c", "lonely": "#1f77b4", "sad": "#7f7f7f"}
BANDS = {"low": (0.0, 1000.0), "mid": (1000.0, 3000.0), "high": (3000.0, 8000.0)}


def str_to_bool(value: str | bool) -> bool:
    if isinstance(value, bool):
        return value
    value = value.lower().strip()
    if value in {"true", "1", "yes", "y"}:
        return True
    if value in {"false", "0", "no", "n"}:
        return False
    raise argparse.ArgumentTypeError(f"Expected true/false, got {value}")


def make_dirs(output_dir: Path) -> dict[str, Path]:
    dirs = {
        "features": output_dir / "features",
        "pca_figures": output_dir / "figures" / "pca",
        "ica_figures": output_dir / "figures" / "ica",
        "loadings": output_dir / "figures" / "component_loadings",
        "boxplots": output_dir / "figures" / "boxplots",
        "reports": output_dir / "reports",
    }
    for path in dirs.values():
        path.mkdir(parents=True, exist_ok=True)
    return dirs


def curve_from_heatmap(path: Path, n_freq: int) -> np.ndarray:
    arr = np.load(path)
    if arr.shape[0] == n_freq:
        return arr.mean(axis=1)
    if arr.shape[1] == n_freq:
        return arr.mean(axis=0)
    raise RuntimeError(f"Cannot infer frequency/time dimensions for {path}: shape={arr.shape}, n_freq={n_freq}")


def load_curves(args: argparse.Namespace, dirs: dict[str, Path]) -> tuple[np.ndarray, pd.DataFrame, np.ndarray]:
    index_df = pd.read_csv(args.index_csv)
    if args.use_only_correct:
        index_df = index_df[index_df["correct"].astype(str).str.lower().isin(["true", "1"])]
    freq_df = pd.read_csv(args.frequency_axis_csv)
    freqs = freq_df["frequency_hz"].to_numpy(dtype=float)
    curves = []
    meta_rows = []
    for row in index_df.itertuples(index=False):
        curve = curve_from_heatmap(Path(row.heatmap_npy_path), len(freqs))
        curves.append(curve)
        meta_rows.append(
            {
                "sample_id": row.sample_id,
                "file_path": row.file_path,
                "emotion": row.true_label,
                "pred_label": row.pred_label,
                "confidence": float(row.confidence),
                "correct": bool(row.correct),
            }
        )
    if not curves:
        raise RuntimeError("No saliency curves available after filtering.")
    X = np.stack(curves).astype(np.float32)
    meta_df = pd.DataFrame(meta_rows)

    wide = meta_df.copy()
    curve_df = pd.DataFrame(X, columns=[f"freq_{freq:.2f}Hz" for freq in freqs])
    wide = pd.concat([wide, curve_df], axis=1)
    wide.to_csv(dirs["features"] / "saliency_frequency_curves.csv", index=False, encoding="utf-8-sig")

    long_rows = []
    for sample_i, meta in meta_df.iterrows():
        for freq_i, freq in enumerate(freqs):
            long_rows.append(
                {
                    "sample_id": meta["sample_id"],
                    "file_path": meta["file_path"],
                    "emotion": meta["emotion"],
                    "frequency_hz": freq,
                    "importance": float(X[sample_i, freq_i]),
                }
            )
    pd.DataFrame(long_rows).to_csv(dirs["features"] / "saliency_frequency_curves_long.csv", index=False, encoding="utf-8-sig")
    return X, meta_df, freqs


def scatter_2d(df: pd.DataFrame, x_col: str, y_col: str, out_path: Path, title: str, xlabel: str, ylabel: str) -> None:
    fig, ax = plt.subplots(figsize=(7, 6))
    for label in LABELS:
        part = df[df["emotion"] == label]
        ax.scatter(part[x_col], part[y_col], s=22, alpha=0.65, label=label, color=COLORS[label], edgecolors="none")
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.grid(alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_path, dpi=300)
    plt.close(fig)


def scatter_3d(df: pd.DataFrame, cols: tuple[str, str, str], out_path: Path, title: str, labels: tuple[str, str, str]) -> None:
    fig = plt.figure(figsize=(8, 7))
    ax = fig.add_subplot(111, projection="3d")
    for label in LABELS:
        part = df[df["emotion"] == label]
        ax.scatter(part[cols[0]], part[cols[1]], part[cols[2]], s=18, alpha=0.65, label=label, color=COLORS[label], depthshade=False)
    ax.set_title(title)
    ax.set_xlabel(labels[0])
    ax.set_ylabel(labels[1])
    ax.set_zlabel(labels[2])
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_path, dpi=300)
    plt.close(fig)


def component_summary(weights: np.ndarray, freqs: np.ndarray, prefix: str) -> pd.DataFrame:
    rows = []
    for i in range(weights.shape[1]):
        values = weights[:, i]
        top_idx = np.argsort(np.abs(values))[-10:][::-1]
        top_freqs = freqs[top_idx]
        counts = {}
        for band, (lo, hi) in BANDS.items():
            counts[band] = int(((top_freqs >= lo) & (top_freqs < hi if hi < freqs.max() else top_freqs <= hi)).sum())
        dominant = max(counts, key=counts.get)
        interp = f"{prefix}{i + 1} 的高权重频率主要落在 {dominant} 频段，可作为后续真实声学参数提取的候选关注区域。"
        rows.append(
            {
                "component": f"{prefix}{i + 1}",
                "top_frequencies_hz": ";".join(f"{f:.2f}" for f in top_freqs),
                "dominant_band": dominant,
                "low_count": counts["low"],
                "mid_count": counts["mid"],
                "high_count": counts["high"],
                "interpretation": interp,
            }
        )
    return pd.DataFrame(rows)


def plot_loadings(freqs: np.ndarray, weights: np.ndarray, names: list[str], out_dir: Path, combined_name: str, ylabel: str) -> None:
    fig, ax = plt.subplots(figsize=(9, 5))
    for i, name in enumerate(names):
        ax.plot(freqs, weights[:, i], label=name)
    ax.set_title(combined_name.replace("_", " ").replace(".png", ""))
    ax.set_xlabel("Frequency (Hz)")
    ax.set_ylabel(ylabel)
    ax.grid(alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_dir / combined_name, dpi=300)
    plt.close(fig)
    for i, name in enumerate(names):
        fig, ax = plt.subplots(figsize=(9, 4))
        ax.plot(freqs, weights[:, i], color="#1f77b4")
        ax.axhline(0, color="black", linewidth=0.8)
        ax.set_title(f"{name} {ylabel}")
        ax.set_xlabel("Frequency (Hz)")
        ax.set_ylabel(ylabel)
        ax.grid(alpha=0.25)
        fig.tight_layout()
        suffix = "loading" if name.startswith("PC") else "weight"
        fig.savefig(out_dir / f"{name.split()[0].lower()}_{name}_{suffix}.png".replace(" ", "_"), dpi=300)
        plt.close(fig)


def plot_boxplots(df: pd.DataFrame, components: list[str], method: str, out_dir: Path) -> None:
    for comp in components[:3]:
        data = [df[df["emotion"] == label][comp].to_numpy() for label in LABELS]
        fig, ax = plt.subplots(figsize=(7, 4.5))
        ax.boxplot(data, tick_labels=LABELS, showfliers=False)
        ax.set_title(f"{method} {comp} score by emotion")
        ax.set_ylabel("Score")
        ax.tick_params(axis="x", rotation=30)
        fig.tight_layout()
        fig.savefig(out_dir / f"{method.lower()}_{comp}_score_by_emotion.png", dpi=300)
        plt.close(fig)


def separation_scores(score_df: pd.DataFrame, method: str, components: list[str]) -> list[dict]:
    rows = []
    for comp in components:
        global_mean = score_df[comp].mean()
        between = 0.0
        within = 0.0
        for label in LABELS:
            vals = score_df[score_df["emotion"] == label][comp].to_numpy()
            if len(vals) == 0:
                continue
            between += len(vals) * float((vals.mean() - global_mean) ** 2)
            within += float(((vals - vals.mean()) ** 2).sum())
        rows.append({"method": method, "component": comp, "separation_score": between / max(within, 1e-12)})
    return rows


def run_pca(X: np.ndarray, meta_df: pd.DataFrame, freqs: np.ndarray, args: argparse.Namespace, dirs: dict[str, Path]) -> tuple[pd.DataFrame, pd.DataFrame]:
    n = min(args.pca_components, X.shape[0], X.shape[1])
    pca = PCA(n_components=n, random_state=args.random_seed)
    Z = pca.fit_transform(X)
    comps = [f"PC{i}" for i in range(1, n + 1)]
    scores = meta_df.copy()
    for i, comp in enumerate(comps):
        scores[comp] = Z[:, i]
    scores.to_csv(dirs["features"] / "saliency_frequency_pca_scores.csv", index=False, encoding="utf-8-sig")
    scatter_2d(scores, "PC1", "PC2", dirs["pca_figures"] / "saliency_frequency_pca_2d.png", "Saliency Frequency PCA 2D", f"PC1 ({pca.explained_variance_ratio_[0] * 100:.2f}% var)", f"PC2 ({pca.explained_variance_ratio_[1] * 100:.2f}% var)")
    scatter_3d(scores, ("PC1", "PC2", "PC3"), dirs["pca_figures"] / "saliency_frequency_pca_3d.png", "Saliency Frequency PCA 3D", ("PC1", "PC2", "PC3"))

    evr = pd.DataFrame({"component": comps, "explained_variance_ratio": pca.explained_variance_ratio_, "cumulative_explained_variance": np.cumsum(pca.explained_variance_ratio_)})
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(range(1, n + 1), evr["cumulative_explained_variance"], marker="o")
    ax.set_title("Saliency Frequency PCA Explained Variance")
    ax.set_xlabel("Number of components")
    ax.set_ylabel("Cumulative explained variance")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(dirs["pca_figures"] / "saliency_frequency_pca_explained_variance.png", dpi=300)
    plt.close(fig)

    loadings = pca.components_.T
    load_df = pd.DataFrame({"frequency_hz": freqs})
    for i, comp in enumerate(comps):
        load_df[f"{comp}_loading"] = loadings[:, i]
    load_df.to_csv(dirs["features"] / "saliency_frequency_pca_loadings.csv", index=False, encoding="utf-8-sig")
    plot_loadings(freqs, loadings, comps, dirs["loadings"], "pca_component_loadings.png", "Loading")
    for i, comp in enumerate(comps):
        src = dirs["loadings"] / f"{comp.lower()}_{comp}_loading.png"
        dst = dirs["loadings"] / f"pca_{comp}_loading.png"
        if src.exists():
            src.replace(dst)
    summary = component_summary(loadings, freqs, "PC")
    summary.to_csv(dirs["reports"] / "pca_component_frequency_summary.csv", index=False, encoding="utf-8-sig")
    plot_boxplots(scores, comps, "pca", dirs["boxplots"])
    return scores, summary


def run_ica(X: np.ndarray, meta_df: pd.DataFrame, freqs: np.ndarray, args: argparse.Namespace, dirs: dict[str, Path]) -> tuple[pd.DataFrame, pd.DataFrame, str]:
    n = min(args.ica_components, X.shape[0], X.shape[1])
    ica = FastICA(n_components=n, random_state=args.random_seed, max_iter=2000, tol=1e-3, whiten="unit-variance")
    note = "FastICA converged without warning."
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always", ConvergenceWarning)
        Z = ica.fit_transform(X)
        if any(issubclass(w.category, ConvergenceWarning) for w in caught):
            note = "FastICA emitted a convergence warning with max_iter=2000 and tol=1e-3."
    comps = [f"IC{i}" for i in range(1, n + 1)]
    scores = meta_df.copy()
    for i, comp in enumerate(comps):
        scores[comp] = Z[:, i]
    scores.to_csv(dirs["features"] / "saliency_frequency_ica_scores.csv", index=False, encoding="utf-8-sig")
    scatter_2d(scores, "IC1", "IC2", dirs["ica_figures"] / "saliency_frequency_ica_2d.png", "Saliency Frequency ICA 2D", "IC1", "IC2")
    scatter_3d(scores, ("IC1", "IC2", "IC3"), dirs["ica_figures"] / "saliency_frequency_ica_3d.png", "Saliency Frequency ICA 3D", ("IC1", "IC2", "IC3"))

    weights = ica.components_.T
    weight_df = pd.DataFrame({"frequency_hz": freqs})
    for i, comp in enumerate(comps):
        weight_df[f"{comp}_weight"] = weights[:, i]
    weight_df.to_csv(dirs["features"] / "saliency_frequency_ica_weights.csv", index=False, encoding="utf-8-sig")
    plot_loadings(freqs, weights, comps, dirs["loadings"], "ica_component_loadings.png", "Component Weight")
    for comp in comps:
        src = dirs["loadings"] / f"{comp.lower()}_{comp}_weight.png"
        dst = dirs["loadings"] / f"ica_{comp}_weight.png"
        if src.exists():
            src.replace(dst)
    summary = component_summary(weights, freqs, "IC")
    summary.to_csv(dirs["reports"] / "ica_component_frequency_summary.csv", index=False, encoding="utf-8-sig")
    plot_boxplots(scores, comps, "ica", dirs["boxplots"])
    return scores, summary, note


def write_report(args: argparse.Namespace, dirs: dict[str, Path], meta_df: pd.DataFrame, pca_summary: pd.DataFrame, ica_summary: pd.DataFrame, sep_df: pd.DataFrame, ica_note: str) -> None:
    counts = meta_df["emotion"].value_counts().reindex(LABELS, fill_value=0).astype(int).to_dict()
    top_sep = sep_df.sort_values("rank").head(5).to_dict("records")
    lines = [
        "# Saliency Frequency PCA / ICA Report",
        "",
        "## 实验目的",
        "",
        "输入不是 waveform，而是每个样本的 saliency frequency importance curve。目标是分解模型关注的频率模式，而不是追求最终分类准确率。",
        "",
        "## 数据数量",
        "",
        f"- total samples: {len(meta_df)}",
        f"- samples per class: {counts}",
        f"- use_only_correct: {args.use_only_correct}",
        "",
        "## PCA 结果",
        "",
        "- PCA 2D: `../figures/pca/saliency_frequency_pca_2d.png`",
        "- PCA 3D: `../figures/pca/saliency_frequency_pca_3d.png`",
        "- Explained variance: `../figures/pca/saliency_frequency_pca_explained_variance.png`",
        "- Loading curves: `../figures/component_loadings/pca_component_loadings.png`",
        "",
        "```text",
        pca_summary.to_string(index=False),
        "```",
        "",
        "## ICA 结果",
        "",
        "- ICA 2D: `../figures/ica/saliency_frequency_ica_2d.png`",
        "- ICA 3D: `../figures/ica/saliency_frequency_ica_3d.png`",
        "- Component weight curves: `../figures/component_loadings/ica_component_loadings.png`",
        f"- ICA status: {ica_note}",
        "",
        "```text",
        ica_summary.to_string(index=False),
        "```",
        "",
        "## 成分差异分析",
        "",
        "- separation scores: `component_separation_scores.csv`",
        f"- top components: {top_sep}",
        "",
        "## 刺激设计连接",
        "",
        "后续可以回到真实音频提取候选声学参数：低频能量占比、中频能量占比、高频能量占比、峰值频率、频率波动、频谱重心、有效发声时长、发声密度。",
        "",
        "## 局限性",
        "",
        "PCA / ICA 是无监督线性分解方法，不能直接证明某频段就是情绪机制。它们的作用是提供候选稳定成分，后续需要真实声学参数验证和动物刺激实验验证。",
    ]
    (dirs["reports"] / "saliency_frequency_pca_ica_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def run(args: argparse.Namespace) -> None:
    dirs = make_dirs(args.output_dir)
    X, meta_df, freqs = load_curves(args, dirs)
    X_model = StandardScaler().fit_transform(X) if args.standardize else X
    pca_scores, pca_summary = run_pca(X_model, meta_df, freqs, args, dirs)
    ica_scores, ica_summary, ica_note = run_ica(X_model, meta_df, freqs, args, dirs)
    sep_rows = []
    sep_rows.extend(separation_scores(pca_scores, "pca", [f"PC{i}" for i in range(1, min(args.pca_components, X.shape[1], X.shape[0]) + 1)]))
    sep_rows.extend(separation_scores(ica_scores, "ica", [f"IC{i}" for i in range(1, min(args.ica_components, X.shape[1], X.shape[0]) + 1)]))
    sep_df = pd.DataFrame(sep_rows).sort_values("separation_score", ascending=False).reset_index(drop=True)
    sep_df["rank"] = np.arange(1, len(sep_df) + 1)
    sep_df.to_csv(dirs["reports"] / "component_separation_scores.csv", index=False, encoding="utf-8-sig")
    write_report(args, dirs, meta_df, pca_summary, ica_summary, sep_df, ica_note)
    print(f"Saved saliency PCA/ICA outputs to: {args.output_dir}")


def parse_args() -> argparse.Namespace:
    base = EXPLAIN_DIR / "outputs"
    parser = argparse.ArgumentParser(description="PCA / ICA for saliency frequency importance curves.")
    parser.add_argument("--heatmap_dir", type=Path, default=base / "heatmaps_raw")
    parser.add_argument("--index_csv", type=Path, default=base / "reports" / "heatmap_index.csv")
    parser.add_argument("--frequency_axis_csv", type=Path, default=base / "reports" / "frequency_axis.csv")
    parser.add_argument("--output_dir", type=Path, default=EXPLAIN_DIR / "saliency_pca_ica" / "outputs")
    parser.add_argument("--use_only_correct", type=str_to_bool, default=True)
    parser.add_argument("--standardize", type=str_to_bool, default=True)
    parser.add_argument("--pca_components", type=int, default=5)
    parser.add_argument("--ica_components", type=int, default=5)
    parser.add_argument("--random_seed", type=int, default=42)
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
