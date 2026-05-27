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
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, make_scorer
from sklearn.model_selection import StratifiedKFold, cross_validate
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm

SCRIPT_PATH = Path(__file__).resolve()
REPRO_DIR = SCRIPT_PATH.parents[3]
if str(REPRO_DIR) not in sys.path:
    sys.path.insert(0, str(REPRO_DIR))

from dataset import EXPECTED_LABELS, SUPPORTED_AUDIO_EXTS, scan_audio_dataset  # noqa: E402

try:
    import torchaudio
except Exception as exc:  # pragma: no cover - environment dependent
    torchaudio = None
    TORCHAUDIO_IMPORT_ERROR = exc
else:
    TORCHAUDIO_IMPORT_ERROR = None

try:
    import soundfile as sf
except Exception as exc:  # pragma: no cover - environment dependent
    sf = None
    SOUNDFILE_IMPORT_ERROR = exc
else:
    SOUNDFILE_IMPORT_ERROR = None

try:
    import librosa
except Exception as exc:  # pragma: no cover - environment dependent
    librosa = None
    LIBROSA_IMPORT_ERROR = exc
else:
    LIBROSA_IMPORT_ERROR = None


COLOR_MAP = {
    "angry": "#d62728",
    "anxious": "#9467bd",
    "happy": "#2ca02c",
    "lonely": "#1f77b4",
    "sad": "#7f7f7f",
}


def str_to_bool(value: str | bool) -> bool:
    if isinstance(value, bool):
        return value
    value = value.strip().lower()
    if value in {"true", "1", "yes", "y"}:
        return True
    if value in {"false", "0", "no", "n"}:
        return False
    raise argparse.ArgumentTypeError(f"Expected true/false, got: {value}")


def prepare_output_dirs(output_dir: Path) -> dict[str, Path]:
    dirs = {
        "root": output_dir,
        "figures": output_dir / "figures",
        "features": output_dir / "features",
        "reports": output_dir / "reports",
    }
    for path in dirs.values():
        path.mkdir(parents=True, exist_ok=True)
    return dirs


def sample_dataset(df: pd.DataFrame, max_per_class: int, random_seed: int) -> pd.DataFrame:
    sampled = []
    for label in EXPECTED_LABELS:
        label_df = df[df["label"] == label]
        if len(label_df) > max_per_class:
            label_df = label_df.sample(n=max_per_class, random_state=random_seed)
        sampled.append(label_df)
    return pd.concat(sampled, ignore_index=True)


def read_with_torchaudio(path: Path) -> tuple[np.ndarray, int]:
    if torchaudio is None:
        raise RuntimeError(f"torchaudio unavailable: {TORCHAUDIO_IMPORT_ERROR}")
    waveform, sr = torchaudio.load(str(path))
    waveform_np = waveform.detach().cpu().numpy()
    if waveform_np.ndim == 2:
        waveform_np = waveform_np.mean(axis=0)
    return waveform_np.astype(np.float32), int(sr)


def read_with_soundfile(path: Path) -> tuple[np.ndarray, int]:
    if sf is None:
        raise RuntimeError(f"soundfile unavailable: {SOUNDFILE_IMPORT_ERROR}")
    audio, sr = sf.read(str(path), dtype="float32", always_2d=True)
    waveform = audio.mean(axis=1)
    return waveform.astype(np.float32), int(sr)


def resample_waveform(waveform: np.ndarray, src_sr: int, target_sr: int) -> np.ndarray:
    if src_sr == target_sr:
        return waveform.astype(np.float32)
    if librosa is None:
        raise RuntimeError(f"resampling requires librosa when sample rates differ: {LIBROSA_IMPORT_ERROR}")
    return librosa.resample(waveform.astype(np.float32), orig_sr=src_sr, target_sr=target_sr).astype(np.float32)


def load_preprocess_waveform(path: Path, sample_rate: int, num_samples: int) -> tuple[np.ndarray | None, str | None]:
    try:
        waveform, sr = read_with_torchaudio(path)
    except Exception as torchaudio_exc:
        try:
            waveform, sr = read_with_soundfile(path)
        except Exception as soundfile_exc:
            if path.suffix.lower() == ".mp3":
                return None, f"torchaudio failed; soundfile failed for mp3: {soundfile_exc}"
            return None, f"torchaudio failed: {torchaudio_exc}; soundfile failed: {soundfile_exc}"

    if waveform.size == 0:
        return None, "empty audio"

    try:
        waveform = resample_waveform(waveform, sr, sample_rate)
    except Exception as exc:
        return None, f"resample failed: {exc}"

    if len(waveform) < num_samples:
        waveform = np.pad(waveform, (0, num_samples - len(waveform)), mode="constant")
    elif len(waveform) > num_samples:
        waveform = waveform[:num_samples]

    max_abs = float(np.max(np.abs(waveform)))
    if max_abs > 0:
        waveform = waveform / max_abs
    return waveform.astype(np.float32), None


def build_waveform_matrix(df: pd.DataFrame, args: argparse.Namespace, reports_dir: Path) -> tuple[np.ndarray, pd.DataFrame, pd.DataFrame]:
    num_samples = int(args.sample_rate * args.duration)
    rows = []
    skipped = []
    waveforms = []

    for row in tqdm(df.itertuples(index=False), total=len(df), desc="Loading waveform"):
        path = Path(row.filepath)
        waveform, reason = load_preprocess_waveform(path, args.sample_rate, num_samples)
        if reason is not None:
            skipped.append({"file_path": str(path), "reason": reason})
            continue
        waveforms.append(waveform)
        rows.append({"file_path": str(path), "emotion": row.label, "class_id": int(row.class_id)})

    skipped_df = pd.DataFrame(skipped, columns=["file_path", "reason"])
    skipped_df.to_csv(reports_dir / "skipped_files.csv", index=False, encoding="utf-8-sig")

    if not waveforms:
        raise RuntimeError("No audio files were successfully loaded.")

    meta_df = pd.DataFrame(rows)
    return np.stack(waveforms).astype(np.float32), meta_df, skipped_df


def maybe_standardize(X: np.ndarray, enabled: bool) -> tuple[np.ndarray, StandardScaler | None]:
    if not enabled:
        return X, None
    scaler = StandardScaler()
    return scaler.fit_transform(X).astype(np.float32), scaler


def scatter_2d(df: pd.DataFrame, x_col: str, y_col: str, title: str, out_path: Path, xlabel: str, ylabel: str) -> None:
    fig, ax = plt.subplots(figsize=(7, 6))
    for label in EXPECTED_LABELS:
        label_df = df[df["emotion"] == label]
        ax.scatter(label_df[x_col], label_df[y_col], s=18, alpha=0.65, color=COLOR_MAP[label], label=label, edgecolors="none")
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.legend(frameon=True)
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(out_path, dpi=300)
    plt.close(fig)


def scatter_3d(df: pd.DataFrame, cols: tuple[str, str, str], title: str, out_path: Path, labels: tuple[str, str, str]) -> None:
    fig = plt.figure(figsize=(8, 7))
    ax = fig.add_subplot(111, projection="3d")
    for label in EXPECTED_LABELS:
        label_df = df[df["emotion"] == label]
        ax.scatter(
            label_df[cols[0]],
            label_df[cols[1]],
            label_df[cols[2]],
            s=14,
            alpha=0.65,
            color=COLOR_MAP[label],
            label=label,
            depthshade=False,
        )
    ax.set_title(title)
    ax.set_xlabel(labels[0])
    ax.set_ylabel(labels[1])
    ax.set_zlabel(labels[2])
    ax.legend(frameon=True)
    fig.tight_layout()
    fig.savefig(out_path, dpi=300)
    plt.close(fig)


def run_pca(X: np.ndarray, meta_df: pd.DataFrame, args: argparse.Namespace, dirs: dict[str, Path]) -> tuple[np.ndarray, PCA, pd.DataFrame]:
    n_components = min(max(args.pca_components, 10), X.shape[0], X.shape[1])
    pca = PCA(n_components=n_components, svd_solver="randomized", random_state=args.random_seed)
    Z = pca.fit_transform(X)

    pca_2d = meta_df[["file_path", "emotion"]].copy()
    pca_2d["pc1"] = Z[:, 0]
    pca_2d["pc2"] = Z[:, 1]
    pca_2d.to_csv(dirs["features"] / "waveform_pca_2d.csv", index=False, encoding="utf-8-sig")
    scatter_2d(
        pca_2d,
        "pc1",
        "pc2",
        "Waveform PCA 2D",
        dirs["figures"] / "waveform_pca_2d.png",
        f"PC1 ({pca.explained_variance_ratio_[0] * 100:.2f}% var)",
        f"PC2 ({pca.explained_variance_ratio_[1] * 100:.2f}% var)",
    )

    pca_3d = pca_2d.copy()
    pca_3d["pc3"] = Z[:, 2]
    pca_3d.to_csv(dirs["features"] / "waveform_pca_3d.csv", index=False, encoding="utf-8-sig")
    scatter_3d(
        pca_3d,
        ("pc1", "pc2", "pc3"),
        "Waveform PCA 3D",
        dirs["figures"] / "waveform_pca_3d.png",
        (
            f"PC1 ({pca.explained_variance_ratio_[0] * 100:.2f}% var)",
            f"PC2 ({pca.explained_variance_ratio_[1] * 100:.2f}% var)",
            f"PC3 ({pca.explained_variance_ratio_[2] * 100:.2f}% var)",
        ),
    )

    evr_df = pd.DataFrame(
        {
            "component": np.arange(1, len(pca.explained_variance_ratio_) + 1),
            "explained_variance_ratio": pca.explained_variance_ratio_,
            "cumulative_explained_variance": np.cumsum(pca.explained_variance_ratio_),
        }
    )
    evr_df.to_csv(dirs["features"] / "waveform_pca_explained_variance.csv", index=False, encoding="utf-8-sig")

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(evr_df["component"], evr_df["cumulative_explained_variance"], marker="o")
    ax.set_title("Waveform PCA Explained Variance")
    ax.set_xlabel("Number of components")
    ax.set_ylabel("Cumulative explained variance")
    ax.set_xticks(evr_df["component"])
    ax.set_ylim(0, min(1.0, max(0.05, evr_df["cumulative_explained_variance"].max() * 1.1)))
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(dirs["figures"] / "waveform_pca_explained_variance.png", dpi=300)
    plt.close(fig)
    return Z, pca, evr_df


def run_ica(X: np.ndarray, meta_df: pd.DataFrame, args: argparse.Namespace, dirs: dict[str, Path]) -> tuple[np.ndarray, str]:
    n_components = min(args.ica_components, X.shape[0], X.shape[1])
    if n_components < 3:
        raise RuntimeError("ICA requires at least 3 components for requested 3D output.")

    ica_note = "FastICA converged without warning."
    ica = FastICA(n_components=n_components, random_state=args.random_seed, max_iter=2000, tol=1e-3, whiten="unit-variance")
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always", ConvergenceWarning)
        Z = ica.fit_transform(X)
        if any(issubclass(w.category, ConvergenceWarning) for w in caught):
            ica_note = "FastICA emitted a convergence warning with max_iter=2000 and tol=1e-3."

    ica_2d = meta_df[["file_path", "emotion"]].copy()
    ica_2d["ic1"] = Z[:, 0]
    ica_2d["ic2"] = Z[:, 1]
    ica_2d.to_csv(dirs["features"] / "waveform_ica_2d.csv", index=False, encoding="utf-8-sig")
    scatter_2d(ica_2d, "ic1", "ic2", "Waveform ICA 2D", dirs["figures"] / "waveform_ica_2d.png", "IC1", "IC2")

    ica_3d = ica_2d.copy()
    ica_3d["ic3"] = Z[:, 2]
    ica_3d.to_csv(dirs["features"] / "waveform_ica_3d.csv", index=False, encoding="utf-8-sig")
    scatter_3d(ica_3d, ("ic1", "ic2", "ic3"), "Waveform ICA 3D", dirs["figures"] / "waveform_ica_3d.png", ("IC1", "IC2", "IC3"))
    return Z, ica_note


def low_dim_separability(Z: np.ndarray, labels: np.ndarray, random_seed: int) -> dict[str, float | None]:
    min_count = int(pd.Series(labels).value_counts().min())
    if min_count < 2:
        return {"acc_mean": None, "acc_std": None, "macro_f1_mean": None, "macro_f1_std": None}

    n_splits = min(5, min_count)
    clf = LogisticRegression(max_iter=1000, class_weight="balanced", solver="lbfgs")
    cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_seed)
    scores = cross_validate(
        clf,
        Z[:, :2],
        labels,
        cv=cv,
        scoring={"accuracy": make_scorer(accuracy_score), "macro_f1": make_scorer(f1_score, average="macro", zero_division=0)},
    )
    return {
        "acc_mean": float(scores["test_accuracy"].mean()),
        "acc_std": float(scores["test_accuracy"].std()),
        "macro_f1_mean": float(scores["test_macro_f1"].mean()),
        "macro_f1_std": float(scores["test_macro_f1"].std()),
    }


def write_report(
    args: argparse.Namespace,
    dirs: dict[str, Path],
    meta_df: pd.DataFrame,
    skipped_df: pd.DataFrame,
    separability: dict,
    pca: PCA,
    ica_note: str,
) -> None:
    pca_sep = separability["pca"]
    ica_sep = separability["ica"]
    report = f"""# Waveform PCA / ICA Analysis

## 实验目的

本实验直接对原始 waveform 做 PCA / ICA，观察低维空间中五类狗叫情绪是否自然分离。它是 exploratory analysis，不涉及模型训练，也不修改原始音频。

## 数据与预处理

- 数据目录：`{args.data_dir}`
- 样本数：{len(meta_df)}
- 每类样本数：{meta_df["emotion"].value_counts().reindex(EXPECTED_LABELS, fill_value=0).to_dict()}
- 跳过文件数：{len(skipped_df)}
- 采样率：{args.sample_rate} Hz
- 时长：{args.duration} 秒
- 声道：单声道
- 长度统一：短音频右侧补零，长音频从开头截断
- 幅值归一化：每条 waveform 除以自身最大绝对值
- 标准化：{args.standardize}

## PCA 结果

![Waveform PCA 2D](../figures/waveform_pca_2d.png)

![Waveform PCA 3D](../figures/waveform_pca_3d.png)

![Waveform PCA Explained Variance](../figures/waveform_pca_explained_variance.png)

PC1 explained variance ratio: {pca.explained_variance_ratio_[0]:.6f}

PC2 explained variance ratio: {pca.explained_variance_ratio_[1]:.6f}

请结合散点图观察 PC1 / PC2 是否形成明显情绪分离。如果不同颜色大量重叠，说明原始 waveform 的前两个线性主成分没有给出稳定的情绪分离。

## ICA 结果

![Waveform ICA 2D](../figures/waveform_ica_2d.png)

![Waveform ICA 3D](../figures/waveform_ica_3d.png)

ICA 状态：{ica_note}

请结合散点图观察 IC1 / IC2 是否形成明显情绪分离。如果不同颜色大量重叠，说明原始 waveform 的独立成分低维投影不容易直接区分五类情绪。

## 简单可分性指标

- PCA 2D Logistic Regression accuracy: {pca_sep["acc_mean"]} ± {pca_sep["acc_std"]}
- PCA 2D Logistic Regression macro F1: {pca_sep["macro_f1_mean"]} ± {pca_sep["macro_f1_std"]}
- ICA 2D Logistic Regression accuracy: {ica_sep["acc_mean"]} ± {ica_sep["acc_std"]}
- ICA 2D Logistic Regression macro F1: {ica_sep["macro_f1_mean"]} ± {ica_sep["macro_f1_std"]}

这些指标只是辅助判断低维空间能否区分类别，不应当作最终分类性能。

## 局限性说明

- waveform PCA / ICA 直接使用原始采样点。
- 结果容易受到时间对齐、相位、静音比例、裁剪位置、响度差异影响。
- 如果五类情绪没有明显分开，不能直接说明情绪没有声学差异。
- 只能说明原始 waveform 的线性低维表示不足以稳定区分情绪。

## 后续建议

- 后续应对 saliency frequency curve、model embedding、guided acoustic features 做 PCA / ICA。
- 这些表示更适合解释情绪相关声学结构和后续动物刺激设计。
"""
    (dirs["reports"] / "waveform_pca_ica_report.md").write_text(report, encoding="utf-8")


def run_analysis(args: argparse.Namespace) -> None:
    dirs = prepare_output_dirs(args.output_dir)
    np.random.seed(args.random_seed)

    df = scan_audio_dataset(args.data_dir)
    df = df[df["extension"].str.lower().isin(SUPPORTED_AUDIO_EXTS)].copy()
    sampled_df = sample_dataset(df, args.max_per_class, args.random_seed)
    sampled_df.to_csv(dirs["reports"] / "sampled_files.csv", index=False, encoding="utf-8-sig")

    X, meta_df, skipped_df = build_waveform_matrix(sampled_df, args, dirs["reports"])
    X_for_dr, _ = maybe_standardize(X, args.standardize)

    pca_Z, pca, _ = run_pca(X_for_dr, meta_df, args, dirs)
    ica_Z, ica_note = run_ica(X_for_dr, meta_df, args, dirs)

    y = meta_df["emotion"].to_numpy()
    pca_sep = low_dim_separability(pca_Z, y, args.random_seed)
    ica_sep = low_dim_separability(ica_Z, y, args.random_seed)
    separability = {
        "num_samples": int(len(meta_df)),
        "samples_per_class": meta_df["emotion"].value_counts().reindex(EXPECTED_LABELS, fill_value=0).astype(int).to_dict(),
        "pca_2d_logreg_acc_mean": pca_sep["acc_mean"],
        "pca_2d_logreg_acc_std": pca_sep["acc_std"],
        "pca_2d_logreg_macro_f1_mean": pca_sep["macro_f1_mean"],
        "pca_2d_logreg_macro_f1_std": pca_sep["macro_f1_std"],
        "ica_2d_logreg_acc_mean": ica_sep["acc_mean"],
        "ica_2d_logreg_acc_std": ica_sep["acc_std"],
        "ica_2d_logreg_macro_f1_mean": ica_sep["macro_f1_mean"],
        "ica_2d_logreg_macro_f1_std": ica_sep["macro_f1_std"],
        "ica_note": ica_note,
    }
    (dirs["reports"] / "waveform_pca_ica_separability.json").write_text(
        json.dumps(separability, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    write_report(args, dirs, meta_df, skipped_df, {"pca": pca_sep, "ica": ica_sep}, pca, ica_note)

    print(f"Loaded samples: {len(meta_df)}")
    print(f"Skipped files: {len(skipped_df)}")
    print(f"Figures saved to: {dirs['figures']}")
    print(f"Features saved to: {dirs['features']}")
    print(f"Reports saved to: {dirs['reports']}")


def parse_args() -> argparse.Namespace:
    default_output = REPRO_DIR / "explain_analysis" / "waveform_pca_ica" / "outputs"
    parser = argparse.ArgumentParser(description="Waveform PCA / ICA exploratory analysis for dog emotion audio.")
    parser.add_argument("--data_dir", type=Path, required=True, help="Root directory containing angry/anxious/happy/lonely/sad subdirectories.")
    parser.add_argument("--output_dir", type=Path, default=default_output)
    parser.add_argument("--sample_rate", type=int, default=16000)
    parser.add_argument("--duration", type=float, default=4.0)
    parser.add_argument("--max_per_class", type=int, default=200)
    parser.add_argument("--random_seed", type=int, default=42)
    parser.add_argument("--standardize", type=str_to_bool, default=True)
    parser.add_argument("--pca_components", type=int, default=10)
    parser.add_argument("--ica_components", type=int, default=10)
    return parser.parse_args()


if __name__ == "__main__":
    run_analysis(parse_args())
