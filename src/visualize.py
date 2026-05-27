from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import confusion_matrix

sys.path.append(str(Path(__file__).resolve().parent))

from dataset import DogEmotionDataset
from features import (
    TARGET_NUM_SAMPLES,
    TARGET_SAMPLE_RATE,
    feature_cache_path,
    feature_display_name,
    extract_sst_stft_feature,
    extract_stft_feature,
    resolve_feature_type,
    warn_if_sst_fallback,
)
from scan_dataset import scan_dataset
from utils import ensure_project_dirs, normalize_waveform

try:
    import torchaudio
except Exception:
    torchaudio = None
    import soundfile as sf


def _load_waveform(path: Path) -> tuple[torch.Tensor, int]:
    if torchaudio is not None:
        waveform, sr = torchaudio.load(str(path))
        return waveform.mean(dim=0), int(sr)
    data, sr = sf.read(str(path), always_2d=True)
    return torch.from_numpy(data.mean(axis=1)).float(), int(sr)


def load_experiment_config(project_dir: Path, requested_feature_type: str, actual_feature_type: str) -> dict:
    summary_path = project_dir / "outputs" / "reports" / "experiment_summary.json"
    config = {
        "model_backend": "unknown",
        "use_hlfe": False,
        "requested_feature_type": requested_feature_type,
        "actual_feature_type": actual_feature_type,
    }
    if summary_path.exists():
        loaded = json.loads(summary_path.read_text(encoding="utf-8"))
        config.update(
            {
                "model_backend": loaded.get("model_backend", config["model_backend"]),
                "use_hlfe": bool(loaded.get("use_hlfe", config["use_hlfe"])),
                "requested_feature_type": loaded.get("requested_feature_type", config["requested_feature_type"]),
                "actual_feature_type": loaded.get("actual_feature_type", config["actual_feature_type"]),
            }
        )
    return config


def config_text(config: dict) -> str:
    feature = feature_display_name(str(config.get("actual_feature_type", "stft")))
    backend = config.get("model_backend", "unknown")
    hlfe = "on" if config.get("use_hlfe") else "off"
    return f"Feature: {feature} | Model backend: {backend} | HL-FE: {hlfe}"


def plot_class_distribution(df: pd.DataFrame, figures_dir: Path) -> None:
    counts = df.groupby("label").size().sort_index()
    plt.figure(figsize=(8, 5))
    plt.bar(counts.index.astype(str), counts.values)
    plt.title("Class Distribution")
    plt.xlabel("Class")
    plt.ylabel("Count")
    plt.tight_layout()
    plt.savefig(figures_dir / "class_distribution.png", dpi=150)
    plt.close()


def plot_duration_distribution(df: pd.DataFrame, figures_dir: Path) -> None:
    plt.figure(figsize=(8, 5))
    plt.hist(df["duration_sec"], bins=30, edgecolor="black")
    plt.title("Duration Distribution")
    plt.xlabel("Duration (sec)")
    plt.ylabel("Count")
    plt.tight_layout()
    plt.savefig(figures_dir / "duration_distribution.png", dpi=150)
    plt.close()


def _pad_or_truncate_waveform(waveform: torch.Tensor) -> torch.Tensor:
    current = waveform.shape[-1]
    if current < TARGET_NUM_SAMPLES:
        return torch.nn.functional.pad(waveform, (0, TARGET_NUM_SAMPLES - current))
    return waveform[..., :TARGET_NUM_SAMPLES]


def _sample_feature(
    waveform: torch.Tensor,
    wav_path: Path,
    actual_feature_type: str,
    cache_dir: Path,
    use_feature_cache: bool,
    rebuild_feature_cache: bool,
    allow_sst_fallback: bool,
) -> torch.Tensor:
    cache_path = None
    if use_feature_cache:
        cache_path = feature_cache_path(cache_dir, wav_path, actual_feature_type)
        if cache_path.exists() and not rebuild_feature_cache:
            return torch.load(cache_path, map_location="cpu")

    if actual_feature_type == "sst_stft":
        feature = extract_sst_stft_feature(waveform.float(), allow_fallback=allow_sst_fallback)
    else:
        feature = extract_stft_feature(waveform.float())
    if cache_path is not None:
        torch.save(feature.cpu(), cache_path)
    return feature


def plot_samples(
    df: pd.DataFrame,
    figures_dir: Path,
    actual_feature_type: str,
    cache_dir: Path,
    use_feature_cache: bool,
    rebuild_feature_cache: bool,
    allow_sst_fallback: bool,
) -> None:
    labels = sorted(df["label"].unique())
    fig, axes = plt.subplots(len(labels), 2, figsize=(12, max(3 * len(labels), 4)))
    if len(labels) == 1:
        axes = np.array([axes])
    for row_idx, label in enumerate(labels):
        sample = df[df["label"] == label].sample(n=1, random_state=42).iloc[0]
        wav_path = Path(sample["filepath"])
        waveform, sr = _load_waveform(wav_path)
        max_points = min(waveform.numel(), sr * 4)
        wav_for_plot = waveform[:max_points]
        time = np.arange(max_points) / sr
        axes[row_idx, 0].plot(time, wav_for_plot.numpy(), linewidth=0.8)
        axes[row_idx, 0].set_title(f"Waveform: {label}")
        axes[row_idx, 0].set_xlabel("Time (sec)")
        axes[row_idx, 0].set_ylabel("Amplitude")

        wav = _pad_or_truncate_waveform(wav_for_plot.float())
        wav = normalize_waveform(wav)
        feature = _sample_feature(
            wav,
            wav_path,
            actual_feature_type,
            cache_dir,
            use_feature_cache,
            rebuild_feature_cache,
            allow_sst_fallback,
        ).numpy().T
        im = axes[row_idx, 1].imshow(feature, origin="lower", aspect="auto", interpolation="nearest")
        feature_label = feature_display_name(actual_feature_type)
        axes[row_idx, 1].set_title(f"{feature_label} Spectrogram: {label}")
        axes[row_idx, 1].set_xlabel("Frame")
        axes[row_idx, 1].set_ylabel("Frequency Bin")
        fig.colorbar(im, ax=axes[row_idx, 1], fraction=0.046, pad=0.04)
    plt.tight_layout()
    plt.savefig(figures_dir / "sample_waveform_and_spectrogram.png", dpi=150)
    plt.close()


def plot_training_curves(project_dir: Path, figures_dir: Path, config: dict) -> None:
    log_path = project_dir / "outputs" / "logs" / "train_log.csv"
    if not log_path.exists():
        print(f"Skip training curves, missing {log_path}")
        return
    log = pd.read_csv(log_path)
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    fig.suptitle(config_text(config))
    axes[0].plot(log["epoch"], log["train_loss"], label="train_loss")
    axes[0].plot(log["epoch"], log["val_loss"], label="val_loss")
    axes[0].set_title("Loss Curves")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Loss")
    axes[0].legend()
    axes[1].plot(log["epoch"], log["train_acc"], label="train_acc")
    axes[1].plot(log["epoch"], log["val_acc"], label="val_acc")
    axes[1].set_title("Accuracy Curves")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Accuracy")
    axes[1].legend()
    plt.tight_layout()
    plt.savefig(figures_dir / "training_curves.png", dpi=150)
    plt.close()


def plot_confusion_and_f1(project_dir: Path, figures_dir: Path, config: dict) -> None:
    pred_path = project_dir / "outputs" / "reports" / "predictions.csv"
    if not pred_path.exists():
        print(f"Skip confusion matrix and F1, missing {pred_path}")
        return
    pred = pd.read_csv(pred_path)
    labels_df = pred[["true_id", "label"]].drop_duplicates().sort_values("true_id")
    class_names = labels_df["label"].tolist()
    cm = confusion_matrix(pred["true_id"], pred["pred_id"], labels=list(range(len(class_names))))
    plt.figure(figsize=(7, 6))
    plt.imshow(cm, interpolation="nearest", cmap="Blues")
    plt.title(f"Confusion Matrix\n{config_text(config)}")
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.xticks(range(len(class_names)), class_names, rotation=45, ha="right")
    plt.yticks(range(len(class_names)), class_names)
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            plt.text(j, i, str(cm[i, j]), ha="center", va="center")
    plt.colorbar()
    plt.tight_layout()
    plt.savefig(figures_dir / "confusion_matrix.png", dpi=150)
    plt.close()

    f1_scores = []
    for class_id in range(len(class_names)):
        tp = int(((pred["true_id"] == class_id) & (pred["pred_id"] == class_id)).sum())
        fp = int(((pred["true_id"] != class_id) & (pred["pred_id"] == class_id)).sum())
        fn = int(((pred["true_id"] == class_id) & (pred["pred_id"] != class_id)).sum())
        precision = tp / max(tp + fp, 1)
        recall = tp / max(tp + fn, 1)
        f1_scores.append(2 * precision * recall / max(precision + recall, 1e-12))
    plt.figure(figsize=(8, 5))
    plt.bar(class_names, f1_scores)
    plt.ylim(0, 1)
    plt.title(f"Per-Class F1 Score\n{config_text(config)}")
    plt.xlabel("Class")
    plt.ylabel("F1 Score")
    plt.tight_layout()
    plt.savefig(figures_dir / "per_class_f1.png", dpi=150)
    plt.close()


def visualize(args: argparse.Namespace) -> None:
    dirs = ensure_project_dirs(args.project_dir)
    requested_feature_type, actual_feature_type = resolve_feature_type(args.feature_type, args.allow_sst_fallback)
    warn_if_sst_fallback(requested_feature_type, actual_feature_type)
    config = load_experiment_config(args.project_dir, requested_feature_type, actual_feature_type)
    index_path = dirs["reports"] / "dataset_index.csv"
    if not index_path.exists():
        scan_dataset(args.data_root, args.project_dir)
    df = pd.read_csv(index_path)
    plot_class_distribution(df, dirs["figures"])
    plot_duration_distribution(df, dirs["figures"])
    plot_samples(
        df,
        dirs["figures"],
        str(config.get("actual_feature_type", actual_feature_type)),
        dirs["features_cache"],
        args.use_feature_cache,
        args.rebuild_feature_cache,
        args.allow_sst_fallback,
    )
    plot_training_curves(args.project_dir, dirs["figures"], config)
    plot_confusion_and_f1(args.project_dir, dirs["figures"], config)
    print(f"Figures saved to: {dirs['figures']}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Create result visualizations.")
    parser.add_argument("--data_root", type=Path, required=True,
                        help="Root directory containing angry/anxious/happy/lonely/sad subdirectories.")
    parser.add_argument("--project_dir", type=Path, default=Path.cwd(),
                        help="Project directory for outputs (default: current working directory).")
    parser.add_argument("--feature_type", choices=["stft", "sst_stft"], default="stft")
    parser.add_argument("--use_feature_cache", action="store_true")
    parser.add_argument("--rebuild_feature_cache", action="store_true")
    parser.add_argument("--allow_sst_fallback", action="store_true")
    return parser.parse_args()

if __name__ == "__main__":
    visualize(parse_args())
