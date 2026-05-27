from __future__ import annotations

import argparse
import csv
import json
import re
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from tqdm import tqdm

SCRIPT_PATH = Path(__file__).resolve()
REPRO_DIR = SCRIPT_PATH.parents[2]
SRC_DIR = REPRO_DIR / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from dataset import DogEmotionDataset  # noqa: E402
from features import FREQ_BINS, HOP_LENGTH, N_FFT, TARGET_DURATION_SEC, TARGET_SAMPLE_RATE, WIN_LENGTH  # noqa: E402
from models import MambaEmotionModel  # noqa: E402

DEFAULT_CHECKPOINT = None  # User must specify --checkpoint unless a default run exists
DEFAULT_OUTPUT_DIR = SCRIPT_PATH.parent.parent / "outputs"
LABELS = ["angry", "anxious", "happy", "lonely", "sad"]
COLORS = {"angry": "#d62728", "anxious": "#9467bd", "happy": "#2ca02c", "lonely": "#1f77b4", "sad": "#7f7f7f"}


def load_checkpoint(path: Path, device: torch.device) -> dict:
    try:
        return torch.load(path, map_location=device, weights_only=False)
    except TypeError:
        return torch.load(path, map_location=device)


def safe_stem(text: str) -> str:
    text = Path(text).stem
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", text).strip("_")[:100]


def resolve_audio_path(path_text: str, label: str, data_dir: Path) -> Path:
    path = Path(path_text)
    try:
        if path.exists():
            return path
    except OSError:
        pass
    candidate = data_dir / label / path.name
    try:
        if candidate.exists():
            return candidate
    except OSError:
        pass
    return path


def prepare_split(args: argparse.Namespace, checkpoint: dict, reports_dir: Path) -> Path:
    split_path = args.checkpoint.parent / f"{args.split}_split.csv"
    if not split_path.exists():
        split_path = args.checkpoint.parent / "test_split.csv"
    if not split_path.exists():
        raise FileNotFoundError(
            f"Cannot find split CSV near checkpoint: {split_path}. "
            "Please pass a checkpoint directory containing test_split.csv or regenerate the split first."
        )

    df = pd.read_csv(split_path)
    rows = []
    for row in df.itertuples(index=False):
        label = str(row.label)
        if label not in LABELS:
            continue
        audio_path = resolve_audio_path(str(row.filepath), label, args.data_dir)
        new_row = row._asdict()
        new_row["filepath"] = str(audio_path)
        rows.append(new_row)
    df = pd.DataFrame(rows)
    sampled = []
    for label in LABELS:
        label_df = df[df["label"] == label]
        if len(label_df) > args.max_per_class:
            label_df = label_df.sample(n=args.max_per_class, random_state=42)
        sampled.append(label_df)
    out_df = pd.concat(sampled, ignore_index=True)
    out_path = reports_dir / "heatmap_sample_split.csv"
    out_df.to_csv(out_path, index=False, encoding="utf-8-sig")
    return out_path


def build_model(checkpoint: dict, device: torch.device) -> MambaEmotionModel:
    model = MambaEmotionModel(
        num_classes=int(checkpoint.get("num_classes", len(checkpoint.get("class_names", LABELS)))),
        freq_bins=int(checkpoint.get("freq_bins", FREQ_BINS)),
        use_hlfe=bool(checkpoint.get("use_hlfe", True)),
        force_backend=str(checkpoint.get("model_backend", "bigru_fallback")),
        dropout=float(checkpoint.get("dropout", 0.5)),
    )
    model.load_state_dict(checkpoint["model_state"])
    model.to(device)
    model.eval()
    return model


def save_feature_image(arr: np.ndarray, path: Path, title: str, cmap: str, vmin: float | None = None, vmax: float | None = None) -> None:
    fig, ax = plt.subplots(figsize=(8, 4.8))
    duration = TARGET_DURATION_SEC
    extent = [0, duration, 0, TARGET_SAMPLE_RATE / 2]
    im = ax.imshow(arr, origin="lower", aspect="auto", cmap=cmap, extent=extent, vmin=vmin, vmax=vmax)
    ax.set_title(title)
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Frequency (Hz)")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    fig.savefig(path, dpi=300)
    plt.close(fig)


def save_overlay(stft: np.ndarray, heatmap: np.ndarray, path: Path, title: str) -> None:
    fig, ax = plt.subplots(figsize=(8, 4.8))
    extent = [0, TARGET_DURATION_SEC, 0, TARGET_SAMPLE_RATE / 2]
    ax.imshow(stft, origin="lower", aspect="auto", cmap="gray_r", extent=extent)
    vmax = float(np.percentile(heatmap, 99)) if heatmap.size else None
    im = ax.imshow(heatmap, origin="lower", aspect="auto", cmap="magma", alpha=0.55, extent=extent, vmin=0, vmax=vmax)
    ax.set_title(title)
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Frequency (Hz)")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    fig.savefig(path, dpi=300)
    plt.close(fig)


def write_summary(args: argparse.Namespace, index_df: pd.DataFrame, reports_dir: Path) -> None:
    counts = index_df["true_label"].value_counts().reindex(LABELS, fill_value=0).astype(int).to_dict()
    lines = [
        "# Explain Summary",
        "",
        f"- checkpoint: `{args.checkpoint}`",
        f"- data split: `{args.split}`",
        f"- samples per class: {counts}",
        "",
        "## Saliency 方法",
        "",
        "本流程使用 gradient-based saliency，对预测类别 logit 关于输入 STFT 特征求梯度，并计算 `abs(gradient * input)`。",
        "",
        "## 时间与频率解释范围",
        "",
        "- 时间维度仅作为局部证据位置。",
        "- 因为音频统一裁剪/补零为 4 秒，不对绝对时间点作强解释。",
        "- 本阶段重点解释频率维度。",
    ]
    (reports_dir / "explain_summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def run(args: argparse.Namespace) -> None:
    if not args.data_dir.exists():
        raise FileNotFoundError(f"Data dir does not exist: {args.data_dir}. Please edit --data_dir.")
    if not args.checkpoint.exists():
        raise FileNotFoundError(f"Checkpoint does not exist: {args.checkpoint}. Please edit --checkpoint.")

    device = torch.device(args.device)
    raw_dir = args.output_dir / "heatmaps_raw"
    reports_dir = args.output_dir / "reports"
    raw_dir.mkdir(parents=True, exist_ok=True)
    reports_dir.mkdir(parents=True, exist_ok=True)
    for label in LABELS:
        (raw_dir / label).mkdir(parents=True, exist_ok=True)

    checkpoint = load_checkpoint(args.checkpoint, device)
    class_names = list(checkpoint.get("class_names", LABELS))
    split_csv = prepare_split(args, checkpoint, reports_dir)
    dataset = DogEmotionDataset(split_csv, feature_type=str(checkpoint.get("actual_feature_type", "stft")))
    model = build_model(checkpoint, device)

    rows = []
    for idx in tqdm(range(len(dataset)), desc="Generating saliency"):
        feature, label_tensor = dataset[idx]
        meta = dataset.df.iloc[idx]
        true_id = int(label_tensor.item())
        true_label = class_names[true_id]
        x = feature.unsqueeze(0).to(device)
        x.requires_grad_(True)
        logits = model(x)
        probs = torch.softmax(logits, dim=1)
        pred_id = int(probs.argmax(dim=1).item())
        pred_label = class_names[pred_id]
        confidence = float(probs[0, pred_id].detach().cpu().item())
        score = logits[0, pred_id]
        model.zero_grad(set_to_none=True)
        score.backward()
        saliency = (x.grad.detach() * x.detach()).abs().squeeze(0).cpu().numpy()
        stft = x.detach().squeeze(0).cpu().numpy()

        heatmap = saliency.T
        stft_map = stft.T
        sample_id = f"{idx:05d}_{true_label}_{safe_stem(str(meta['filepath']))}_pred-{pred_label}_conf-{confidence:.3f}"
        label_dir = raw_dir / true_label
        heatmap_npy = label_dir / f"{sample_id}_heatmap.npy"
        stft_npy = label_dir / f"{sample_id}_stft.npy"
        heatmap_png = label_dir / f"{sample_id}_heatmap.png"
        stft_png = label_dir / f"{sample_id}_stft.png"
        overlay_png = label_dir / f"{sample_id}_overlay.png"
        np.save(heatmap_npy, heatmap.astype(np.float32))
        np.save(stft_npy, stft_map.astype(np.float32))
        save_feature_image(stft_map, stft_png, f"STFT {true_label} pred={pred_label}", "viridis")
        save_feature_image(heatmap, heatmap_png, f"Saliency {true_label} pred={pred_label}", "magma", vmin=0)
        save_overlay(stft_map, heatmap, overlay_png, f"Overlay {true_label} pred={pred_label}")
        rows.append(
            {
                "sample_id": sample_id,
                "file_path": str(meta["filepath"]),
                "true_label": true_label,
                "pred_label": pred_label,
                "confidence": confidence,
                "correct": bool(pred_id == true_id),
                "heatmap_npy_path": str(heatmap_npy),
                "stft_npy_path": str(stft_npy),
                "heatmap_png_path": str(heatmap_png),
                "overlay_path": str(overlay_png),
            }
        )

    index_df = pd.DataFrame(rows)
    index_df.to_csv(reports_dir / "heatmap_index.csv", index=False, encoding="utf-8-sig")
    write_summary(args, index_df, reports_dir)
    print(f"Saved heatmap index: {reports_dir / 'heatmap_index.csv'}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate STFT saliency heatmaps from an existing STFT + HL-FE + BiGRU checkpoint.")
    parser.add_argument("--data_dir", type=Path, required=True, help="Root directory containing angry/anxious/happy/lonely/sad subdirectories.")
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--output_dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--split", type=str, default="test")
    parser.add_argument("--max_per_class", type=int, default=50)
    parser.add_argument("--device", type=str, default="cpu")
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
