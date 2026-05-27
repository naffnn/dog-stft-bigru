from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd
import torch
from sklearn.metrics import accuracy_score, classification_report
from torch.utils.data import DataLoader
from tqdm import tqdm

sys.path.append(str(Path(__file__).resolve().parent))

from dataset import DogEmotionDataset
from features import resolve_feature_type, warn_if_sst_fallback
from models import MambaEmotionModel, SimpleCNNBaseline
from scan_dataset import scan_dataset
from train import make_splits
from utils import ensure_project_dirs, get_device, set_seed


def load_checkpoint(path: Path) -> dict:
    try:
        return torch.load(path, map_location="cpu", weights_only=True)
    except TypeError:
        return torch.load(path, map_location="cpu")


def build_model_from_checkpoint(
    checkpoint: dict,
    requested_model: str | None,
    requested_hlfe: bool,
) -> torch.nn.Module:
    model_name = checkpoint.get("model", requested_model or "mamba")
    use_hlfe = bool(checkpoint.get("use_hlfe", requested_hlfe))
    num_classes = int(checkpoint["num_classes"])
    if model_name == "cnn":
        return SimpleCNNBaseline(num_classes)
    return MambaEmotionModel(num_classes, use_hlfe=use_hlfe, force_backend=checkpoint.get("model_backend"))


def evaluate(args: argparse.Namespace) -> None:
    set_seed(args.seed)
    dirs = ensure_project_dirs(args.project_dir)
    index_path = dirs["reports"] / "dataset_index.csv"
    test_path = dirs["reports"] / "test_split.csv"
    if not index_path.exists():
        scan_dataset(args.data_root, args.project_dir)
    if not test_path.exists():
        make_splits(index_path, dirs["reports"], args.seed)

    ckpt_path = dirs["checkpoints"] / "best_model.pt"
    if not ckpt_path.exists():
        raise FileNotFoundError(f"Missing checkpoint: {ckpt_path}")
    checkpoint = load_checkpoint(ckpt_path)
    class_names = checkpoint["class_names"]
    model_name = checkpoint.get("model", args.model or "mamba")
    model_backend = checkpoint.get("model_backend", "cnn" if model_name == "cnn" else "bigru_fallback")
    use_hlfe = bool(checkpoint.get("use_hlfe", args.use_hlfe))
    requested_feature_type = checkpoint.get("requested_feature_type")
    actual_feature_type = checkpoint.get("actual_feature_type")
    if requested_feature_type is None or actual_feature_type is None:
        requested_feature_type, actual_feature_type = resolve_feature_type(args.feature_type, args.allow_sst_fallback)
    warn_if_sst_fallback(requested_feature_type, actual_feature_type)
    feature_cache = bool(checkpoint.get("feature_cache", args.use_feature_cache))
    allow_sst_fallback = bool(checkpoint.get("allow_sst_fallback", args.allow_sst_fallback))

    device = get_device()
    model = build_model_from_checkpoint(checkpoint, args.model, args.use_hlfe).to(device)
    model.load_state_dict(checkpoint["model_state"])
    model.eval()
    print(f"Model name: {model_name}")
    print(f"Model backend: {model_backend}")
    print(f"Requested feature type: {requested_feature_type}")
    print(f"Actual feature type: {actual_feature_type}")
    print(f"Use HLFE: {use_hlfe}")
    print(f"Checkpoint path: {ckpt_path}")
    print(f"Feature cache: {feature_cache}")
    print(f"Cache dir: {dirs['features_cache']}")
    print(f"Allow SST fallback: {allow_sst_fallback}")
    print(f"Device: {device}")

    ds = DogEmotionDataset(
        test_path,
        feature_type=actual_feature_type,
        cache_dir=dirs["features_cache"],
        use_feature_cache=feature_cache,
        rebuild_feature_cache=args.rebuild_feature_cache,
        allow_sst_fallback=allow_sst_fallback,
    )
    loader = DataLoader(ds, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)
    probs_all, preds_all, true_all = [], [], []
    with torch.no_grad():
        for features, labels in tqdm(loader):
            features = features.to(device)
            logits = model(features)
            probs = torch.softmax(logits, dim=1).cpu()
            preds = probs.argmax(dim=1)
            probs_all.extend(probs.max(dim=1).values.tolist())
            preds_all.extend(preds.tolist())
            true_all.extend(labels.tolist())

    report = classification_report(true_all, preds_all, target_names=class_names, digits=4)
    acc = accuracy_score(true_all, preds_all)
    print(f"accuracy: {acc:.4f}")
    print(report)
    header = "\n".join(
        [
            f"Model name: {model_name}",
            f"Model backend: {model_backend}",
            f"Requested feature type: {requested_feature_type}",
            f"Actual feature type: {actual_feature_type}",
            f"Use HLFE: {use_hlfe}",
            f"Checkpoint path: {ckpt_path}",
            f"Feature cache: {feature_cache}",
            f"Cache dir: {dirs['features_cache']}",
            f"Allow SST fallback: {allow_sst_fallback}",
            f"Device: {device}",
            "",
        ]
    )
    (dirs["reports"] / "classification_report.txt").write_text(
        f"{header}accuracy: {acc:.6f}\n\n{report}", encoding="utf-8"
    )

    test_df = pd.read_csv(test_path).reset_index(drop=True)
    pred_labels = [class_names[i] for i in preds_all]
    out = pd.DataFrame(
        {
            "filepath": test_df["filepath"],
            "label": test_df["label"],
            "true_id": true_all,
            "pred_id": preds_all,
            "pred_label": pred_labels,
            "probability": probs_all,
        }
    )
    out.to_csv(dirs["reports"] / "predictions.csv", index=False, encoding="utf-8-sig")
    alpha = model.hlfe_alpha_value() if hasattr(model, "hlfe_alpha_value") else None
    if alpha is not None:
        print(f"HLFE alpha sigmoid value: {alpha:.4f}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate dog emotion model.")
    parser.add_argument("--data_root", type=Path, required=True,
                        help="Root directory containing angry/anxious/happy/lonely/sad subdirectories.")
    parser.add_argument("--project_dir", type=Path, default=Path.cwd(),
                        help="Project directory for outputs (default: current working directory).")
    parser.add_argument("--model", choices=["cnn", "mamba"], default=None)
    parser.add_argument("--use_hlfe", action="store_true")
    parser.add_argument("--feature_type", choices=["stft", "sst_stft"], default="stft")
    parser.add_argument("--use_feature_cache", action="store_true")
    parser.add_argument("--rebuild_feature_cache", action="store_true")
    parser.add_argument("--allow_sst_fallback", action="store_true")
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num_workers", type=int, default=0)
    return parser.parse_args()

if __name__ == "__main__":
    evaluate(parse_args())
