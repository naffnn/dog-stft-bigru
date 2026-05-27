from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd
import torch
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parent
SRC = ROOT / "src"
sys.path.insert(0, str(SRC))

from dataset import DogEmotionDataset
from features import resolve_feature_type, warn_if_sst_fallback
from models import MambaEmotionModel, SimpleCNNBaseline
from scan_dataset import scan_dataset
from train import evaluate_to_files, make_splits
from utils import ensure_project_dirs, get_device, set_seed


def load_checkpoint(path: Path) -> dict:
    try:
        return torch.load(path, map_location="cpu", weights_only=True)
    except TypeError:
        return torch.load(path, map_location="cpu")
    except Exception:
        return torch.load(path, map_location="cpu", weights_only=False)


def build_model_from_checkpoint(checkpoint: dict, args: argparse.Namespace) -> torch.nn.Module:
    model_name = checkpoint.get("model", "mamba")
    num_classes = int(checkpoint["num_classes"])
    use_hlfe = bool(checkpoint.get("use_hlfe", args.use_hlfe))
    dropout = float(checkpoint.get("dropout", args.dropout))
    backend = args.backend if args.backend != "auto" else checkpoint.get("model_backend", "bigru_fallback")
    if model_name == "cnn":
        return SimpleCNNBaseline(num_classes)
    return MambaEmotionModel(num_classes, use_hlfe=use_hlfe, force_backend=backend, dropout=dropout)


def find_test_split(args: argparse.Namespace, checkpoint_path: Path, project_dirs: dict[str, Path]) -> Path:
    run_test_split = checkpoint_path.parent / "test_split.csv"
    if run_test_split.exists():
        return run_test_split
    legacy_test_split = project_dirs["reports"] / "test_split.csv"
    if legacy_test_split.exists():
        return legacy_test_split
    index_path = project_dirs["reports"] / "dataset_index.csv"
    if not index_path.exists():
        scan_dataset(args.data_root, args.project_dir)
    _, test_path, _ = make_splits(index_path, project_dirs["reports"], args.seed)
    return test_path


def evaluate(args: argparse.Namespace) -> None:
    set_seed(args.seed)
    project_dirs = ensure_project_dirs(args.project_dir)
    checkpoint_path = args.checkpoint.resolve()
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Missing checkpoint: {checkpoint_path}")

    checkpoint = load_checkpoint(checkpoint_path)
    output_dir = args.output_dir or (checkpoint_path.parent / "eval")
    output_dir.mkdir(parents=True, exist_ok=True)

    requested_feature_type = args.feature_type or checkpoint.get("requested_feature_type", "stft")
    _, actual_feature_type = resolve_feature_type(requested_feature_type, args.allow_sst_fallback)
    actual_feature_type = checkpoint.get("actual_feature_type", actual_feature_type)
    warn_if_sst_fallback(requested_feature_type, actual_feature_type)
    use_feature_cache = bool(checkpoint.get("feature_cache", args.use_feature_cache))
    allow_sst_fallback = bool(checkpoint.get("allow_sst_fallback", args.allow_sst_fallback))
    class_names = list(checkpoint["class_names"])

    device = get_device()
    model = build_model_from_checkpoint(checkpoint, args).to(device)
    model.load_state_dict(checkpoint["model_state"], strict=False)
    model.eval()

    test_split = find_test_split(args, checkpoint_path, project_dirs)
    ds = DogEmotionDataset(
        test_split,
        feature_type=actual_feature_type,
        cache_dir=project_dirs["features_cache"],
        use_feature_cache=use_feature_cache,
        rebuild_feature_cache=args.rebuild_feature_cache,
        allow_sst_fallback=allow_sst_fallback,
    )
    loader = DataLoader(ds, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)

    best_epoch = int(checkpoint.get("best_epoch", checkpoint.get("epoch", 0)))
    best_val_acc = float(checkpoint.get("best_val_acc", checkpoint.get("val_acc", 0.0)))
    metrics = evaluate_to_files(
        model,
        loader,
        device,
        class_names,
        output_dir,
        best_epoch,
        best_val_acc,
        checkpoint_path,
        extra_metrics={
            "loss_type": checkpoint.get("loss_type"),
            "class_names": class_names,
            "class_counts": checkpoint.get("class_counts"),
            "class_weights": checkpoint.get("class_weights"),
            "focal_gamma": checkpoint.get("focal_gamma"),
            "lambda_hl_prior": checkpoint.get("lambda_hl_prior"),
            "hl_logits_grad_norm": checkpoint.get("hl_logits_grad_norm"),
        },
    )
    config = {
        "data_root": str(args.data_root),
        "checkpoint": str(checkpoint_path),
        "test_split": str(test_split),
        "feature_type": requested_feature_type,
        "actual_feature_type": actual_feature_type,
        "use_hlfe": bool(checkpoint.get("use_hlfe", args.use_hlfe)),
        "backend": checkpoint.get("model_backend", args.backend),
        "batch_size": args.batch_size,
        "output_dir": str(output_dir),
        "metrics": metrics,
    }
    (output_dir / "eval_config.json").write_text(json.dumps(config, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"test_acc: {metrics['test_acc']:.4f}")
    print(f"macro_f1: {metrics['macro_f1']:.4f}")
    print(f"weighted_f1: {metrics['weighted_f1']:.4f}")
    print(f"best_epoch: {metrics['best_epoch']}")
    print(f"best_val_acc: {metrics['best_val_acc']:.4f}")
    print(f"HL-FE weights: low={metrics['hl_low_weight']}, high={metrics['hl_high_weight']}")
    print(f"HL-FE logits: low={metrics['hl_logit_low']}, high={metrics['hl_logit_high']}")
    print(f"HL-FE logits delta norm: {metrics['hl_logits_delta_norm']}")
    print(f"Evaluation artifacts saved: {output_dir}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate a saved STFT + HL-FE + BiGRU fallback checkpoint.")
    parser.add_argument("--data_dir", "--data_root", dest="data_root", type=Path, required=True,
                        help="Root directory containing angry/anxious/happy/lonely/sad subdirectories.")
    parser.add_argument("--project_dir", type=Path, default=ROOT)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--feature_type", choices=["stft", "sst_stft"], default="stft")
    parser.add_argument("--use_hl_fe", "--use_hlfe", dest="use_hlfe", action="store_true")
    parser.add_argument("--backend", choices=["auto", "bigru_fallback", "real_mamba_ssm"], default="bigru_fallback")
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--output_dir", type=Path, default=None)
    parser.add_argument("--use_feature_cache", action="store_true")
    parser.add_argument("--rebuild_feature_cache", action="store_true")
    parser.add_argument("--allow_sst_fallback", action="store_true")
    parser.add_argument("--dropout", type=float, default=0.4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num_workers", type=int, default=0)
    return parser.parse_args()

if __name__ == "__main__":
    evaluate(parse_args())
