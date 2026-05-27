from __future__ import annotations

import argparse
import csv
import json
import sys
from datetime import datetime
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_recall_fscore_support,
)
from sklearn.model_selection import train_test_split
from torch import nn
from torch.utils.data import DataLoader
from tqdm import tqdm

sys.path.append(str(Path(__file__).resolve().parent))

from dataset import DogEmotionDataset
from features import (
    HOP_LENGTH,
    N_FFT,
    TARGET_DURATION_SEC,
    TARGET_NUM_SAMPLES,
    TARGET_SAMPLE_RATE,
    WIN_LENGTH,
    resolve_feature_type,
    warn_if_sst_fallback,
)
from models import MambaEmotionModel, SimpleCNNBaseline
from scan_dataset import scan_dataset
from utils import class_names_from_dataframe, ensure_project_dirs, get_device, set_seed


HLFE_PARAM_KEYWORDS = ("hl", "high", "low", "freq", "gate", "alpha", "beta", "weight")


def make_splits(index_path: Path, reports_dir: Path, seed: int) -> tuple[Path, Path, pd.DataFrame]:
    df = pd.read_csv(index_path)
    if df.empty:
        raise RuntimeError(f"No valid wav files found in {index_path}")
    stratify = df["class_id"] if df["class_id"].value_counts().min() >= 2 else None
    train_df, test_df = train_test_split(df, test_size=0.2, random_state=seed, stratify=stratify)
    reports_dir.mkdir(parents=True, exist_ok=True)
    train_path = reports_dir / "train_split.csv"
    test_path = reports_dir / "test_split.csv"
    train_df.to_csv(train_path, index=False, encoding="utf-8-sig")
    test_df.to_csv(test_path, index=False, encoding="utf-8-sig")
    return train_path, test_path, df


def build_model(
    model_name: str,
    num_classes: int,
    use_hlfe: bool,
    force_backend: str | None = None,
    dropout: float = 0.4,
) -> nn.Module:
    if model_name == "cnn":
        return SimpleCNNBaseline(num_classes=num_classes)
    if model_name == "mamba":
        return MambaEmotionModel(
            num_classes=num_classes,
            use_hlfe=use_hlfe,
            force_backend=force_backend,
            dropout=dropout,
        )
    raise ValueError(f"Unknown model: {model_name}")


def get_model_backend(model_name: str, model: nn.Module) -> str:
    if model_name == "cnn":
        return "cnn"
    return str(getattr(model, "model_backend", "bigru_fallback"))


def jsonable_args(args: argparse.Namespace) -> dict:
    return {key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()}


def write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def load_checkpoint(path: Path, map_location: torch.device | str = "cpu") -> dict:
    try:
        return torch.load(path, map_location=map_location, weights_only=False)
    except TypeError:
        return torch.load(path, map_location=map_location)


def append_experiment_row(path: Path, row: dict) -> None:
    fieldnames = list(row.keys())
    exists = path.exists()
    if exists:
        with path.open("r", newline="", encoding="utf-8-sig") as f:
            reader = csv.DictReader(f)
            existing_fieldnames = reader.fieldnames or []
            existing_rows = list(reader)
        merged = list(dict.fromkeys([*existing_fieldnames, *fieldnames]))
        if merged != existing_fieldnames:
            with path.open("w", newline="", encoding="utf-8-sig") as f:
                writer = csv.DictWriter(f, fieldnames=merged)
                writer.writeheader()
                for existing_row in existing_rows:
                    writer.writerow({name: existing_row.get(name, "") for name in merged})
            fieldnames = merged
    with path.open("a", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if not exists:
            writer.writeheader()
        writer.writerow({name: row.get(name, "") for name in fieldnames})


def class_counts_from_frame(df: pd.DataFrame, num_classes: int) -> np.ndarray:
    counts = df["class_id"].value_counts().sort_index()
    return np.array([int(counts.get(i, 0)) for i in range(num_classes)], dtype=np.float32)


def make_class_weights(train_df: pd.DataFrame, num_classes: int, device: torch.device) -> tuple[list[int], torch.Tensor]:
    counts = class_counts_from_frame(train_df, num_classes)
    if np.any(counts <= 0):
        raise RuntimeError(f"Every class needs at least one train sample. Counts: {counts.tolist()}")
    total = float(counts.sum())
    weights = total / (float(num_classes) * counts)
    weights = weights / weights.mean()
    return counts.astype(int).tolist(), torch.tensor(weights, dtype=torch.float32, device=device)


def build_criterion(loss_type: str, class_weights: torch.Tensor | None) -> nn.Module | None:
    if loss_type == "ce":
        return nn.CrossEntropyLoss()
    if loss_type == "weighted_ce":
        return nn.CrossEntropyLoss(weight=class_weights)
    if loss_type == "focal":
        return None
    raise ValueError(f"Unknown loss_type: {loss_type}")


def compute_ce_loss(
    logits: torch.Tensor,
    labels: torch.Tensor,
    criterion: nn.Module | None,
    loss_type: str,
    class_weights: torch.Tensor | None,
    focal_gamma: float,
) -> torch.Tensor:
    if loss_type == "focal":
        ce = F.cross_entropy(logits, labels, weight=class_weights, reduction="none")
        pt = torch.exp(-ce)
        return ((1.0 - pt) ** focal_gamma * ce).mean()
    if criterion is None:
        raise RuntimeError("criterion cannot be None unless loss_type=focal")
    return criterion(logits, labels)


def compute_total_loss(
    model: nn.Module,
    logits: torch.Tensor,
    labels: torch.Tensor,
    criterion: nn.Module | None,
    loss_type: str,
    class_weights: torch.Tensor | None,
    focal_gamma: float,
    lambda_hl_prior: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    ce_loss = compute_ce_loss(logits, labels, criterion, loss_type, class_weights, focal_gamma)
    hl_prior_loss = logits.new_tensor(0.0)
    if lambda_hl_prior > 0 and hasattr(model, "get_hl_weights"):
        hl_weights = model.get_hl_weights()
        if hl_weights is not None:
            hl_prior_loss = torch.relu(hl_weights[1] - hl_weights[0] + 0.10)
    total_loss = ce_loss + lambda_hl_prior * hl_prior_loss
    return total_loss, ce_loss, hl_prior_loss


def get_hl_weight_values(model: nn.Module) -> tuple[float | None, float | None]:
    if hasattr(model, "hlfe_weights_value"):
        values = model.hlfe_weights_value()
        if values is not None:
            return values
    return None, None


def get_hl_logit_values(model: nn.Module) -> tuple[float | None, float | None]:
    if hasattr(model, "get_hl_logits"):
        logits = model.get_hl_logits()
        if logits is not None:
            values = logits.detach().cpu()
            return float(values[0].item()), float(values[1].item())
    return None, None


def get_hl_logit_init_values(model: nn.Module) -> tuple[float | None, float | None]:
    if hasattr(model, "get_hl_logits_init"):
        logits = model.get_hl_logits_init()
        if logits is not None:
            values = logits.detach().cpu()
            return float(values[0].item()), float(values[1].item())
    return None, None


def get_hl_logits_delta_norm(model: nn.Module) -> float | None:
    if hasattr(model, "get_hl_logits_delta_norm"):
        return model.get_hl_logits_delta_norm()
    return None


def print_hlfe_parameter_report(model: nn.Module, optimizer: torch.optim.Optimizer) -> None:
    optimizer_param_ids = {id(param) for group in optimizer.param_groups for param in group["params"]}
    print("\nHL-FE related parameter check:")
    found = False
    for name, param in model.named_parameters():
        if any(keyword in name.lower() for keyword in HLFE_PARAM_KEYWORDS):
            found = True
            data = param.detach().float().cpu()
            std = float(data.std(unbiased=False).item()) if data.numel() > 1 else 0.0
            print(
                f"  {name}: requires_grad={param.requires_grad} "
                f"mean={float(data.mean().item()):.6f} std={std:.6f} "
                f"in_optimizer={id(param) in optimizer_param_ids}"
            )
    if not found:
        print("  No parameter name matched HL-FE keywords.")


def run_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module | None,
    device: torch.device,
    loss_type: str,
    class_weights: torch.Tensor | None,
    focal_gamma: float,
    lambda_hl_prior: float,
    optimizer: torch.optim.Optimizer | None = None,
) -> dict:
    training = optimizer is not None
    model.train(training)
    total_loss_sum = 0.0
    ce_loss_sum = 0.0
    hl_prior_loss_sum = 0.0
    total = 0
    hl_grad_norm_sum = 0.0
    hl_grad_norm_count = 0
    hl_grad_norm_last = None
    true_all: list[int] = []
    pred_all: list[int] = []
    with torch.set_grad_enabled(training):
        for features, labels in tqdm(loader, leave=False):
            features = features.to(device)
            labels = labels.to(device)
            logits = model(features)
            loss, ce_loss, hl_prior_loss = compute_total_loss(
                model,
                logits,
                labels,
                criterion,
                loss_type,
                class_weights,
                focal_gamma,
                lambda_hl_prior,
            )
            if training:
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                if hasattr(model, "get_hl_logits_grad_norm"):
                    grad_norm = model.get_hl_logits_grad_norm()
                    if grad_norm is not None:
                        hl_grad_norm_last = grad_norm
                        hl_grad_norm_sum += grad_norm
                        hl_grad_norm_count += 1
                optimizer.step()
            batch_size = labels.size(0)
            total_loss_sum += float(loss.item()) * batch_size
            ce_loss_sum += float(ce_loss.item()) * batch_size
            hl_prior_loss_sum += float(hl_prior_loss.item()) * batch_size
            total += batch_size
            preds = logits.argmax(dim=1)
            true_all.extend(labels.detach().cpu().tolist())
            pred_all.extend(preds.detach().cpu().tolist())
    acc = accuracy_score(true_all, pred_all) if true_all else 0.0
    macro_f1 = f1_score(true_all, pred_all, average="macro", zero_division=0) if true_all else 0.0
    return {
        "loss": total_loss_sum / max(total, 1),
        "ce_loss": ce_loss_sum / max(total, 1),
        "hl_prior_loss": hl_prior_loss_sum / max(total, 1),
        "acc": acc,
        "macro_f1": macro_f1,
        "hl_logits_grad_norm": hl_grad_norm_sum / hl_grad_norm_count if hl_grad_norm_count else None,
        "hl_logits_grad_norm_last": hl_grad_norm_last,
        "true": true_all,
        "pred": pred_all,
    }


def save_confusion_matrix_plot(cm: np.ndarray, class_names: list[str], path: Path) -> None:
    fig, ax = plt.subplots(figsize=(8, 7))
    im = ax.imshow(cm, interpolation="nearest", cmap="Blues")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    ax.set_xticks(np.arange(len(class_names)))
    ax.set_yticks(np.arange(len(class_names)))
    ax.set_xticklabels(class_names, rotation=45, ha="right")
    ax.set_yticklabels(class_names)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title("Confusion Matrix")
    threshold = cm.max() / 2.0 if cm.size else 0
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, int(cm[i, j]), ha="center", va="center", color="white" if cm[i, j] > threshold else "black")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def save_training_plots(log_df: pd.DataFrame, run_dir: Path) -> None:
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(log_df["epoch"], log_df["train_loss"], label="train_loss")
    ax.plot(log_df["epoch"], log_df["val_loss"], label="val_loss")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    ax.legend()
    fig.tight_layout()
    fig.savefig(run_dir / "loss_curve.png", dpi=150)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(log_df["epoch"], log_df["train_acc"], label="train_acc")
    ax.plot(log_df["epoch"], log_df["val_acc"], label="val_acc")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Accuracy")
    ax.legend()
    fig.tight_layout()
    fig.savefig(run_dir / "acc_curve.png", dpi=150)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(log_df["epoch"], log_df["train_macro_f1"], label="train_macro_f1")
    ax.plot(log_df["epoch"], log_df["val_macro_f1"], label="val_macro_f1")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Macro F1")
    ax.legend()
    fig.tight_layout()
    fig.savefig(run_dir / "macro_f1_curve.png", dpi=150)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(log_df["epoch"], log_df["low_weight"], label="low_weight")
    ax.plot(log_df["epoch"], log_df["high_weight"], label="high_weight")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Weight")
    ax.set_ylim(0.0, 1.0)
    ax.legend()
    fig.tight_layout()
    fig.savefig(run_dir / "hl_weights_curve.png", dpi=150)
    plt.close(fig)


def save_per_class_metrics(report_dict: dict, class_names: list[str], path: Path) -> None:
    precision = [report_dict[name]["precision"] for name in class_names]
    recall = [report_dict[name]["recall"] for name in class_names]
    f1 = [report_dict[name]["f1-score"] for name in class_names]
    x = np.arange(len(class_names))
    width = 0.25
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.bar(x - width, precision, width, label="precision")
    ax.bar(x, recall, width, label="recall")
    ax.bar(x + width, f1, width, label="f1")
    ax.set_xticks(x)
    ax.set_xticklabels(class_names, rotation=45, ha="right")
    ax.set_ylim(0.0, 1.0)
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def evaluate_to_files(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    class_names: list[str],
    run_dir: Path,
    best_epoch: int,
    best_val_acc: float,
    checkpoint_path: Path,
    extra_metrics: dict | None = None,
) -> dict:
    model.eval()
    true_all: list[int] = []
    pred_all: list[int] = []
    with torch.no_grad():
        for features, labels in tqdm(loader, leave=False):
            logits = model(features.to(device))
            pred_all.extend(logits.argmax(dim=1).detach().cpu().tolist())
            true_all.extend(labels.tolist())

    acc = accuracy_score(true_all, pred_all)
    macro_f1 = f1_score(true_all, pred_all, average="macro", zero_division=0)
    weighted_f1 = f1_score(true_all, pred_all, average="weighted", zero_division=0)
    report_text = classification_report(true_all, pred_all, target_names=class_names, digits=4, zero_division=0)
    report_dict = classification_report(
        true_all, pred_all, target_names=class_names, digits=4, output_dict=True, zero_division=0
    )
    cm = confusion_matrix(true_all, pred_all, labels=list(range(len(class_names))))
    pd.DataFrame(cm, index=class_names, columns=class_names).to_csv(run_dir / "confusion_matrix.csv", encoding="utf-8-sig")
    save_confusion_matrix_plot(cm, class_names, run_dir / "confusion_matrix.png")
    save_per_class_metrics(report_dict, class_names, run_dir / "per_class_metrics.png")
    (run_dir / "classification_report.txt").write_text(
        f"test_acc: {acc:.6f}\nmacro_f1: {macro_f1:.6f}\nweighted_f1: {weighted_f1:.6f}\n\n{report_text}",
        encoding="utf-8",
    )
    low_w, high_w = get_hl_weight_values(model)
    hl_logit_low, hl_logit_high = get_hl_logit_values(model)
    hl_logit_init_low, hl_logit_init_high = get_hl_logit_init_values(model)
    hl_logits_delta_norm = get_hl_logits_delta_norm(model)
    metrics = {
        "test_acc": acc,
        "macro_f1": macro_f1,
        "weighted_f1": weighted_f1,
        "per_class_precision": {name: report_dict[name]["precision"] for name in class_names},
        "per_class_recall": {name: report_dict[name]["recall"] for name in class_names},
        "per_class_f1": {name: report_dict[name]["f1-score"] for name in class_names},
        "per_class_support": {name: int(report_dict[name]["support"]) for name in class_names},
        "best_epoch": best_epoch,
        "best_val_acc": best_val_acc,
        "checkpoint": str(checkpoint_path),
        "hl_low_weight": low_w,
        "hl_high_weight": high_w,
        "hl_logit_low": hl_logit_low,
        "hl_logit_high": hl_logit_high,
        "hl_logit_init_low": hl_logit_init_low,
        "hl_logit_init_high": hl_logit_init_high,
        "hl_logits_delta_norm": hl_logits_delta_norm,
    }
    if extra_metrics:
        metrics.update(extra_metrics)
    write_json(run_dir / "metrics.json", metrics)
    return metrics


def train(args: argparse.Namespace) -> None:
    set_seed(args.seed)
    project_dirs = ensure_project_dirs(args.project_dir)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = args.output_dir or (args.project_dir / "runs" / f"stft_hlfe_bigru_{timestamp}")
    run_dir.mkdir(parents=True, exist_ok=True)

    requested_feature_type, actual_feature_type = resolve_feature_type(args.feature_type, args.allow_sst_fallback)
    warn_if_sst_fallback(requested_feature_type, actual_feature_type)

    index_path = project_dirs["reports"] / "dataset_index.csv"
    if not index_path.exists():
        scan_dataset(args.data_root, args.project_dir)

    train_path, test_path, df = make_splits(index_path, run_dir, args.seed)
    train_df = pd.read_csv(train_path)
    class_names = class_names_from_dataframe(df)
    num_classes = len(class_names)
    if num_classes < 2:
        raise RuntimeError("Need at least two classes to train a classifier.")

    device = get_device()
    train_ds = DogEmotionDataset(
        train_path,
        feature_type=actual_feature_type,
        cache_dir=project_dirs["features_cache"],
        use_feature_cache=args.use_feature_cache,
        rebuild_feature_cache=args.rebuild_feature_cache,
        allow_sst_fallback=args.allow_sst_fallback,
    )
    val_ds = DogEmotionDataset(
        test_path,
        feature_type=actual_feature_type,
        cache_dir=project_dirs["features_cache"],
        use_feature_cache=args.use_feature_cache,
        rebuild_feature_cache=args.rebuild_feature_cache,
        allow_sst_fallback=args.allow_sst_fallback,
    )
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)

    force_backend = args.backend if args.backend != "auto" else None
    model = build_model(args.model, num_classes, args.use_hlfe, force_backend=force_backend, dropout=args.dropout).to(device)
    model_backend = get_model_backend(args.model, model)

    train_class_counts, computed_class_weights = make_class_weights(train_df, num_classes, device)
    active_class_weights = computed_class_weights if args.loss_type in ("weighted_ce", "focal") else None
    criterion = build_criterion(args.loss_type, active_class_weights if args.loss_type == "weighted_ce" else None)
    focal_weights = active_class_weights if args.loss_type == "focal" else None
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    print(f"Device: {device}")
    print(f"Model backend: {model_backend}")
    print(f"Requested feature type: {requested_feature_type}")
    print(f"Actual feature type: {actual_feature_type}")
    print(f"Feature cache: {args.use_feature_cache}")
    print(f"Cache dir: {project_dirs['features_cache']}")
    print(f"Allow SST fallback: {args.allow_sst_fallback}")
    print(f"Run dir: {run_dir}")
    print(f"Loss type: {args.loss_type}")
    print(f"Focal gamma: {args.focal_gamma}")
    print(f"Class names: {class_names}")
    print(f"Class counts: {train_class_counts}")
    if active_class_weights is None:
        print("Class weights: None")
    else:
        print(f"Class weights: {[round(float(x), 6) for x in active_class_weights.detach().cpu().tolist()]}")
    low_w, high_w = get_hl_weight_values(model)
    hl_logit_init_low, hl_logit_init_high = get_hl_logit_init_values(model)
    if low_w is not None and high_w is not None:
        print(f"Initial HL-FE weights: low={low_w:.4f}, high={high_w:.4f}")
    if hl_logit_init_low is not None and hl_logit_init_high is not None:
        print(f"Initial HL-FE logits: low={hl_logit_init_low:.6f}, high={hl_logit_init_high:.6f}")
    print_hlfe_parameter_report(model, optimizer)

    class_weights_for_log = None if active_class_weights is None else active_class_weights.detach().cpu().tolist()
    config = {
        **jsonable_args(args),
        "run_dir": str(run_dir),
        "loss_type": args.loss_type,
        "class_names": class_names,
        "class_counts": train_class_counts,
        "class_weights": class_weights_for_log,
        "focal_gamma": args.focal_gamma,
        "hl_logit_init_low": hl_logit_init_low,
        "hl_logit_init_high": hl_logit_init_high,
        "model_backend": model_backend,
        "requested_feature_type": requested_feature_type,
        "actual_feature_type": actual_feature_type,
    }
    write_json(run_dir / "config.json", config)

    best_acc = -1.0
    best_epoch = 0
    no_improve_epochs = 0
    log_rows: list[dict] = []
    best_path = run_dir / "best_model.pt"
    last_path = run_dir / "last_model.pt"

    for epoch in range(1, args.epochs + 1):
        print(f"\nEpoch {epoch}/{args.epochs}")
        train_metrics = run_epoch(
            model,
            train_loader,
            criterion,
            device,
            args.loss_type,
            focal_weights,
            args.focal_gamma,
            args.lambda_hl_prior,
            optimizer,
        )
        val_metrics = run_epoch(
            model,
            val_loader,
            criterion,
            device,
            args.loss_type,
            focal_weights,
            args.focal_gamma,
            args.lambda_hl_prior,
        )
        low_w, high_w = get_hl_weight_values(model)
        hl_logit_low, hl_logit_high = get_hl_logit_values(model)
        hl_logits_delta_norm = get_hl_logits_delta_norm(model)
        current_lr = optimizer.param_groups[0]["lr"]
        row = {
            "epoch": epoch,
            "train_loss": train_metrics["loss"],
            "val_loss": val_metrics["loss"],
            "train_acc": train_metrics["acc"],
            "val_acc": val_metrics["acc"],
            "train_macro_f1": train_metrics["macro_f1"],
            "val_macro_f1": val_metrics["macro_f1"],
            "lr": current_lr,
            "low_weight": low_w,
            "high_weight": high_w,
            "ce_loss": train_metrics["ce_loss"],
            "focal_loss": train_metrics["ce_loss"] if args.loss_type == "focal" else None,
            "focal_gamma": args.focal_gamma if args.loss_type == "focal" else None,
            "hl_prior_loss": train_metrics["hl_prior_loss"],
            "total_loss": train_metrics["loss"],
            "hl_logit_low": hl_logit_low,
            "hl_logit_high": hl_logit_high,
            "hl_logits_grad_norm": train_metrics["hl_logits_grad_norm"],
            "hl_logits_delta_norm": hl_logits_delta_norm,
        }
        log_rows.append(row)
        log_df = pd.DataFrame(log_rows)
        log_df.to_csv(run_dir / "train_log.csv", index=False, encoding="utf-8-sig")
        log_df[
            [
                "epoch",
                "low_weight",
                "high_weight",
                "hl_logit_low",
                "hl_logit_high",
                "hl_logits_grad_norm",
                "hl_logits_delta_norm",
                "ce_loss",
                "focal_loss",
                "focal_gamma",
                "hl_prior_loss",
                "total_loss",
                "val_acc",
                "val_macro_f1",
            ]
        ].to_csv(run_dir / "hl_weights_log.csv", index=False, encoding="utf-8-sig")
        print(
            f"train_loss={train_metrics['loss']:.4f} train_acc={train_metrics['acc']:.4f} "
            f"train_macro_f1={train_metrics['macro_f1']:.4f} val_loss={val_metrics['loss']:.4f} "
            f"val_acc={val_metrics['acc']:.4f} val_macro_f1={val_metrics['macro_f1']:.4f}"
        )
        if low_w is not None and high_w is not None:
            print(f"HL-FE weights: low={low_w:.4f}, high={high_w:.4f}")
        if hl_logit_low is not None and hl_logit_high is not None:
            print(
                f"HL-FE logits: low={hl_logit_low:.6f}, high={hl_logit_high:.6f} "
                f"grad_norm={train_metrics['hl_logits_grad_norm']} delta_norm={hl_logits_delta_norm}"
            )
        print(
            f"ce_loss={train_metrics['ce_loss']:.4f} "
            f"hl_prior_loss={train_metrics['hl_prior_loss']:.4f} total_loss={train_metrics['loss']:.4f}"
        )

        checkpoint = {
            "model_state": model.state_dict(),
            "model": args.model,
            "model_backend": model_backend,
            "use_hlfe": args.use_hlfe,
            "num_classes": num_classes,
            "class_names": class_names,
            "class_counts": train_class_counts,
            "class_weights": class_weights_for_log,
            "epoch": epoch,
            "val_acc": val_metrics["acc"],
            "val_macro_f1": val_metrics["macro_f1"],
            "best_epoch": best_epoch,
            "best_val_acc": best_acc,
            "requested_feature_type": requested_feature_type,
            "actual_feature_type": actual_feature_type,
            "feature_cache": args.use_feature_cache,
            "allow_sst_fallback": args.allow_sst_fallback,
            "sample_rate": TARGET_SAMPLE_RATE,
            "duration_sec": TARGET_DURATION_SEC,
            "num_samples": TARGET_NUM_SAMPLES,
            "n_fft": N_FFT,
            "win_length": WIN_LENGTH,
            "hop_length": HOP_LENGTH,
            "batch_size": args.batch_size,
            "learning_rate": args.lr,
            "weight_decay": args.weight_decay,
            "dropout": args.dropout,
            "loss_type": args.loss_type,
            "lambda_hl_prior": args.lambda_hl_prior,
            "focal_gamma": args.focal_gamma,
            "hl_logit_low": hl_logit_low,
            "hl_logit_high": hl_logit_high,
            "hl_logits_grad_norm": train_metrics["hl_logits_grad_norm"],
            "hl_logits_delta_norm": hl_logits_delta_norm,
            "epochs": args.epochs,
            "seed": args.seed,
        }
        torch.save(checkpoint, last_path)
        if val_metrics["acc"] > best_acc:
            best_acc = val_metrics["acc"]
            best_epoch = epoch
            checkpoint["best_epoch"] = best_epoch
            checkpoint["best_val_acc"] = best_acc
            torch.save(checkpoint, best_path)
            no_improve_epochs = 0
        else:
            no_improve_epochs += 1
            if no_improve_epochs >= args.patience:
                print(f"Early stopping at epoch {epoch}; patience={args.patience}")
                break

    save_training_plots(pd.DataFrame(log_rows), run_dir)
    hl_grad_norms = [
        float(row["hl_logits_grad_norm"])
        for row in log_rows
        if row.get("hl_logits_grad_norm") is not None and not pd.isna(row.get("hl_logits_grad_norm"))
    ]
    if hl_grad_norms and max(hl_grad_norms[-min(3, len(hl_grad_norms)) :]) < 1e-6:
        print("Warning: HL-FE logits grad_norm is near zero. HL-FE may not be effectively trained.")
    final_hl_delta_norm = log_rows[-1].get("hl_logits_delta_norm") if log_rows else None
    if final_hl_delta_norm is not None and not pd.isna(final_hl_delta_norm) and float(final_hl_delta_norm) < 0.01:
        print("Warning: HL-FE logits changed very little. Current low/high weights may mainly reflect initialization.")

    best_checkpoint = load_checkpoint(best_path, map_location=device)
    model.load_state_dict(best_checkpoint["model_state"])
    metrics = evaluate_to_files(
        model,
        val_loader,
        device,
        class_names,
        run_dir,
        best_epoch,
        best_acc,
        best_path,
        extra_metrics={
            "loss_type": args.loss_type,
            "class_names": class_names,
            "class_counts": train_class_counts,
            "class_weights": class_weights_for_log,
            "focal_gamma": args.focal_gamma,
            "lambda_hl_prior": args.lambda_hl_prior,
            "weight_decay": args.weight_decay,
            "dropout": args.dropout,
            "hl_logits_grad_norm": best_checkpoint.get("hl_logits_grad_norm"),
        },
    )
    low_w = metrics.get("hl_low_weight")
    high_w = metrics.get("hl_high_weight")
    if low_w is not None and high_w is not None and abs(low_w - high_w) < 0.1:
        print("HL-FE weights are still close to 0.5/0.5. The module may not be learning useful frequency preference.")

    append_experiment_row(
        project_dirs["reports"] / "experiments.csv",
        {
            "timestamp": datetime.now().isoformat(timespec="seconds"),
            "model_name": args.model,
            "model_backend": model_backend,
            "use_hlfe": bool(args.use_hlfe),
            "requested_feature_type": requested_feature_type,
            "actual_feature_type": actual_feature_type,
            "feature_cache": bool(args.use_feature_cache),
            "allow_sst_fallback": bool(args.allow_sst_fallback),
            "epochs": args.epochs,
            "batch_size": args.batch_size,
            "lr": args.lr,
            "weight_decay": args.weight_decay,
            "dropout": args.dropout,
            "loss_type": args.loss_type,
            "focal_gamma": args.focal_gamma,
            "lambda_hl_prior": args.lambda_hl_prior,
            "seed": args.seed,
            "device": str(device),
            "best_epoch": best_epoch,
            "best_val_acc": best_acc,
            "test_acc": metrics["test_acc"],
            "macro_f1": metrics["macro_f1"],
            "weighted_f1": metrics["weighted_f1"],
            "hl_low_weight": low_w,
            "hl_high_weight": high_w,
            "checkpoint_path": str(best_path),
            "run_dir": str(run_dir),
        },
    )
    print(f"Best validation accuracy: {best_acc:.4f} at epoch {best_epoch}")
    print(f"Test accuracy from best_model.pt: {metrics['test_acc']:.4f}")
    print(f"Macro F1: {metrics['macro_f1']:.4f} Weighted F1: {metrics['weighted_f1']:.4f}")
    print(f"Run artifacts saved: {run_dir}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train STFT + HL-FE + BiGRU fallback dog emotion model.")
    parser.add_argument("--data_root", "--data_dir", dest="data_root", type=Path, required=True,
                        help="Root directory containing angry/anxious/happy/lonely/sad subdirectories.")
    parser.add_argument("--project_dir", type=Path, default=Path.cwd(),
                        help="Project directory for outputs (default: current working directory).")
    parser.add_argument("--exp_name", type=str, default=None,
                        help="Optional experiment name override. Auto-generates from feature/model if omitted.")
    parser.add_argument("--output_dir", type=Path, default=None)
    parser.add_argument("--model", choices=["cnn", "mamba"], default="mamba")
    parser.add_argument("--backend", choices=["auto", "bigru_fallback", "real_mamba_ssm"], default="bigru_fallback")
    parser.add_argument("--use_hlfe", "--use_hl_fe", dest="use_hlfe", action="store_true")
    parser.add_argument("--feature_type", choices=["stft", "sst_stft"], default="stft")
    parser.add_argument("--use_feature_cache", action="store_true")
    parser.add_argument("--rebuild_feature_cache", action="store_true")
    parser.add_argument("--allow_sst_fallback", action="store_true")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--weight_decay", type=float, default=1e-3)
    parser.add_argument("--dropout", type=float, default=0.5)
    parser.add_argument("--patience", type=int, default=4)
    parser.add_argument("--loss_type", choices=["ce", "weighted_ce", "focal"], default="weighted_ce")
    parser.add_argument("--lambda_hl_prior", type=float, default=0.0)
    parser.add_argument("--focal_gamma", type=float, default=1.5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num_workers", type=int, default=0)
    return parser.parse_args()

if __name__ == "__main__":
    train(parse_args())
