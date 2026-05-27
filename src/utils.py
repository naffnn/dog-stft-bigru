from __future__ import annotations

import random
from pathlib import Path
from typing import Iterable

import numpy as np
import torch


def ensure_project_dirs(project_dir: Path) -> dict[str, Path]:
    project_dir = Path(project_dir)
    dirs = {
        "logs": project_dir / "outputs" / "logs",
        "checkpoints": project_dir / "outputs" / "checkpoints",
        "figures": project_dir / "outputs" / "figures",
        "reports": project_dir / "outputs" / "reports",
        "features_cache": project_dir / "outputs" / "features_cache",
    }
    for path in [project_dir / "src", project_dir / "outputs", *dirs.values()]:
        path.mkdir(parents=True, exist_ok=True)
    return dirs


def set_seed(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def get_device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def class_names_from_dataframe(df) -> list[str]:
    pairs = df[["class_id", "label"]].drop_duplicates().sort_values("class_id")
    return pairs["label"].astype(str).tolist()


def save_lines(path: Path, lines: Iterable[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def normalize_waveform(waveform: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    waveform = waveform - waveform.mean()
    peak = waveform.abs().max()
    if peak > eps:
        waveform = waveform / peak
    return waveform

