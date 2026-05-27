from __future__ import annotations

import argparse
import wave
from pathlib import Path

import pandas as pd
import soundfile as sf

from utils import ensure_project_dirs


def _duration_with_soundfile(path: Path) -> tuple[float, int]:
    info = sf.info(str(path))
    return float(info.frames) / float(info.samplerate), int(info.samplerate)


def _duration_with_wave(path: Path) -> tuple[float, int]:
    with wave.open(str(path), "rb") as handle:
        sample_rate = handle.getframerate()
        frames = handle.getnframes()
    return float(frames) / float(sample_rate), int(sample_rate)


def scan_dataset(data_root: Path, project_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    data_root = Path(data_root)
    project_dir = Path(project_dir)
    dirs = ensure_project_dirs(project_dir)
    project_name = project_dir.name.lower()

    class_dirs = []
    for p in sorted(data_root.iterdir(), key=lambda x: x.name.lower()):
        if not p.is_dir() or p.name.lower() == project_name or p.name.startswith("."):
            continue
        if any(p.glob("*.wav")):
            class_dirs.append(p)

    rows: list[dict] = []
    bad_rows: list[dict] = []
    class_names = [p.name for p in class_dirs]
    class_to_id = {name: idx for idx, name in enumerate(class_names)}

    for class_dir in class_dirs:
        wav_paths = sorted(class_dir.glob("*.wav"), key=lambda x: x.name.lower())
        for wav_path in wav_paths:
            try:
                try:
                    duration_sec, sample_rate = _duration_with_soundfile(wav_path)
                except Exception:
                    duration_sec, sample_rate = _duration_with_wave(wav_path)
                rows.append(
                    {
                        "filepath": str(wav_path.resolve()),
                        "label": class_dir.name,
                        "class_id": class_to_id[class_dir.name],
                        "duration_sec": duration_sec,
                        "sample_rate": sample_rate,
                    }
                )
            except Exception as exc:
                bad_rows.append(
                    {
                        "filepath": str(wav_path.resolve()),
                        "label": class_dir.name,
                        "class_id": class_to_id[class_dir.name],
                        "error": repr(exc),
                    }
                )

    df = pd.DataFrame(rows, columns=["filepath", "label", "class_id", "duration_sec", "sample_rate"])
    bad_df = pd.DataFrame(bad_rows, columns=["filepath", "label", "class_id", "error"])

    index_path = dirs["reports"] / "dataset_index.csv"
    bad_path = dirs["reports"] / "bad_files.csv"
    df.to_csv(index_path, index=False, encoding="utf-8-sig")
    bad_df.to_csv(bad_path, index=False, encoding="utf-8-sig")

    print(f"Saved dataset index: {index_path}")
    print(f"Saved bad file list: {bad_path}")
    print("\nClass counts:")
    if not df.empty:
        print(df.groupby(["class_id", "label"]).size().rename("count").to_string())
        print("\nDuration statistics:")
        print(df["duration_sec"].describe().to_string())
        print("\nSample-rate counts:")
        print(df["sample_rate"].value_counts().sort_index().to_string())
    print(f"\nBad files skipped: {len(bad_df)}")
    return df, bad_df


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Scan dog emotion wav dataset.")
    parser.add_argument("--data_root", type=Path, required=True,
                        help="Root directory containing angry/anxious/happy/lonely/sad subdirectories.")
    parser.add_argument("--project_dir", type=Path, default=Path.cwd(),
                        help="Project directory for outputs (default: current working directory).")
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()
    scan_dataset(args.data_root, args.project_dir)
