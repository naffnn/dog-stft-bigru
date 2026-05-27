from __future__ import annotations

import argparse
import math
import warnings
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


DEFAULT_DATA_ROOT = Path(r"E:\newdog_emo")
DEFAULT_OUT_DIR = DEFAULT_DATA_ROOT / "dog_acoustic_analysis"
CLASS_NAMES = ["angry", "anxious", "happy", "lonely", "sad"]
AUDIO_EXTENSIONS = {".wav", ".mp3", ".flac"}
EPS = 1e-10


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Extract fixed 4-second handcrafted acoustic features for dog emotion audio."
    )
    parser.add_argument("--data_root", type=Path, default=DEFAULT_DATA_ROOT)
    parser.add_argument("--out_dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--sample_rate", type=int, default=16000)
    parser.add_argument("--duration", type=float, default=4.0)
    return parser.parse_args()


def to_mono(audio: np.ndarray) -> np.ndarray:
    audio = np.asarray(audio, dtype=np.float32)
    if audio.ndim == 1:
        return audio
    return np.mean(audio, axis=1).astype(np.float32)


def fix_length(audio: np.ndarray, target_samples: int) -> np.ndarray:
    if len(audio) >= target_samples:
        return audio[:target_samples].astype(np.float32, copy=False)
    padded = np.zeros(target_samples, dtype=np.float32)
    padded[: len(audio)] = audio
    return padded


def load_audio(path: Path, target_sr: int, duration: float) -> np.ndarray:
    import librosa

    audio: np.ndarray
    sample_rate: int

    try:
        import soundfile as sf

        audio, sample_rate = sf.read(str(path), dtype="float32", always_2d=False)
        audio = to_mono(audio)
        if sample_rate != target_sr:
            audio = librosa.resample(audio, orig_sr=sample_rate, target_sr=target_sr)
    except Exception:
        audio, _ = librosa.load(str(path), sr=target_sr, mono=True)

    target_samples = int(round(target_sr * duration))
    return fix_length(np.asarray(audio, dtype=np.float32), target_samples)


def safe_mean(values: np.ndarray) -> float:
    values = np.asarray(values, dtype=float)
    finite = values[np.isfinite(values)]
    if finite.size == 0:
        return math.nan
    return float(np.mean(finite))


def safe_std(values: np.ndarray) -> float:
    values = np.asarray(values, dtype=float)
    finite = values[np.isfinite(values)]
    if finite.size == 0:
        return math.nan
    return float(np.std(finite, ddof=0))


def safe_min(values: np.ndarray) -> float:
    values = np.asarray(values, dtype=float)
    finite = values[np.isfinite(values)]
    if finite.size == 0:
        return math.nan
    return float(np.min(finite))


def safe_max(values: np.ndarray) -> float:
    values = np.asarray(values, dtype=float)
    finite = values[np.isfinite(values)]
    if finite.size == 0:
        return math.nan
    return float(np.max(finite))


def estimate_f0(audio: np.ndarray, sample_rate: int) -> dict[str, float]:
    import librosa

    empty = {
        "f0_mean": math.nan,
        "f0_std": math.nan,
        "f0_min": math.nan,
        "f0_max": math.nan,
        "f0_range": math.nan,
        "f0_voiced_ratio": math.nan,
    }
    try:
        fmax = min(2000.0, sample_rate / 2.0 - 1.0)
        f0, voiced_flag, _ = librosa.pyin(
            audio,
            fmin=50.0,
            fmax=fmax,
            sr=sample_rate,
            frame_length=1024,
            hop_length=256,
        )
        voiced = np.asarray(f0, dtype=float)[np.isfinite(f0)]
        voiced_ratio = float(np.mean(voiced_flag)) if voiced_flag is not None and len(voiced_flag) else math.nan
        if voiced.size == 0:
            empty["f0_voiced_ratio"] = voiced_ratio
            return empty
        f0_min = safe_min(voiced)
        f0_max = safe_max(voiced)
        return {
            "f0_mean": safe_mean(voiced),
            "f0_std": safe_std(voiced),
            "f0_min": f0_min,
            "f0_max": f0_max,
            "f0_range": float(f0_max - f0_min) if np.isfinite(f0_min) and np.isfinite(f0_max) else math.nan,
            "f0_voiced_ratio": voiced_ratio,
        }
    except Exception:
        return empty


def count_energy_peaks(rms: np.ndarray) -> int:
    envelope = np.asarray(rms, dtype=float).reshape(-1)
    if envelope.size < 3 or not np.isfinite(envelope).any():
        return 0

    threshold = float(np.nanmean(envelope) + np.nanstd(envelope))
    try:
        from scipy.signal import find_peaks

        peaks, _ = find_peaks(envelope, height=threshold, distance=2)
        return int(len(peaks))
    except Exception:
        middle = envelope[1:-1]
        peaks = (middle > envelope[:-2]) & (middle >= envelope[2:]) & (middle >= threshold)
        return int(np.sum(peaks))


def extract_features(path: Path, label: str, sample_rate: int, duration: float) -> dict[str, Any]:
    import librosa

    audio = load_audio(path, sample_rate, duration)
    frame_length = 1024
    hop_length = 256

    rms = librosa.feature.rms(y=audio, frame_length=frame_length, hop_length=hop_length)[0]
    zcr = librosa.feature.zero_crossing_rate(y=audio, frame_length=frame_length, hop_length=hop_length)[0]
    centroid = librosa.feature.spectral_centroid(y=audio, sr=sample_rate, hop_length=hop_length)[0]
    bandwidth = librosa.feature.spectral_bandwidth(y=audio, sr=sample_rate, hop_length=hop_length)[0]
    rolloff = librosa.feature.spectral_rolloff(y=audio, sr=sample_rate, hop_length=hop_length)[0]
    flatness = librosa.feature.spectral_flatness(y=audio, hop_length=hop_length)[0]
    mfcc = librosa.feature.mfcc(y=audio, sr=sample_rate, n_mfcc=13, hop_length=hop_length)

    onset_frames = librosa.onset.onset_detect(y=audio, sr=sample_rate, hop_length=hop_length, units="frames")
    onset_times = librosa.frames_to_time(onset_frames, sr=sample_rate, hop_length=hop_length)
    onset_intervals = np.diff(onset_times) if len(onset_times) >= 2 else np.array([], dtype=float)

    row: dict[str, Any] = {
        "filepath": str(path),
        "filename": path.name,
        "label": label,
        "sample_rate": sample_rate,
        "duration_sec": duration,
        "rms_mean": safe_mean(rms),
        "rms_std": safe_std(rms),
        "rms_max": safe_max(rms),
        "log_energy": float(np.log(np.sum(np.square(audio, dtype=np.float64)) + EPS)),
        "zero_crossing_rate_mean": safe_mean(zcr),
        "zero_crossing_rate_std": safe_std(zcr),
        "spectral_centroid_mean": safe_mean(centroid),
        "spectral_centroid_std": safe_std(centroid),
        "spectral_bandwidth_mean": safe_mean(bandwidth),
        "spectral_bandwidth_std": safe_std(bandwidth),
        "spectral_rolloff_mean": safe_mean(rolloff),
        "spectral_rolloff_std": safe_std(rolloff),
        "spectral_flatness_mean": safe_mean(flatness),
        "spectral_flatness_std": safe_std(flatness),
        "onset_count": int(len(onset_times)),
        "onset_rate_per_sec": float(len(onset_times) / duration) if duration > 0 else math.nan,
        "onset_interval_mean": safe_mean(onset_intervals),
        "onset_interval_std": safe_std(onset_intervals),
        "energy_peak_count": count_energy_peaks(rms),
    }

    for i in range(13):
        row[f"mfcc_{i + 1}_mean"] = safe_mean(mfcc[i])
    for i in range(13):
        row[f"mfcc_{i + 1}_std"] = safe_std(mfcc[i])

    row.update(estimate_f0(audio, sample_rate))
    return row


def iter_audio_files(data_root: Path) -> list[tuple[Path, str]]:
    files: list[tuple[Path, str]] = []
    for label in CLASS_NAMES:
        class_dir = data_root / label
        if not class_dir.exists():
            print(f"Warning: class directory not found: {class_dir}")
            continue
        for path in sorted(class_dir.rglob("*")):
            if path.is_file() and path.suffix.lower() in AUDIO_EXTENSIONS:
                files.append((path, label))
    return files


def main() -> None:
    args = parse_args()
    data_root = args.data_root
    out_dir = args.out_dir
    feature_dir = out_dir / "features"
    feature_dir.mkdir(parents=True, exist_ok=True)

    audio_files = iter_audio_files(data_root)
    rows: list[dict[str, Any]] = []
    failures: list[tuple[Path, str]] = []

    warnings.filterwarnings("ignore", category=UserWarning)

    for index, (path, label) in enumerate(audio_files, start=1):
        try:
            rows.append(extract_features(path, label, args.sample_rate, args.duration))
        except Exception as exc:
            failures.append((path, str(exc)))
            print(f"[{index}/{len(audio_files)}] Failed: {path} ({exc})")

    output_csv = feature_dir / "acoustic_features_4s.csv"
    pd.DataFrame(rows).to_csv(output_csv, index=False, encoding="utf-8-sig")

    label_counts = Counter(label for _, label in audio_files)
    print("Total samples:", len(audio_files))
    print("Samples per class:")
    for label in CLASS_NAMES:
        print(f"  {label}: {label_counts.get(label, 0)}")
    print("Successfully extracted:", len(rows))
    print("Failed files:")
    if failures:
        for failed_path, reason in failures:
            print(f"  {failed_path} | {reason}")
    else:
        print("  None")
    print("Output csv:", output_csv)


if __name__ == "__main__":
    main()
