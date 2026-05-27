from __future__ import annotations

from pathlib import Path

import pandas as pd
import torch
from torch.utils.data import Dataset

from features import (
    HOP_LENGTH,
    N_FFT,
    TARGET_DURATION_SEC,
    TARGET_NUM_SAMPLES,
    TARGET_SAMPLE_RATE,
    WIN_LENGTH,
    feature_cache_path,
    extract_sst_stft_feature,
    extract_stft_feature,
)
from utils import normalize_waveform

try:
    import torchaudio
except Exception:
    torchaudio = None

import soundfile as sf

_SOUNDFILE_LOAD_FAILED = False


class DogEmotionDataset(Dataset):
    def __init__(
        self,
        csv_path: Path,
        sample_rate: int = TARGET_SAMPLE_RATE,
        num_samples: int = TARGET_NUM_SAMPLES,
        feature_type: str = "stft",
        cache_dir: Path | None = None,
        use_feature_cache: bool = False,
        rebuild_feature_cache: bool = False,
        allow_sst_fallback: bool = False,
    ) -> None:
        self.csv_path = Path(csv_path)
        self.df = pd.read_csv(self.csv_path)
        self.sample_rate = sample_rate
        self.num_samples = num_samples
        self.feature_type = feature_type
        self.cache_dir = Path(cache_dir) if cache_dir is not None else None
        self.use_feature_cache = use_feature_cache
        self.rebuild_feature_cache = rebuild_feature_cache
        self.allow_sst_fallback = allow_sst_fallback
        if self.cache_dir is not None and self.use_feature_cache:
            self.cache_dir.mkdir(parents=True, exist_ok=True)

    def __len__(self) -> int:
        return len(self.df)

    def _load_audio(self, path: Path) -> tuple[torch.Tensor, int]:
        global _SOUNDFILE_LOAD_FAILED
        try:
            data, sr = sf.read(str(path), dtype="float32")
            waveform = torch.tensor(data, dtype=torch.float32)
            if waveform.ndim == 2:
                waveform = waveform.mean(dim=1)
            elif waveform.ndim != 1:
                raise RuntimeError(f"Unsupported audio shape from soundfile: {tuple(waveform.shape)}")
            return waveform.unsqueeze(0), int(sr)
        except Exception as exc:
            if not _SOUNDFILE_LOAD_FAILED:
                _SOUNDFILE_LOAD_FAILED = True
                print(f"Warning: soundfile.read failed once; falling back to torchaudio for this run. Error: {exc.__class__.__name__}")
            if torchaudio is None:
                raise RuntimeError("Audio loading fallback requires torchaudio. Please install torchaudio.") from exc
            waveform, sr = torchaudio.load(str(path))
            waveform = waveform.float()
            if waveform.ndim == 2 and waveform.shape[0] > 1:
                waveform = waveform.mean(dim=0, keepdim=True)
            elif waveform.ndim == 1:
                waveform = waveform.unsqueeze(0)
            return waveform, int(sr)

    def _resample(self, waveform: torch.Tensor, sr: int) -> torch.Tensor:
        if sr == self.sample_rate:
            return waveform
        if torchaudio is None:
            raise RuntimeError("Resampling requires torchaudio. Please install torchaudio.")
        return torchaudio.functional.resample(waveform, sr, self.sample_rate)

    def _pad_or_truncate(self, waveform: torch.Tensor) -> torch.Tensor:
        current = waveform.shape[-1]
        if current < self.num_samples:
            waveform = torch.nn.functional.pad(waveform, (0, self.num_samples - current))
        elif current > self.num_samples:
            waveform = waveform[..., : self.num_samples]
        return waveform

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        row = self.df.iloc[idx]
        path = Path(row["filepath"])
        label = int(row["class_id"])
        cache_path = None
        if self.use_feature_cache and self.cache_dir is not None:
            cache_path = feature_cache_path(
                self.cache_dir,
                path,
                self.feature_type,
                sample_rate=self.sample_rate,
                duration_sec=TARGET_DURATION_SEC,
                n_fft=N_FFT,
                win_length=WIN_LENGTH,
                hop_length=HOP_LENGTH,
            )
            if cache_path.exists() and not self.rebuild_feature_cache:
                feature = torch.load(cache_path, map_location="cpu")
                return feature.float(), torch.tensor(label, dtype=torch.long)

        waveform, sr = self._load_audio(path)
        waveform = waveform.mean(dim=0)
        waveform = self._resample(waveform.unsqueeze(0), sr).squeeze(0)
        waveform = self._pad_or_truncate(waveform)
        waveform = normalize_waveform(waveform)

        if self.feature_type == "sst_stft":
            feature = extract_sst_stft_feature(waveform, sample_rate=self.sample_rate, allow_fallback=self.allow_sst_fallback)
        elif self.feature_type == "stft_fallback_from_sst":
            feature = extract_stft_feature(waveform)
        else:
            feature = extract_stft_feature(waveform)
        if cache_path is not None:
            torch.save(feature.cpu(), cache_path)
        return feature.float(), torch.tensor(label, dtype=torch.long)
