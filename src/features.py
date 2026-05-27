from __future__ import annotations

from pathlib import Path
import hashlib
import os

import numpy as np
import torch
import torch.nn.functional as F


N_FFT = 512
WIN_LENGTH = 480
HOP_LENGTH = 160
TARGET_SAMPLE_RATE = 16000
TARGET_DURATION_SEC = 4.0
TARGET_NUM_SAMPLES = int(TARGET_SAMPLE_RATE * TARGET_DURATION_SEC)
FREQ_BINS = N_FFT // 2 + 1
VALID_FEATURE_TYPES = ("stft", "sst_stft")
ACTUAL_FEATURE_TYPES = ("stft", "sst_stft", "stft_fallback_from_sst")


def _ssqueezepy_available() -> bool:
    try:
        os.environ.setdefault("NUMBA_DISABLE_JIT", "1")
        from ssqueezepy import ssq_stft  # noqa: F401

        return True
    except Exception:
        return False


def resolve_feature_type(requested_feature_type: str, allow_sst_fallback: bool = False) -> tuple[str, str]:
    requested = requested_feature_type.lower()
    if requested not in VALID_FEATURE_TYPES:
        raise ValueError(f"Unknown feature_type: {requested_feature_type}. Expected one of {VALID_FEATURE_TYPES}.")
    if requested == "sst_stft":
        if _ssqueezepy_available():
            return requested, "sst_stft"
        if allow_sst_fallback:
            return requested, "stft_fallback_from_sst"
        raise RuntimeError(
            "SST-STFT requires ssqueezepy, but it is not installed or could not be imported. "
            "Install it with: pip install ssqueezepy"
        )
    return requested, "stft"


def warn_if_sst_fallback(requested_feature_type: str, actual_feature_type: str) -> None:
    if requested_feature_type == "sst_stft" and actual_feature_type == "stft_fallback_from_sst":
        print("Warning: SST-STFT failed or is unavailable. Falling back to STFT because --allow_sst_fallback is set.")


def feature_display_name(feature_type: str) -> str:
    if feature_type == "sst_stft":
        return "SST-STFT"
    if feature_type == "stft_fallback_from_sst":
        return "STFT fallback from SST-STFT request"
    return "STFT"


def feature_cache_path(
    cache_dir: Path,
    wav_path: Path,
    feature_type: str,
    sample_rate: int = TARGET_SAMPLE_RATE,
    duration_sec: float = TARGET_DURATION_SEC,
    n_fft: int = N_FFT,
    win_length: int = WIN_LENGTH,
    hop_length: int = HOP_LENGTH,
) -> Path:
    key = "|".join(
        [
            str(Path(wav_path).resolve()).lower(),
            feature_type,
            str(sample_rate),
            f"{duration_sec:.6f}",
            str(n_fft),
            str(win_length),
            str(hop_length),
        ]
    )
    digest = hashlib.sha256(key.encode("utf-8")).hexdigest()[:24]
    safe_feature_type = feature_type.replace("stft_fallback_from_sst", "sst_fallback_stft")
    return Path(cache_dir) / f"{safe_feature_type}_{digest}.pt"


def extract_stft_feature(
    waveform: torch.Tensor,
    n_fft: int = N_FFT,
    win_length: int = WIN_LENGTH,
    hop_length: int = HOP_LENGTH,
) -> torch.Tensor:
    """Return log-magnitude STFT as [time, freq]."""
    if waveform.ndim == 2:
        waveform = waveform.mean(dim=0)
    waveform = waveform.float().flatten()
    window = torch.hann_window(win_length, device=waveform.device, dtype=waveform.dtype)
    stft = torch.stft(
        waveform,
        n_fft=n_fft,
        hop_length=hop_length,
        win_length=win_length,
        window=window,
        center=True,
        return_complex=True,
    )
    feature = torch.log1p(stft.abs()).transpose(0, 1).contiguous()
    return feature


def _resize_magnitude_to_reference(magnitude: torch.Tensor, target_time: int, target_freq: int) -> torch.Tensor:
    if magnitude.ndim != 2:
        raise RuntimeError(f"Expected 2D SST-STFT magnitude, got shape {tuple(magnitude.shape)}")
    magnitude = magnitude.float()
    if magnitude.shape == (target_time, target_freq):
        return magnitude.contiguous()
    if magnitude.shape == (target_freq, target_time):
        magnitude = magnitude.transpose(0, 1)
        return magnitude.contiguous()

    # ssqueezepy commonly returns [freq, time]. If neither axis matches exactly,
    # treat the smaller/first axis as frequency and interpolate to [257, time].
    if magnitude.shape[0] <= magnitude.shape[1]:
        freq_time = magnitude
    else:
        freq_time = magnitude.transpose(0, 1)
    resized = F.interpolate(
        freq_time.unsqueeze(0).unsqueeze(0),
        size=(target_freq, target_time),
        mode="bilinear",
        align_corners=False,
    ).squeeze(0).squeeze(0)
    return resized.transpose(0, 1).contiguous()


def extract_sst_stft_feature(
    waveform: torch.Tensor,
    sample_rate: int = TARGET_SAMPLE_RATE,
    n_fft: int = N_FFT,
    win_length: int = WIN_LENGTH,
    hop_length: int = HOP_LENGTH,
    allow_fallback: bool = False,
) -> torch.Tensor:
    """Return log-magnitude synchrosqueezed STFT as [time, freq].

    This uses ssqueezepy's ssq_stft. It only falls back to ordinary STFT when
    allow_fallback=True; otherwise SST-STFT dependency or computation failures
    raise an error and stop the experiment.
    """
    if waveform.ndim == 2:
        waveform = waveform.mean(dim=0)
    waveform = waveform.float().flatten().detach().cpu()
    reference = extract_stft_feature(waveform, n_fft=n_fft, win_length=win_length, hop_length=hop_length)
    target_time, target_freq = reference.shape

    try:
        os.environ.setdefault("NUMBA_DISABLE_JIT", "1")
        from ssqueezepy import ssq_stft
    except Exception as exc:
        if allow_fallback:
            print("Warning: SST-STFT unavailable. Falling back to STFT because --allow_sst_fallback is set.")
            return reference
        raise RuntimeError(
            "SST-STFT requires ssqueezepy, but it is not installed or could not be imported. "
            "Install it with: pip install ssqueezepy"
        ) from exc

    try:
        x = waveform.numpy().astype(np.float32, copy=False)
        result = ssq_stft(
            x,
            window="hann",
            n_fft=n_fft,
            win_len=win_length,
            hop_len=hop_length,
            fs=sample_rate,
            squeezing="sum",
        )
        tx = result[0] if isinstance(result, tuple) else result
        magnitude_np = np.abs(np.asarray(tx)).squeeze()
        magnitude = torch.from_numpy(magnitude_np.astype(np.float32, copy=False))
        magnitude = _resize_magnitude_to_reference(magnitude, target_time=target_time, target_freq=target_freq)
        return torch.log1p(magnitude).contiguous()
    except Exception as exc:
        if allow_fallback:
            print("Warning: SST-STFT computation failed. Falling back to STFT because --allow_sst_fallback is set.")
            return reference
        raise RuntimeError(
            "SST-STFT computation failed. Install or verify ssqueezepy with: pip install ssqueezepy"
        ) from exc
