from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

SCRIPT_PATH = Path(__file__).resolve()
REPRO_DIR = SCRIPT_PATH.parents[2]
SRC_DIR = REPRO_DIR / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

try:
    from features import N_FFT, TARGET_SAMPLE_RATE
except Exception:
    N_FFT = 512
    TARGET_SAMPLE_RATE = 16000


def run(args: argparse.Namespace) -> None:
    args.output_dir.mkdir(parents=True, exist_ok=True)
    out_path = args.output_dir / "frequency_axis.csv"
    n_fft = args.n_fft or N_FFT
    sample_rate = args.sample_rate or TARGET_SAMPLE_RATE
    freqs = [i * sample_rate / n_fft for i in range(n_fft // 2 + 1)]
    with out_path.open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=["bin_index", "frequency_hz"])
        writer.writeheader()
        for i, freq in enumerate(freqs):
            writer.writerow({"bin_index": i, "frequency_hz": freq})
    print(f"Saved frequency axis: {out_path}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Create frequency bin to Hz table for STFT.")
    parser.add_argument("--output_dir", type=Path, default=REPRO_DIR / "explain_analysis" / "outputs" / "reports")
    parser.add_argument("--sample_rate", type=int, default=None)
    parser.add_argument("--n_fft", type=int, default=None)
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
