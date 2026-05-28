from __future__ import annotations

import argparse
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd


DEFAULT_OUT_DIR = Path(r"E:\newdog_emo\dog_acoustic_analysis")
DEFAULT_FEATURE_CSV = DEFAULT_OUT_DIR / "features" / "acoustic_features_4s.csv"
DEFAULT_SEARCH_DIRS = [
    Path(r"E:\newdog_emo\dog_emo_repro\runs"),
    Path(r"E:\newdog_emo\dog_emo_repro\reports"),
    Path(r"E:\newdog_emo\reports"),
]
SEARCH_KEYWORDS = ["predictions", "test_predictions", "pred"]
KEY_FEATURES = [
    "rms_mean",
    "f0_mean",
    "spectral_centroid_mean",
    "onset_rate_per_sec",
    "log_energy",
    "spectral_bandwidth_mean",
    "f0_std",
    "f0_range",
    "energy_peak_count",
]
CLASS_NAMES = ["angry", "anxious", "happy", "lonely", "sad"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Merge handcrafted acoustic features with BiGRU predictions.")
    parser.add_argument("--feature_csv", type=Path, default=DEFAULT_FEATURE_CSV)
    parser.add_argument("--pred_csv", type=Path, default=None)
    parser.add_argument("--out_dir", type=Path, default=DEFAULT_OUT_DIR)
    return parser.parse_args()


def normalize_path_text(value: object) -> str:
    if pd.isna(value):
        return ""
    text = str(value).strip().strip('"').strip("'").replace("/", "\\")
    while "\\\\" in text:
        text = text.replace("\\\\", "\\")
    return text.lower()


def filename_from_value(value: object) -> str:
    text = normalize_path_text(value)
    if not text:
        return ""
    return Path(text).name.lower()


def find_prediction_csv() -> Path | None:
    candidates: list[Path] = []
    for directory in DEFAULT_SEARCH_DIRS:
        if not directory.exists():
            continue
        for path in directory.rglob("*.csv"):
            name = path.name.lower()
            if any(keyword in name for keyword in SEARCH_KEYWORDS):
                candidates.append(path)
    if not candidates:
        return None
    candidates.sort(key=lambda item: item.stat().st_mtime, reverse=True)
    return candidates[0]


def first_existing_column(columns: Iterable[str], candidates: Iterable[str]) -> str | None:
    lower_to_original = {col.lower(): col for col in columns}
    for candidate in candidates:
        if candidate.lower() in lower_to_original:
            return lower_to_original[candidate.lower()]
    return None


def parse_correct_column(series: pd.Series) -> pd.Series:
    if pd.api.types.is_bool_dtype(series):
        return series.astype("boolean")
    if pd.api.types.is_numeric_dtype(series):
        return series.astype(float).map(lambda value: bool(value) if np.isfinite(value) else pd.NA).astype("boolean")

    mapping = {
        "true": True,
        "t": True,
        "1": True,
        "yes": True,
        "y": True,
        "correct": True,
        "false": False,
        "f": False,
        "0": False,
        "no": False,
        "n": False,
        "wrong": False,
        "incorrect": False,
    }
    return series.astype(str).str.strip().str.lower().map(mapping).astype("boolean")


def standardize_prediction_columns(pred_df: pd.DataFrame) -> pd.DataFrame:
    pred_df = pred_df.copy()
    filepath_col = first_existing_column(pred_df.columns, ["filepath", "path", "file_path", "audio_path", "wav_path"])
    true_col = first_existing_column(pred_df.columns, ["true_label", "label", "y_true", "ground_truth", "target"])
    pred_col = first_existing_column(pred_df.columns, ["pred_label", "prediction", "pred", "y_pred", "predicted_label"])
    prob_col = first_existing_column(pred_df.columns, ["probability", "prob", "confidence", "score"])
    correct_col = first_existing_column(pred_df.columns, ["correct", "is_correct", "match"])

    if filepath_col is None:
        raise ValueError("Prediction csv must contain a filepath/path-like column.")

    output = pd.DataFrame()
    output["pred_filepath"] = pred_df[filepath_col].astype(str)
    output["merge_path"] = pred_df[filepath_col].map(normalize_path_text)
    output["merge_filename"] = pred_df[filepath_col].map(filename_from_value)
    output["pred_filename"] = output["merge_filename"]

    if true_col is not None:
        output["true_label"] = pred_df[true_col]
    if pred_col is not None:
        output["pred_label"] = pred_df[pred_col]
    if prob_col is not None:
        output["probability"] = pd.to_numeric(pred_df[prob_col], errors="coerce")

    if correct_col is not None:
        output["correct"] = parse_correct_column(pred_df[correct_col])
    elif true_col is not None and pred_col is not None:
        output["correct"] = (
            pred_df[true_col].astype(str).str.strip().str.lower()
            == pred_df[pred_col].astype(str).str.strip().str.lower()
        )
    else:
        output["correct"] = pd.NA

    return output.drop_duplicates(subset=["merge_path", "merge_filename"])


def prepare_feature_df(feature_df: pd.DataFrame) -> pd.DataFrame:
    feature_df = feature_df.copy()
    if "filepath" not in feature_df.columns:
        raise ValueError("Feature csv must contain a 'filepath' column.")
    if "filename" not in feature_df.columns:
        feature_df["filename"] = feature_df["filepath"].map(filename_from_value)
    feature_df["merge_path"] = feature_df["filepath"].map(normalize_path_text)
    feature_df["merge_filename"] = feature_df["filename"].astype(str).str.lower()
    return feature_df


def merge_features_and_predictions(feature_df: pd.DataFrame, pred_df: pd.DataFrame) -> pd.DataFrame:
    by_path = feature_df.merge(
        pred_df.drop(columns=["merge_filename"], errors="ignore"),
        how="left",
        on="merge_path",
        suffixes=("", "_pred"),
    )
    matched = by_path["pred_filepath"].notna() if "pred_filepath" in by_path.columns else pd.Series(False, index=by_path.index)

    unmatched_features = feature_df.loc[~matched].copy()
    if unmatched_features.empty:
        merged = by_path
    else:
        filename_pred = pred_df.drop(columns=["merge_path"], errors="ignore")
        by_filename = unmatched_features.merge(
            filename_pred,
            how="left",
            on="merge_filename",
            suffixes=("", "_pred"),
        )
        merged = pd.concat([by_path.loc[matched], by_filename], ignore_index=True, sort=False)

    return merged.drop(columns=["merge_path", "merge_filename"], errors="ignore")


def save_correct_wrong_summary(merged: pd.DataFrame, report_dir: Path) -> Path:
    available = [feature for feature in KEY_FEATURES if feature in merged.columns]
    summary_parts = []

    correct_summary = (
        merged.groupby("correct", dropna=False)[available]
        .mean(numeric_only=True)
        .reset_index()
    )
    correct_summary.insert(0, "grouping", "overall_correct_vs_wrong")
    correct_summary.insert(1, "label", "all")
    summary_parts.append(correct_summary)

    if "label" in merged.columns:
        class_summary = (
            merged.groupby(["label", "correct"], dropna=False)[available]
            .mean(numeric_only=True)
            .reset_index()
        )
        class_summary.insert(0, "grouping", "by_true_class")
        summary_parts.append(class_summary)

    summary = pd.concat(summary_parts, ignore_index=True, sort=False)
    output = report_dir / "correct_vs_wrong_feature_summary.csv"
    summary.to_csv(output, index=False, encoding="utf-8-sig")
    return output


def save_correct_wrong_plots(merged: pd.DataFrame, figure_dir: Path, report_dir: Path) -> tuple[list[Path], list[str], list[str]]:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    try:
        import seaborn as sns
    except Exception:
        sns = None

    plot_specs = [
        ("rms_mean", "correct_vs_wrong_rms_mean.png"),
        ("log_energy", "correct_vs_wrong_log_energy.png"),
        ("spectral_centroid_mean", "correct_vs_wrong_spectral_centroid_mean.png"),
        ("spectral_bandwidth_mean", "correct_vs_wrong_spectral_bandwidth_mean.png"),
        ("spectral_rolloff_mean", "correct_vs_wrong_spectral_rolloff_mean.png"),
        ("onset_rate_per_sec", "correct_vs_wrong_onset_rate_per_sec.png"),
        ("onset_count", "correct_vs_wrong_onset_count.png"),
        ("energy_peak_count", "correct_vs_wrong_energy_peak_count.png"),
    ]
    outputs: list[Path] = []
    failed_features: list[str] = []
    skipped_features: list[str] = []

    if "correct" not in merged.columns:
        print("[WARN] No 'correct' column in merged data; skipping all plots.")
        return outputs, failed_features, skipped_features

    base_df = merged.dropna(subset=["correct"]).copy()
    if base_df.empty:
        print("[WARN] No rows with valid 'correct' values; skipping all plots.")
        return outputs, failed_features, skipped_features
    base_df["correct_group"] = base_df["correct"].map({True: "correct", False: "wrong"})

    for feature, filename in plot_specs:
        print(f"  [PLOT] {feature}")
        # --- per-feature validation ---
        if feature not in merged.columns:
            skipped_features.append(feature)
            print(f"  [SKIP] {feature} not found in merged columns.")
            continue

        plot_df = base_df.copy()

        # Convert to numeric, drop NaNs introduced by conversion
        plot_df[feature] = pd.to_numeric(plot_df[feature], errors="coerce")
        plot_df = plot_df.dropna(subset=[feature])

        # Drop rows where correct_group is empty/NA
        plot_df = plot_df.dropna(subset=["correct_group"])

        # Keep only known groups
        plot_df = plot_df[plot_df["correct_group"].isin(["correct", "wrong"])]

        if plot_df.empty:
            skipped_features.append(feature)
            print(f"  [SKIP] {feature} has no valid values.")
            continue

        unique_groups = plot_df["correct_group"].unique()
        if len(unique_groups) < 2:
            skipped_features.append(feature)
            print(f"  [SKIP] {feature} has fewer than 2 valid groups (groups: {list(unique_groups)}).")
            continue

        group_order = [g for g in ["correct", "wrong"] if g in unique_groups]

        # --- try/except per feature ---
        try:
            plt.figure(figsize=(6.5, 4.5), dpi=150)
            if sns is not None:
                sns.boxplot(
                    data=plot_df,
                    x="correct_group",
                    y=feature,
                    order=group_order,
                    color="#86a873",
                )
                sns.stripplot(
                    data=plot_df,
                    x="correct_group",
                    y=feature,
                    order=group_order,
                    color="#2f2f2f",
                    size=2,
                    alpha=0.35,
                )
            else:
                plot_df.boxplot(column=feature, by="correct_group", grid=False)
                plt.suptitle("")
            plt.title(f"{feature}: correct vs wrong")
            plt.xlabel("Prediction result")
            plt.ylabel(feature)
            plt.tight_layout()
            output = figure_dir / filename
            plt.savefig(output)
            plt.close()
            outputs.append(output)
            print(f"  [OK] saved: {output}")
        except Exception as exc:
            plt.close("all")
            failed_features.append(feature)
            print(f"  [FAIL] {feature} plotting raised {type(exc).__name__}: {exc}")

    # Write failed features log
    if failed_features:
        failed_log = report_dir / "merge_plot_failed_features.txt"
        failed_log.parent.mkdir(parents=True, exist_ok=True)
        with open(failed_log, "w", encoding="utf-8") as f:
            for feat in failed_features:
                f.write(feat + "\n")
        print(f"  Failed features written to: {failed_log}")

    return outputs, failed_features, skipped_features

def main() -> None:
    args = parse_args()
    pred_csv = args.pred_csv or find_prediction_csv()
    if pred_csv is None:
        print("未找到 BiGRU 预测结果，请使用 --pred_csv 手动指定预测结果 csv 路径。")
        return
    if not pred_csv.exists():
        print(f"Prediction csv not found: {pred_csv}")
        print("未找到 BiGRU 预测结果，请使用 --pred_csv 手动指定预测结果 csv 路径。")
        return

    feature_dir = args.out_dir / "features"
    report_dir = args.out_dir / "reports"
    figure_dir = args.out_dir / "figures"
    feature_dir.mkdir(parents=True, exist_ok=True)
    report_dir.mkdir(parents=True, exist_ok=True)
    figure_dir.mkdir(parents=True, exist_ok=True)

    feature_df = prepare_feature_df(pd.read_csv(args.feature_csv))
    pred_df = standardize_prediction_columns(pd.read_csv(pred_csv))
    merged = merge_features_and_predictions(feature_df, pred_df)

    output_csv = feature_dir / "features_with_bigru_predictions.csv"
    merged.to_csv(output_csv, index=False, encoding="utf-8-sig")

    matched_count = int(merged["pred_filepath"].notna().sum()) if "pred_filepath" in merged.columns else 0
    print("Prediction csv:", pred_csv)
    print("Feature rows:", len(feature_df))
    print("Matched rows:", matched_count)
    print("Merged output:", output_csv)

    if matched_count == 0:
        print("No rows matched by filepath or filename. Please check path/filename consistency.")
        return

    summary_path = save_correct_wrong_summary(merged, report_dir)
    plot_paths, failed_features, skipped_features = save_correct_wrong_plots(merged, figure_dir, report_dir)

    success_count = len(plot_paths)
    skip_count = len(skipped_features)
    fail_count = len(failed_features)

    print("=" * 50)
    print("Correct-vs-wrong plots saved:", success_count)
    print("Skipped/failed plots:", skip_count + fail_count)
    print("Correct vs wrong summary:", summary_path)
    print("Merged CSV:", output_csv)
    print("Summary CSV:", summary_path)
    if success_count > 0:
        print("Figures saved to:", figure_dir)
    print("=" * 50)

if __name__ == "__main__":
    main()

