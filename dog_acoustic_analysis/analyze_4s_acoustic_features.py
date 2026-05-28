from __future__ import annotations

import argparse
import math
from pathlib import Path

import numpy as np
import pandas as pd


DEFAULT_OUT_DIR = Path(r"E:\newdog_emo\dog_acoustic_analysis")
DEFAULT_FEATURE_CSV = DEFAULT_OUT_DIR / "features" / "acoustic_features_4s.csv"
CLASS_NAMES = ["angry", "anxious", "happy", "lonely", "sad"]
KRUSKAL_FEATURES = [
    "rms_mean",
    "log_energy",
    "spectral_centroid_mean",
    "spectral_bandwidth_mean",
    "spectral_rolloff_mean",
    "f0_mean",
    "f0_std",
    "f0_range",
    "f0_voiced_ratio",
    "onset_rate_per_sec",
    "onset_count",
    "energy_peak_count",
]
BOX_FEATURES = [
    "rms_mean",
    "log_energy",
    "spectral_centroid_mean",
    "spectral_bandwidth_mean",
    "spectral_rolloff_mean",
    "f0_mean",
    "f0_std",
    "f0_range",
    "onset_rate_per_sec",
    "onset_count",
    "energy_peak_count",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Analyze 4-second dog acoustic features by emotion class.")
    parser.add_argument("--feature_csv", type=Path, default=DEFAULT_FEATURE_CSV)
    parser.add_argument("--out_dir", type=Path, default=DEFAULT_OUT_DIR)
    return parser.parse_args()


def numeric_feature_columns(df: pd.DataFrame) -> list[str]:
    excluded = {"filepath", "filename", "label", "sample_rate", "duration_sec"}
    return [
        col
        for col in df.columns
        if col not in excluded and pd.api.types.is_numeric_dtype(df[col])
    ]


def save_feature_summary(df: pd.DataFrame, report_dir: Path) -> Path:
    columns = numeric_feature_columns(df)
    summary = df.groupby("label")[columns].agg(["mean", "std", "median", "min", "max"])
    summary.columns = [f"{feature}_{stat}" for feature, stat in summary.columns]
    summary = summary.reindex(CLASS_NAMES)
    output = report_dir / "feature_summary_by_class.csv"
    summary.to_csv(output, encoding="utf-8-sig")
    return output


def run_kruskal_tests(df: pd.DataFrame, report_dir: Path) -> tuple[Path, pd.DataFrame]:
    from scipy.stats import kruskal

    rows = []
    for feature in KRUSKAL_FEATURES:
        if feature not in df.columns:
            rows.append(
                {
                    "feature": feature,
                    "statistic": math.nan,
                    "p_value": math.nan,
                    "significant_p_lt_0_05": False,
                    "note": "feature not found",
                }
            )
            continue

        groups = [
            pd.to_numeric(df.loc[df["label"] == label, feature], errors="coerce").dropna().to_numpy()
            for label in CLASS_NAMES
        ]
        valid_groups = [group for group in groups if len(group) > 0]
        if len(valid_groups) < 2:
            rows.append(
                {
                    "feature": feature,
                    "statistic": math.nan,
                    "p_value": math.nan,
                    "significant_p_lt_0_05": False,
                    "note": "not enough valid groups",
                }
            )
            continue

        try:
            statistic, p_value = kruskal(*valid_groups, nan_policy="omit")
            note = ""
        except Exception as exc:
            statistic, p_value = math.nan, math.nan
            note = str(exc)

        rows.append(
            {
                "feature": feature,
                "statistic": statistic,
                "p_value": p_value,
                "significant_p_lt_0_05": bool(np.isfinite(p_value) and p_value < 0.05),
                "note": note,
            }
        )

    result = pd.DataFrame(rows)
    output = report_dir / "kruskal_test_results.csv"
    result.to_csv(output, index=False, encoding="utf-8-sig")
    return output, result


def save_boxplots(df: pd.DataFrame, figure_dir: Path, report_dir: Path) -> list[Path]:
    import matplotlib.pyplot as plt

    try:
        import seaborn as sns
    except Exception:
        sns = None

    outputs: list[Path] = []
    failures: list[str] = []
    base_df = df[df["label"].isin(CLASS_NAMES)].copy()
    for feature in BOX_FEATURES:
        if feature not in base_df.columns:
            print(f"[SKIP] {feature} not found, skip boxplot.")
            continue

        plot_df = base_df[["label", feature]].copy()
        plot_df[feature] = pd.to_numeric(plot_df[feature], errors="coerce")
        plot_df = plot_df.dropna(subset=[feature])

        if plot_df.empty:
            print(f"[SKIP] {feature} has no valid values, skip boxplot.")
            continue

        valid_labels = [label for label in CLASS_NAMES if (plot_df["label"] == label).any()]
        if len(valid_labels) < 2:
            print(f"[SKIP] {feature} has fewer than 2 valid classes, skip boxplot.")
            continue

        try:
            plt.figure(figsize=(8, 5), dpi=150)
            if sns is not None:
                sns.boxplot(data=plot_df, x="label", y=feature, order=valid_labels, color="#7aa6c2")
                sns.stripplot(
                    data=plot_df,
                    x="label",
                    y=feature,
                    order=valid_labels,
                    color="#2f2f2f",
                    size=2,
                    alpha=0.35,
                )
            else:
                grouped_values = [
                    plot_df.loc[plot_df["label"] == label, feature].to_numpy()
                    for label in valid_labels
                ]
                plt.boxplot(grouped_values, labels=valid_labels)
                plt.suptitle("")
            plt.title(f"{feature} by emotion class")
            plt.xlabel("Emotion class")
            plt.ylabel(feature)
            plt.tight_layout()
            output = figure_dir / f"box_{feature}.png"
            plt.savefig(output)
            outputs.append(output)
        except Exception as exc:
            failures.append(f"{feature}\t{exc}")
            print(f"[WARN] {feature} boxplot failed: {exc}")
        finally:
            plt.close()

    failed_path = report_dir / "boxplot_failed_features.txt"
    if failures:
        failed_path.write_text("\n".join(failures), encoding="utf-8")
    else:
        failed_path.write_text("No failed boxplots.\n", encoding="utf-8")
    return outputs


def valid_key_features(df: pd.DataFrame) -> list[str]:
    features: list[str] = []
    for feature in KRUSKAL_FEATURES:
        if feature not in df.columns:
            continue
        values = pd.to_numeric(df[feature], errors="coerce")
        if values.notna().any():
            features.append(feature)
    return features


def save_heatmap(df: pd.DataFrame, figure_dir: Path) -> tuple[Path | None, str]:
    import matplotlib.pyplot as plt

    try:
        import seaborn as sns
    except Exception:
        sns = None

    available = valid_key_features(df)
    if not available:
        return None, "Heatmap skipped: no key feature had valid numeric values."

    heatmap_df = df[df["label"].isin(CLASS_NAMES)].copy()
    for feature in available:
        heatmap_df[feature] = pd.to_numeric(heatmap_df[feature], errors="coerce")

    class_means = heatmap_df.groupby("label")[available].mean(numeric_only=True).reindex(CLASS_NAMES)
    class_means = class_means.dropna(axis=1, how="all")
    if class_means.empty or class_means.shape[1] == 0:
        return None, "Heatmap skipped: class means were all NaN after filtering."

    zscore = class_means.copy()
    for feature in class_means.columns:
        mean = zscore[feature].mean(skipna=True)
        std = zscore[feature].std(skipna=True, ddof=0)
        if not np.isfinite(mean):
            zscore[feature] = np.nan
        elif not np.isfinite(std) or std == 0:
            zscore[feature] = 0.0
        else:
            zscore[feature] = (zscore[feature] - mean) / std
    zscore = zscore.dropna(axis=1, how="all")
    if zscore.empty or zscore.shape[1] == 0:
        return None, "Heatmap skipped: no usable z-scored feature remained."

    plt.figure(figsize=(12, 5.5), dpi=150)
    if sns is not None:
        sns.heatmap(
            zscore,
            cmap="vlag",
            center=0,
            annot=True,
            fmt=".2f",
            linewidths=0.5,
            cbar_kws={"label": "Z-scored class mean"},
        )
    else:
        image = plt.imshow(zscore.to_numpy(dtype=float), aspect="auto", cmap="coolwarm")
        plt.colorbar(image, label="Z-scored class mean")
        plt.xticks(range(len(zscore.columns)), zscore.columns, rotation=45, ha="right")
        plt.yticks(range(len(CLASS_NAMES)), CLASS_NAMES)
    plt.title("Class-level acoustic feature mean heatmap")
    plt.xlabel("Acoustic feature")
    plt.ylabel("Emotion class")
    plt.tight_layout()
    output = figure_dir / "class_feature_mean_heatmap.png"
    plt.savefig(output)
    plt.close()
    return output, ""


def describe_class_patterns(df: pd.DataFrame) -> list[str]:
    available = valid_key_features(df)
    if not available:
        return ["No key features were available for class pattern description."]

    df = df.copy()
    for feature in available:
        df[feature] = pd.to_numeric(df[feature], errors="coerce")
    means = df.groupby("label")[available].mean(numeric_only=True).reindex(CLASS_NAMES)
    means = means.dropna(axis=1, how="all")
    if means.empty or means.shape[1] == 0:
        return ["No key features had usable class means for pattern description."]

    zscore = means.copy()
    for feature in means.columns:
        feature_mean = zscore[feature].mean(skipna=True)
        feature_std = zscore[feature].std(skipna=True, ddof=0)
        if not np.isfinite(feature_mean):
            zscore[feature] = np.nan
        elif not np.isfinite(feature_std) or feature_std == 0:
            zscore[feature] = 0.0
        else:
            zscore[feature] = (zscore[feature] - feature_mean) / feature_std

    lines: list[str] = []
    for label in CLASS_NAMES:
        if label not in zscore.index:
            continue
        row = zscore.loc[label].dropna()
        if row.empty:
            lines.append(f"- {label}: no valid key feature means available.")
            continue
        high = row[row >= 0.5].sort_values(ascending=False).index.tolist()
        low = row[row <= -0.5].sort_values().index.tolist()
        if not high:
            high = row.sort_values(ascending=False).head(3).index.tolist()
        if not low:
            low = row.sort_values().head(3).index.tolist()
        lines.append(f"- {label}: higher mean features: {', '.join(high)}; lower mean features: {', '.join(low)}.")
    return lines


def f0_features_all_empty(df: pd.DataFrame) -> bool:
    f0_features = ["f0_mean", "f0_std", "f0_min", "f0_max", "f0_range", "f0_voiced_ratio"]
    present = [feature for feature in f0_features if feature in df.columns]
    if not present:
        return False
    return all(pd.to_numeric(df[feature], errors="coerce").dropna().empty for feature in present)


def save_text_report(
    df: pd.DataFrame,
    kruskal_df: pd.DataFrame,
    report_dir: Path,
    heatmap_note: str = "",
    boxplot_filenames: list[str] | None = None,
) -> Path:
    significant = kruskal_df.loc[kruskal_df["significant_p_lt_0_05"], "feature"].tolist()
    output = report_dir / "analysis_summary.txt"
    lines = [
        "4-second handcrafted acoustic feature analysis summary",
        "",
        "Significant differences across the five emotion classes (Kruskal-Wallis, p < 0.05):",
    ]
    if significant:
        lines.append(", ".join(significant))
    else:
        lines.append("No key feature reached p < 0.05 in this run.")

    lines.extend(
        [
            "",
            "Class-level mean patterns based on z-scored class means:",
            *describe_class_patterns(df),
            "",
            "Missing feature note:",
        ]
    )
    if f0_features_all_empty(df):
        lines.append(
            "F0 features are all empty; this may happen when feature extraction was run with --skip_f0."
        )
    lines.extend(
        [
            "The current analysis can still use RMS, spectral, MFCC, onset, and other non-F0 acoustic features.",
        ]
    )
    if heatmap_note:
        lines.extend(["", heatmap_note])

    if boxplot_filenames:
        lines.extend(
            [
                "",
                "Generated boxplot figures:",
                *[f"  - {name}" for name in boxplot_filenames],
            ]
        )

    lines.extend(
        [
            "",
            "Interpretation note:",
            "These results describe statistical associations in the current dataset. They should not be interpreted as direct proof of true canine emotional mechanisms.",
        ]
    )
    output.write_text("\n".join(lines), encoding="utf-8")
    return output


def main() -> None:
    args = parse_args()
    report_dir = args.out_dir / "reports"
    figure_dir = args.out_dir / "figures"
    report_dir.mkdir(parents=True, exist_ok=True)
    figure_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(args.feature_csv)
    if "label" not in df.columns:
        raise ValueError("Feature csv must contain a 'label' column.")

    summary_path = save_feature_summary(df, report_dir)
    kruskal_path, kruskal_df = run_kruskal_tests(df, report_dir)
    boxplot_paths = save_boxplots(df, figure_dir, report_dir)
    heatmap_path, heatmap_note = save_heatmap(df, figure_dir)
    if heatmap_note:
        print("[SKIP]", heatmap_note)
    boxplot_names = [p.name for p in boxplot_paths]
    report_path = save_text_report(df, kruskal_df, report_dir, heatmap_note, boxplot_filenames=boxplot_names)

    print("Feature summary:", summary_path)
    print("Kruskal-Wallis results:", kruskal_path)
    print("Boxplots:", len(boxplot_paths), "files saved to", figure_dir)
    print("Heatmap:", heatmap_path if heatmap_path is not None else "skipped")
    print("Text report:", report_path)


if __name__ == "__main__":
    main()
