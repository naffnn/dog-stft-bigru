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
    "f0_mean",
    "f0_std",
    "f0_range",
    "onset_rate_per_sec",
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
                    "note": "fewer than two valid class groups",
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


def save_boxplots(df: pd.DataFrame, figure_dir: Path) -> list[Path]:
    import matplotlib.pyplot as plt

    try:
        import seaborn as sns
    except Exception:
        sns = None

    outputs: list[Path] = []
    plot_df = df[df["label"].isin(CLASS_NAMES)].copy()
    for feature in BOX_FEATURES:
        if feature not in plot_df.columns:
            continue

        plt.figure(figsize=(8, 5), dpi=150)
        if sns is not None:
            sns.boxplot(data=plot_df, x="label", y=feature, order=CLASS_NAMES, color="#7aa6c2")
            sns.stripplot(
                data=plot_df,
                x="label",
                y=feature,
                order=CLASS_NAMES,
                color="#2f2f2f",
                size=2,
                alpha=0.35,
            )
        else:
            plot_df.boxplot(column=feature, by="label", grid=False)
            plt.suptitle("")
        plt.title(f"{feature} by emotion class")
        plt.xlabel("Emotion class")
        plt.ylabel(feature)
        plt.tight_layout()
        output = figure_dir / f"box_{feature}.png"
        plt.savefig(output)
        plt.close()
        outputs.append(output)
    return outputs


def save_heatmap(df: pd.DataFrame, figure_dir: Path) -> Path:
    import matplotlib.pyplot as plt

    try:
        import seaborn as sns
    except Exception:
        sns = None

    available = [feature for feature in KRUSKAL_FEATURES if feature in df.columns]
    class_means = df.groupby("label")[available].mean(numeric_only=True).reindex(CLASS_NAMES)
    zscore = class_means.copy()
    for feature in available:
        mean = zscore[feature].mean(skipna=True)
        std = zscore[feature].std(skipna=True, ddof=0)
        if not np.isfinite(std) or std == 0:
            zscore[feature] = 0.0
        else:
            zscore[feature] = (zscore[feature] - mean) / std

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
        plt.xticks(range(len(available)), available, rotation=45, ha="right")
        plt.yticks(range(len(CLASS_NAMES)), CLASS_NAMES)
    plt.title("Class-level acoustic feature mean heatmap")
    plt.xlabel("Acoustic feature")
    plt.ylabel("Emotion class")
    plt.tight_layout()
    output = figure_dir / "class_feature_mean_heatmap.png"
    plt.savefig(output)
    plt.close()
    return output


def describe_class_patterns(df: pd.DataFrame) -> list[str]:
    available = [feature for feature in KRUSKAL_FEATURES if feature in df.columns]
    if not available:
        return ["No key features were available for class pattern description."]

    means = df.groupby("label")[available].mean(numeric_only=True).reindex(CLASS_NAMES)
    zscore = means.copy()
    for feature in available:
        feature_mean = zscore[feature].mean(skipna=True)
        feature_std = zscore[feature].std(skipna=True, ddof=0)
        if not np.isfinite(feature_std) or feature_std == 0:
            zscore[feature] = 0.0
        else:
            zscore[feature] = (zscore[feature] - feature_mean) / feature_std

    lines: list[str] = []
    for label in CLASS_NAMES:
        if label not in zscore.index:
            continue
        row = zscore.loc[label].dropna()
        high = row[row >= 0.5].sort_values(ascending=False).index.tolist()
        low = row[row <= -0.5].sort_values().index.tolist()
        if not high:
            high = row.sort_values(ascending=False).head(3).index.tolist()
        if not low:
            low = row.sort_values().head(3).index.tolist()
        lines.append(f"- {label}: higher mean features: {', '.join(high)}; lower mean features: {', '.join(low)}.")
    return lines


def save_text_report(df: pd.DataFrame, kruskal_df: pd.DataFrame, report_dir: Path) -> Path:
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
    boxplot_paths = save_boxplots(df, figure_dir)
    heatmap_path = save_heatmap(df, figure_dir)
    report_path = save_text_report(df, kruskal_df, report_dir)

    print("Feature summary:", summary_path)
    print("Kruskal-Wallis results:", kruskal_path)
    print("Boxplots:", len(boxplot_paths), "files saved to", figure_dir)
    print("Heatmap:", heatmap_path)
    print("Text report:", report_path)


if __name__ == "__main__":
    main()
