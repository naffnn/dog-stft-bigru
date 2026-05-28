from __future__ import annotations

import argparse
import itertools
import math
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


DEFAULT_OUT_DIR = Path(r"E:\dog_stft_bigru_github\dog_acoustic_analysis")
DEFAULT_FEATURE_CSV = DEFAULT_OUT_DIR / "features" / "acoustic_features_4s.csv"

CLASS_NAMES = ["angry", "anxious", "happy", "lonely", "sad"]
KEY_FEATURES = [
    "rms_mean",
    "log_energy",
    "spectral_centroid_mean",
    "spectral_bandwidth_mean",
    "spectral_rolloff_mean",
    "onset_rate_per_sec",
    "onset_count",
    "energy_peak_count",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run Dunn post-hoc tests and Kruskal effect-size analysis for dog acoustic features."
    )
    parser.add_argument("--feature_csv", type=Path, default=DEFAULT_FEATURE_CSV)
    parser.add_argument("--out_dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--alpha", type=float, default=0.05)
    return parser.parse_args()


def ensure_label_column(df: pd.DataFrame) -> None:
    if "label" not in df.columns:
        raise ValueError("Input feature csv must contain a 'label' column.")


def prepare_feature_data(df: pd.DataFrame, feature: str) -> tuple[pd.DataFrame | None, str]:
    if feature not in df.columns:
        return None, "feature not found"

    feature_df = df.loc[df["label"].isin(CLASS_NAMES), ["label", feature]].copy()
    feature_df[feature] = pd.to_numeric(feature_df[feature], errors="coerce")
    feature_df = feature_df.dropna(subset=[feature])
    if feature_df.empty:
        return None, "feature has no valid numeric values"

    valid_labels = [label for label in CLASS_NAMES if (feature_df["label"] == label).any()]
    if len(valid_labels) < 2:
        return None, "not enough valid groups"

    feature_df["label"] = pd.Categorical(feature_df["label"], categories=CLASS_NAMES, ordered=True)
    return feature_df.sort_values("label"), ""


def interpret_epsilon_squared(value: float) -> str:
    if not np.isfinite(value):
        return "not_available"
    if value >= 0.14:
        return "large"
    if value >= 0.06:
        return "medium"
    if value >= 0.01:
        return "small"
    return "negligible"


def run_kruskal_effect_sizes(
    df: pd.DataFrame,
    alpha: float,
) -> tuple[pd.DataFrame, dict[str, pd.DataFrame], list[str]]:
    from scipy.stats import kruskal

    rows: list[dict[str, Any]] = []
    valid_feature_data: dict[str, pd.DataFrame] = {}
    failures: list[str] = []

    for feature in KEY_FEATURES:
        feature_df, reason = prepare_feature_data(df, feature)
        if feature_df is None:
            failures.append(f"{feature}\t{reason}")
            rows.append(
                {
                    "feature": feature,
                    "kruskal_statistic": math.nan,
                    "p_value": math.nan,
                    "significant_p_lt_0_05": False,
                    "n_total": 0,
                    "n_groups": 0,
                    "epsilon_squared": math.nan,
                    "eta_squared_approx": math.nan,
                    "effect_size_rank": math.nan,
                    "interpretation": "not_available",
                    "note": reason,
                }
            )
            continue

        groups = [
            feature_df.loc[feature_df["label"] == label, feature].dropna().to_numpy(dtype=float)
            for label in CLASS_NAMES
        ]
        valid_groups = [group for group in groups if len(group) > 0]
        n_total = int(sum(len(group) for group in valid_groups))
        n_groups = int(len(valid_groups))

        try:
            statistic, p_value = kruskal(*valid_groups, nan_policy="omit")
        except Exception as exc:
            failures.append(f"{feature}\tKruskal failed: {exc}")
            statistic, p_value = math.nan, math.nan

        note = ""
        if n_total <= n_groups:
            epsilon_squared = math.nan
            eta_squared_approx = math.nan
            note = "n_total <= n_groups; effect size not computed"
            failures.append(f"{feature}\t{note}")
        elif np.isfinite(statistic):
            epsilon_squared = float((statistic - n_groups + 1) / (n_total - n_groups))
            epsilon_squared = max(0.0, epsilon_squared)
            eta_squared_approx = float(statistic / (n_total - 1)) if n_total > 1 else math.nan
        else:
            epsilon_squared = math.nan
            eta_squared_approx = math.nan

        rows.append(
            {
                "feature": feature,
                "kruskal_statistic": statistic,
                "p_value": p_value,
                "significant_p_lt_0_05": bool(np.isfinite(p_value) and p_value < alpha),
                "n_total": n_total,
                "n_groups": n_groups,
                "epsilon_squared": epsilon_squared,
                "eta_squared_approx": eta_squared_approx,
                "effect_size_rank": math.nan,
                "interpretation": interpret_epsilon_squared(epsilon_squared),
                "note": note,
            }
        )
        valid_feature_data[feature] = feature_df

    result = pd.DataFrame(rows)
    valid_rank = result["epsilon_squared"].replace([np.inf, -np.inf], np.nan)
    ranked_indices = valid_rank.dropna().sort_values(ascending=False).index
    for rank, index in enumerate(ranked_indices, start=1):
        result.loc[index, "effect_size_rank"] = rank

    return result.sort_values(
        by=["effect_size_rank", "epsilon_squared"],
        ascending=[True, False],
        na_position="last",
    ), valid_feature_data, failures


def save_median_ranking(valid_feature_data: dict[str, pd.DataFrame], report_dir: Path) -> Path:
    rows: list[dict[str, Any]] = []
    for feature, feature_df in valid_feature_data.items():
        stats = (
            feature_df.groupby("label", observed=False)[feature]
            .agg(["median", "mean", "std", "count"])
            .reindex(CLASS_NAMES)
            .dropna(subset=["median"])
            .sort_values("median", ascending=False)
        )
        for rank, (label, row) in enumerate(stats.iterrows(), start=1):
            rows.append(
                {
                    "feature": feature,
                    "rank": rank,
                    "label": label,
                    "median": row["median"],
                    "mean": row["mean"],
                    "std": row["std"],
                    "n": int(row["count"]),
                }
            )

    output = report_dir / "feature_class_median_ranking.csv"
    pd.DataFrame(rows).to_csv(output, index=False, encoding="utf-8-sig")
    return output


def load_scikit_posthocs() -> tuple[Any | None, str]:
    try:
        import scikit_posthocs as sp

        return sp, ""
    except Exception:
        message = "Missing dependency: scikit-posthocs\nPlease install with: pip install scikit-posthocs"
        print(message)
        return None, message


def save_dunn_heatmap(dunn_matrix: pd.DataFrame, feature: str, figure_dir: Path) -> Path | None:
    import matplotlib.pyplot as plt

    try:
        import seaborn as sns
    except Exception:
        sns = None

    numeric_matrix = dunn_matrix.apply(pd.to_numeric, errors="coerce").reindex(index=CLASS_NAMES, columns=CLASS_NAMES)
    if numeric_matrix.dropna(how="all").empty:
        return None

    safe_p = numeric_matrix.clip(lower=1e-300)
    log_p = -np.log10(safe_p)

    plt.figure(figsize=(7, 6), dpi=150)
    if sns is not None:
        sns.heatmap(
            log_p,
            annot=numeric_matrix,
            fmt=".3g",
            cmap="mako",
            linewidths=0.5,
            cbar_kws={"label": "-log10(FDR-adjusted p-value)"},
        )
    else:
        image = plt.imshow(log_p.to_numpy(dtype=float), aspect="auto", cmap="viridis")
        plt.colorbar(image, label="-log10(FDR-adjusted p-value)")
        plt.xticks(range(len(CLASS_NAMES)), CLASS_NAMES, rotation=45, ha="right")
        plt.yticks(range(len(CLASS_NAMES)), CLASS_NAMES)

    plt.title(f"Dunn post-hoc pairwise comparison: {feature}")
    plt.xlabel("Emotion class")
    plt.ylabel("Emotion class")
    plt.tight_layout()
    output = figure_dir / f"dunn_{feature}_heatmap.png"
    plt.savefig(output)
    plt.close()
    return output


def save_dunn_outputs(
    valid_feature_data: dict[str, pd.DataFrame],
    report_dir: Path,
    figure_dir: Path,
    alpha: float,
) -> tuple[Path, list[Path], list[str], str]:
    sp, missing_message = load_scikit_posthocs()
    long_rows: list[dict[str, Any]] = []
    heatmap_paths: list[Path] = []
    failures: list[str] = []
    long_output = report_dir / "dunn_posthoc_all_features_long.csv"
    columns = [
        "feature",
        "group1",
        "group2",
        "p_adjusted",
        "significant_p_lt_0_05",
        "group1_median",
        "group2_median",
        "median_diff_group1_minus_group2",
        "direction",
    ]

    if sp is None:
        pd.DataFrame(columns=columns).to_csv(long_output, index=False, encoding="utf-8-sig")
        failures.append(f"Dunn post-hoc skipped\t{missing_message.replace(chr(10), ' | ')}")
        return long_output, heatmap_paths, failures, missing_message

    for feature, feature_df in valid_feature_data.items():
        try:
            dunn_matrix = sp.posthoc_dunn(
                feature_df,
                val_col=feature,
                group_col="label",
                p_adjust="fdr_bh",
            )
            dunn_matrix = dunn_matrix.reindex(index=CLASS_NAMES, columns=CLASS_NAMES)
            matrix_output = report_dir / f"dunn_{feature}_pvalues.csv"
            dunn_matrix.to_csv(matrix_output, encoding="utf-8-sig")

            medians = feature_df.groupby("label", observed=False)[feature].median().reindex(CLASS_NAMES)
            for group1, group2 in itertools.combinations(CLASS_NAMES, 2):
                p_adjusted = dunn_matrix.loc[group1, group2]
                group1_median = medians.loc[group1]
                group2_median = medians.loc[group2]
                median_diff = group1_median - group2_median
                direction = "group1_higher" if group1_median > group2_median else "group2_higher"
                long_rows.append(
                    {
                        "feature": feature,
                        "group1": group1,
                        "group2": group2,
                        "p_adjusted": p_adjusted,
                        "significant_p_lt_0_05": bool(pd.notna(p_adjusted) and p_adjusted < alpha),
                        "group1_median": group1_median,
                        "group2_median": group2_median,
                        "median_diff_group1_minus_group2": median_diff,
                        "direction": direction,
                    }
                )

            heatmap_path = save_dunn_heatmap(dunn_matrix, feature, figure_dir)
            if heatmap_path is not None:
                heatmap_paths.append(heatmap_path)
        except Exception as exc:
            failures.append(f"{feature}\tDunn failed: {exc}")
            print(f"[WARN] Dunn failed for {feature}: {exc}")

    pd.DataFrame(long_rows, columns=columns).to_csv(long_output, index=False, encoding="utf-8-sig")
    return long_output, heatmap_paths, failures, missing_message


def save_effect_size_plot(effect_df: pd.DataFrame, figure_dir: Path) -> Path | None:
    import matplotlib.pyplot as plt

    plot_df = effect_df.dropna(subset=["epsilon_squared"]).sort_values("epsilon_squared", ascending=False)
    if plot_df.empty:
        return None

    plt.figure(figsize=(10, 5), dpi=150)
    plt.bar(plot_df["feature"], plot_df["epsilon_squared"], color="#5b8fb9")
    plt.title("Kruskal-Wallis effect size ranking")
    plt.xlabel("Feature")
    plt.ylabel("Epsilon squared")
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    output = figure_dir / "feature_effect_size_ranking.png"
    plt.savefig(output)
    plt.close()
    return output


def summarize_top_pairs(dunn_long: pd.DataFrame, top_features: list[str]) -> list[str]:
    if dunn_long.empty:
        return ["Dunn post-hoc pairwise details were not available."]

    lines: list[str] = []
    for feature in top_features:
        feature_pairs = dunn_long.loc[dunn_long["feature"] == feature].copy()
        feature_pairs["p_adjusted"] = pd.to_numeric(feature_pairs["p_adjusted"], errors="coerce")
        feature_pairs = feature_pairs.dropna(subset=["p_adjusted"]).sort_values("p_adjusted")
        if feature_pairs.empty:
            continue
        top_pair = feature_pairs.iloc[0]
        lines.append(
            f"- {feature}: strongest pairwise hint is {top_pair['group1']} vs {top_pair['group2']} "
            f"(FDR-adjusted p={top_pair['p_adjusted']:.4g}, {top_pair['direction']})."
        )
    return lines or ["No valid Dunn pairwise p-values were available for top-ranked features."]


def save_summary(
    effect_df: pd.DataFrame,
    dunn_long_path: Path,
    report_dir: Path,
    dunn_missing_message: str,
    heatmap_paths: list[Path],
) -> Path:
    output = report_dir / "posthoc_effect_analysis_summary.txt"
    ranked = effect_df.dropna(subset=["epsilon_squared"]).sort_values("epsilon_squared", ascending=False)
    top5 = ranked.head(5)
    top_features = top5["feature"].tolist()

    try:
        dunn_long = pd.read_csv(dunn_long_path)
    except Exception:
        dunn_long = pd.DataFrame()

    spectrum_features = {"spectral_centroid_mean", "spectral_bandwidth_mean", "spectral_rolloff_mean"}
    onset_energy_features = {"onset_rate_per_sec", "onset_count", "energy_peak_count"}
    spectrum_top = [feature for feature in top_features if feature in spectrum_features]
    onset_energy_top = [feature for feature in top_features if feature in onset_energy_features]

    lines = [
        "Post-hoc and effect size analysis summary",
        "",
        "Notes:",
        "- Smaller p-values indicate stronger statistical evidence against equal distributions.",
        "- Kruskal-Wallis tests indicate whether the five classes differ overall, but do not identify which two classes differ.",
        "- Dunn post-hoc tests are used to inspect pairwise class differences after the overall nonparametric comparison.",
        "- Epsilon squared is used as an auxiliary effect-size estimate for the strength of class differences.",
        "- Effect-size labels are rough descriptors and should not be treated as absolute biological conclusions.",
        "",
        "Top 5 features by epsilon_squared:",
    ]
    if top5.empty:
        lines.append("No feature had a computable epsilon_squared value.")
    else:
        for _, row in top5.iterrows():
            lines.append(
                f"- {row['feature']}: epsilon_squared={row['epsilon_squared']:.4g}, "
                f"p={row['p_value']:.4g}, interpretation={row['interpretation']}"
            )

    lines.extend(["", "Brief interpretation:"])
    if spectrum_top:
        lines.append(
            "Spectral features appear among the higher-ranked effects, which may indicate that frequency-distribution differences remain important in these 4-second samples."
        )
    else:
        lines.append(
            "Spectral features do not dominate the top-ranked effects in this run, suggesting their relative contribution may be feature- or dataset-dependent."
        )

    if onset_energy_top:
        lines.append(
            "Onset or energy-peak features appear in the top-ranked set, suggesting temporal event density may contribute to class separation."
        )
    else:
        lines.append(
            "Onset and energy-peak features appear relatively lower in the ranking, suggesting they may be weaker separators than the leading features in this run."
        )

    lines.extend(["", "Pairwise class hints from Dunn post-hoc:"])
    if dunn_missing_message:
        lines.append(dunn_missing_message.replace("\n", " "))
        lines.append("Dunn test was not executed; install scikit-posthocs to generate pairwise comparisons and heatmaps.")
    elif not heatmap_paths:
        lines.append("Dunn test did not produce heatmaps, likely because no valid pairwise matrices were available.")
    else:
        lines.extend(summarize_top_pairs(dunn_long, top_features))

    lines.extend(
        [
            "",
            "Caution:",
            "These results are statistical descriptions of the current dataset. They may suggest acoustic patterns, but they do not prove a direct emotional mechanism.",
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

    print("feature_csv:", args.feature_csv)
    print("out_dir:", args.out_dir)
    print("analysis feature count:", len(KEY_FEATURES))

    df = pd.read_csv(args.feature_csv)
    ensure_label_column(df)

    effect_df, valid_feature_data, failures = run_kruskal_effect_sizes(df, args.alpha)
    effect_output = report_dir / "feature_effect_size_ranking.csv"
    effect_df.to_csv(effect_output, index=False, encoding="utf-8-sig")

    median_output = save_median_ranking(valid_feature_data, report_dir)
    effect_plot = save_effect_size_plot(effect_df, figure_dir)
    dunn_long_output, heatmap_paths, dunn_failures, dunn_missing_message = save_dunn_outputs(
        valid_feature_data,
        report_dir,
        figure_dir,
        args.alpha,
    )
    failures.extend(dunn_failures)

    failed_output = report_dir / "posthoc_effect_failed_features.txt"
    if failures:
        failed_output.write_text("\n".join(failures), encoding="utf-8")
    else:
        failed_output.write_text("No skipped or failed features.\n", encoding="utf-8")

    summary_output = save_summary(
        effect_df=effect_df,
        dunn_long_path=dunn_long_output,
        report_dir=report_dir,
        dunn_missing_message=dunn_missing_message,
        heatmap_paths=heatmap_paths,
    )

    success_count = len(valid_feature_data)
    skipped_count = len(KEY_FEATURES) - success_count
    print("successful feature count:", success_count)
    print("skipped/failed feature count:", skipped_count)
    print("effect size ranking:", effect_output)
    print("Dunn post-hoc long table:", dunn_long_output)
    print("median ranking:", median_output)
    print("effect size figure:", effect_plot if effect_plot is not None else "skipped")
    print("Dunn heatmaps:", len(heatmap_paths), "files saved to", figure_dir)
    print("summary:", summary_output)
    print("failed feature log:", failed_output)


if __name__ == "__main__":
    main()
