"""Audit whether the proposed perception DDI thresholds separate nuScenes samples.

The audit is intentionally descriptive. It does not declare empirical quantiles to
be ground truth; they are comparison points for testing the presentation's fixed
thresholds on the full trainval split.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


COMPONENT_COLUMNS = (
    "count_score",
    "visibility_mean_score",
    "visibility_max_score",
    "lidar_sparsity_mean_score",
    "lidar_sparsity_max_score",
    "truncation_mean_score",
    "truncation_max_score",
)

RAW_COLUMNS = (
    "annotation_count",
    "vru_count",
    "vehicle_count",
    "weighted_complexity",
    "visibility_mean_score",
    "lidar_sparsity_mean_score",
    "truncation_mean_ratio",
    "perception_difficulty_score",
)


def fmt(value: float, digits: int = 2) -> str:
    if pd.isna(value):
        return "NA"
    return f"{value:.{digits}f}"


def dataframe_to_markdown(df: pd.DataFrame, digits: int = 3) -> str:
    columns = [str(column) for column in df.columns]
    lines = [
        "| Feature | " + " | ".join(columns) + " |",
        "|---|" + "---:|" * len(columns),
    ]
    for index, row in df.iterrows():
        values = [fmt(float(row[column]), digits) for column in df.columns]
        lines.append(f"| {index} | " + " | ".join(values) + " |")
    return "\n".join(lines)


def component_audit(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for column in COMPONENT_COLUMNS:
        values = df[column].dropna()
        rows.append(
            {
                "component": column,
                "n": len(values),
                "missing_pct": 100.0 * df[column].isna().mean(),
                "mean": values.mean(),
                "std": values.std(),
                "p10": values.quantile(0.10),
                "p25": values.quantile(0.25),
                "p50": values.quantile(0.50),
                "p75": values.quantile(0.75),
                "p90": values.quantile(0.90),
                "min_bin_pct": 100.0 * (values == values.min()).mean(),
                "max_bin_pct": 100.0 * (values == values.max()).mean(),
            }
        )
    return pd.DataFrame(rows)


def score_distribution(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for column in COMPONENT_COLUMNS:
        counts = df[column].value_counts(dropna=False).sort_index()
        for value, count in counts.items():
            rows.append(
                {
                    "component": column,
                    "score": value,
                    "count": int(count),
                    "share_pct": 100.0 * count / len(df),
                }
            )
    return pd.DataFrame(rows)


def difficulty_groups(df: pd.DataFrame) -> pd.DataFrame:
    scores = df["perception_difficulty_score"]
    equal_width = pd.cut(scores, bins=[0, 33.333, 66.667, 100], labels=["Low", "Medium", "High"], include_lowest=True)
    empirical = pd.qcut(scores.rank(method="first"), q=3, labels=["Low", "Medium", "High"])
    rows = []
    for method, groups in (("fixed_0_100", equal_width), ("empirical_tertile", empirical)):
        counts = groups.value_counts().reindex(["Low", "Medium", "High"], fill_value=0)
        for group, count in counts.items():
            rows.append({"method": method, "group": group, "count": int(count), "share_pct": 100.0 * count / len(df)})
    return pd.DataFrame(rows)


def object_threshold_audit(objects: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict] = []

    def add_counts(family: str, labels: pd.Series) -> None:
        counts = labels.value_counts(dropna=False)
        for label, count in counts.items():
            rows.append(
                {
                    "family": family,
                    "bin": str(label),
                    "count": int(count),
                    "share_pct": 100.0 * count / len(objects),
                }
            )

    lidar_bins = pd.cut(
        objects["num_lidar_pts"],
        bins=[-np.inf, 4, 19, 49, np.inf],
        labels=["0-4", "5-19", "20-49", "50+"],
    )
    add_counts("lidar_points_team", lidar_bins)
    add_counts(
        "official_eval_point_filter",
        objects["official_eval_has_point"].map({True: "kept", False: "filtered_zero_lidar_plus_radar"}),
    )
    add_counts("visibility_token_official", objects["visibility_token"].astype("Int64").astype(str))

    valid_truncation = objects["truncation_ratio"].dropna()
    team_truncation = pd.cut(
        valid_truncation,
        bins=[-np.inf, 0.05, 0.30, 0.50, np.inf],
        labels=["<=0.05", "0.05-0.30", "0.30-0.50", ">0.50"],
    )
    kitti_truncation = pd.cut(
        valid_truncation,
        bins=[-np.inf, 0.15, 0.30, 0.50, np.inf],
        labels=["<=0.15", "0.15-0.30", "0.30-0.50", ">0.50"],
    )
    add_counts("truncation_team", team_truncation.reindex(objects.index))
    add_counts("truncation_kitti_candidate", kitti_truncation.reindex(objects.index))
    return pd.DataFrame(rows)


def class_distance_lidar_audit(objects: pd.DataFrame) -> pd.DataFrame:
    frame = objects.copy()
    frame["distance_band_m"] = pd.cut(
        frame["distance_ego_m"],
        bins=[0, 20, 40, 60, np.inf],
        labels=["0-20", "20-40", "40-60", "60+"],
        include_lowest=True,
    )
    grouped = frame.groupby(["category_group", "distance_band_m"], observed=True)
    return grouped["num_lidar_pts"].agg(
        object_count="size",
        lidar_mean="mean",
        lidar_median="median",
        lidar_p10=lambda values: values.quantile(0.10),
        zero_point_share=lambda values: (values == 0).mean(),
    ).reset_index()


def write_report(
    df: pd.DataFrame,
    audit: pd.DataFrame,
    groups: pd.DataFrame,
    correlations: pd.DataFrame,
    output_path: Path,
) -> None:
    score = df["perception_difficulty_score"]
    fixed = groups[groups["method"] == "fixed_0_100"]
    high_share = fixed.loc[fixed["group"] == "High", "share_pct"].iloc[0]
    low_share = fixed.loc[fixed["group"] == "Low", "share_pct"].iloc[0]

    lines = [
        "# nuScenes Perception DDI Threshold Audit",
        "",
        "## Scope",
        "",
        f"- Samples: {len(df):,}",
        f"- Scenes: {df['scene_name'].nunique():,}",
        "- Unit of analysis: annotated 2 Hz keyframe",
        "- Components: object complexity, visibility, LiDAR sparsity, camera truncation",
        "",
        "## Composite score",
        "",
        f"- Mean: {fmt(score.mean())}",
        f"- Standard deviation: {fmt(score.std())}",
        f"- Range: {fmt(score.min())} to {fmt(score.max())}",
        f"- 25/50/75 percentiles: {fmt(score.quantile(.25))} / {fmt(score.quantile(.5))} / {fmt(score.quantile(.75))}",
        f"- Fixed 0-33/33-67/67-100 split: Low {fmt(low_share, 1)}%, High {fmt(high_share, 1)}%",
        "",
        "## Component saturation",
        "",
        "| Component | Mean | P25 | P50 | P75 | Minimum-bin share | Maximum-bin share |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for row in audit.itertuples(index=False):
        lines.append(
            f"| {row.component} | {fmt(row.mean)} | {fmt(row.p25)} | {fmt(row.p50)} | {fmt(row.p75)} | "
            f"{fmt(row.min_bin_pct, 1)}% | {fmt(row.max_bin_pct, 1)}% |"
        )

    lines.extend(
        [
            "",
            "## Interpretation rules",
            "",
            "- A component with a large maximum-bin share is saturated and has weak ranking power.",
            "- Empty or highly imbalanced Low/Medium/High groups indicate that fixed cutoffs do not match the observed distribution.",
            "- Empirical tertiles are diagnostics, not final thresholds. Final thresholds require model-error validation on trainval.",
            "- High correlation between components indicates duplicated evidence and should trigger a weighting review.",
            "",
            "## Spearman correlations",
            "",
        ]
    )
    lines.append(dataframe_to_markdown(correlations))
    lines.extend(
        [
            "",
            "## Next validation step",
            "",
            "Run multiple perception models on the same keyframes, calculate per-object false-negative and localization errors, "
            "and test monotonic error growth across DDI groups while controlling for raw object count and distance.",
            "",
        ]
    )
    output_path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--samples-csv", required=True)
    parser.add_argument("--objects-csv")
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(args.samples_csv)

    audit = component_audit(df)
    distribution = score_distribution(df)
    groups = difficulty_groups(df)
    correlations = df[list(RAW_COLUMNS)].corr(method="spearman")

    audit.to_csv(output_dir / "component_threshold_audit.csv", index=False, encoding="utf-8-sig")
    distribution.to_csv(output_dir / "component_score_distribution.csv", index=False, encoding="utf-8-sig")
    groups.to_csv(output_dir / "difficulty_group_distribution.csv", index=False, encoding="utf-8-sig")
    correlations.to_csv(output_dir / "perception_feature_spearman.csv", encoding="utf-8-sig")
    write_report(df, audit, groups, correlations, output_dir / "perception_threshold_audit.md")

    if args.objects_csv:
        objects = pd.read_csv(args.objects_csv)
        object_audit = object_threshold_audit(objects)
        lidar_audit = class_distance_lidar_audit(objects)
        object_audit.to_csv(output_dir / "object_threshold_distribution.csv", index=False, encoding="utf-8-sig")
        lidar_audit.to_csv(output_dir / "class_distance_lidar_audit.csv", index=False, encoding="utf-8-sig")
        print("\nObject-level threshold distributions:")
        print(object_audit.to_string(index=False))

    print(audit.to_string(index=False))
    print("\nDifficulty groups:")
    print(groups.to_string(index=False))


if __name__ == "__main__":
    main()
