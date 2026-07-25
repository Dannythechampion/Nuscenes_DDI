#!/usr/bin/env python3
"""Validate preregistered camera or LiDAR DDI v1 on nuScenes model errors."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
from scipy.stats import spearmanr
from sklearn.metrics import roc_auc_score
from statsmodels.genmod.families import Binomial


ORDER = ["Easy", "Moderate", "Hard"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--modality", choices=("camera", "lidar"), required=True)
    parser.add_argument("--samples-csv", type=Path, required=True)
    parser.add_argument("--manifest-csv", type=Path, required=True)
    parser.add_argument("--matches-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--bootstrap-repetitions", type=int, default=2000)
    return parser.parse_args()


def safe_spearman(x: pd.Series, y: pd.Series) -> tuple[float, float]:
    valid = x.notna() & y.notna()
    if valid.sum() < 3 or x[valid].nunique() < 2 or y[valid].nunique() < 2:
        return np.nan, np.nan
    result = spearmanr(x[valid], y[valid])
    return float(result.statistic), float(result.pvalue)


def safe_auc(target: pd.Series, predictor: pd.Series) -> float:
    valid = target.notna() & predictor.notna()
    if valid.sum() < 2 or target[valid].nunique() < 2 or predictor[valid].nunique() < 2:
        return np.nan
    return float(roc_auc_score(target[valid].astype(int), predictor[valid]))


def scene_bootstrap(
    frame: pd.DataFrame,
    value: str,
    repetitions: int,
    rng: np.random.Generator,
) -> tuple[float, float]:
    scenes = frame["scene_name"].drop_duplicates().to_numpy()
    if len(scenes) < 2:
        return np.nan, np.nan
    grouped = {scene: frame.loc[frame["scene_name"] == scene, value].to_numpy() for scene in scenes}
    estimates = []
    for _ in range(repetitions):
        sampled = rng.choice(scenes, size=len(scenes), replace=True)
        values = np.concatenate([grouped[scene] for scene in sampled])
        estimates.append(float(np.nanmean(values)))
    return tuple(float(value) for value in np.quantile(estimates, [0.025, 0.975]))


def load_runs(root: Path) -> list[dict]:
    runs = []
    for summary_path in sorted(root.glob("*/matching_summary.json")):
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        summary["run_name"] = summary_path.parent.name
        summary["run_dir"] = summary_path.parent
        runs.append(summary)
    if not runs:
        raise RuntimeError(f"No matching_summary.json files found below {root}")
    return runs


def monotonic_rates(part: pd.DataFrame, value: str) -> tuple[list[float], bool]:
    indexed = part.set_index("difficulty").reindex(ORDER)
    rates = indexed[value].tolist()
    valid = all(pd.notna(rate) for rate in rates)
    return rates, bool(valid and rates[0] < rates[1] < rates[2])


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    figures = args.output_dir / "figures"
    figures.mkdir(exist_ok=True)
    rng = np.random.default_rng(20260725)

    score = f"{args.modality}_ddi_prior_v1"
    difficulty = f"{args.modality}_difficulty_prior_v1"
    object_score = f"{args.modality}_object_ddi_prior"
    if args.modality == "camera":
        frame_components = [
            "density_prior_score",
            "camera_occlusion_p75_prior_score",
            "camera_truncation_p75_kitti_score",
        ]
        object_components = [
            "camera_object_ddi_prior",
            "camera_occlusion_prior_score",
            "camera_truncation_kitti_score",
        ]
    else:
        frame_components = ["density_prior_score", "lidar_le5_point_prop"]
        object_components = ["lidar_object_ddi_prior", "num_lidar_pts"]

    samples = pd.read_csv(args.samples_csv)
    manifest = pd.read_csv(args.manifest_csv)[
        ["sample_token", "scene_name", "official_split", "location"]
    ]
    samples = samples.merge(
        manifest,
        on=["sample_token", "scene_name"],
        how="inner",
        validate="one_to_one",
    )
    samples = samples.loc[samples["official_split"] == "val"].copy()
    samples[difficulty] = pd.Categorical(samples[difficulty], categories=ORDER, ordered=True)
    if samples["scene_name"].nunique() != 150 or len(samples) != 6019:
        raise RuntimeError(
            f"Expected 150 scenes and 6,019 samples, got "
            f"{samples['scene_name'].nunique()} scenes and {len(samples)} samples"
        )

    distribution = (
        samples.groupby(difficulty, observed=False)
        .agg(keyframes=("sample_token", "size"), scenes=("scene_name", "nunique"), mean_score=(score, "mean"))
        .reset_index()
        .rename(columns={difficulty: "difficulty"})
    )
    distribution.to_csv(
        args.output_dir / "preregistered_group_distribution.csv", index=False, encoding="utf-8-sig"
    )

    runs = load_runs(args.matches_root)
    group_rows = []
    association_rows = []
    component_rows = []
    regression_rows = []
    density_regression_rows = []
    model_rows = []
    primary_frames: dict[str, pd.DataFrame] = {}

    frame_feature_columns = [
        "sample_token",
        "scene_name",
        "location",
        difficulty,
        score,
        "annotation_count",
        "weighted_complexity",
        "count_score",
        "vru_count",
        "vehicle_count",
        "static_count",
        "moving_vehicle_count_8m",
        "nuplan_near_multiple_vehicles",
        "distance_ego_mean_m",
        *frame_components,
    ]
    for run in runs:
        run_dir = run["run_dir"]
        frames = pd.read_csv(run_dir / "frame_detection_errors.csv").merge(
            samples[frame_feature_columns],
            on=["sample_token", "scene_name"],
            how="inner",
            validate="one_to_one",
        )
        objects = pd.read_csv(run_dir / "object_detection_errors.csv").merge(
            samples[["sample_token", score, difficulty, "annotation_count", "location"]],
            on="sample_token",
            how="inner",
            validate="many_to_one",
            suffixes=("", "_frame"),
        )
        frames["fp_per_gt_plus_one"] = frames["false_positive_count"] / (frames["gt_count"] + 1)
        run_name = run["run_name"]
        if run["distance_threshold_m"] == 2.0 and run["score_threshold"] == 0.05:
            primary_frames[run["model_name"]] = frames

        model_rows.append(
            {
                "run_name": run_name,
                "model_name": run["model_name"],
                "distance_threshold_m": run["distance_threshold_m"],
                "score_threshold": run["score_threshold"],
                "keyframes": len(frames),
                "scenes": frames["scene_name"].nunique(),
                "eligible_gt_objects": len(objects),
                "false_negative_rate": objects["false_negative"].mean(),
                "false_positives": frames["false_positive_count"].sum(),
            }
        )

        for level in ORDER:
            part = frames.loc[frames[difficulty].astype(str) == level]
            ci_low, ci_high = scene_bootstrap(
                part, "false_negative_rate", args.bootstrap_repetitions, rng
            )
            group_rows.append(
                {
                    "run_name": run_name,
                    "model_name": run["model_name"],
                    "distance_threshold_m": run["distance_threshold_m"],
                    "score_threshold": run["score_threshold"],
                    "difficulty": level,
                    "keyframes": len(part),
                    "scenes": part["scene_name"].nunique(),
                    "mean_frame_false_negative_rate": part["false_negative_rate"].mean(),
                    "fn_rate_ci_low": ci_low,
                    "fn_rate_ci_high": ci_high,
                    "mean_fp_per_gt_plus_one": part["fp_per_gt_plus_one"].mean(),
                    "mean_translation_error_m": part["mean_translation_error_m"].mean(),
                }
            )

        for outcome in (
            "false_negative_rate",
            "fp_per_gt_plus_one",
            "mean_translation_error_m",
        ):
            rho, p_value = safe_spearman(frames[score], frames[outcome])
            association_rows.append(
                {
                    "run_name": run_name,
                    "model_name": run["model_name"],
                    "outcome": outcome,
                    "spearman_rho": rho,
                    "p_value": p_value,
                    "n_keyframes": int((frames[score].notna() & frames[outcome].notna()).sum()),
                }
            )

        for component in [
            score,
            *frame_components,
            "annotation_count",
            "weighted_complexity",
            "count_score",
            "vru_count",
            "vehicle_count",
            "static_count",
            "moving_vehicle_count_8m",
            "nuplan_near_multiple_vehicles",
        ]:
            predictor = frames[component]
            if component == "lidar_le5_point_prop":
                predictor = predictor * 10.0
            rho, p_value = safe_spearman(predictor, frames["false_negative_rate"])
            component_rows.append(
                {
                    "run_name": run_name,
                    "model_name": run["model_name"],
                    "level": "frame",
                    "component": component,
                    "spearman_rho": rho,
                    "p_value": p_value,
                    "object_fn_auc": np.nan,
                }
            )
        for component in object_components:
            predictor = objects[component]
            if component == "num_lidar_pts":
                predictor = -np.log1p(predictor)
            component_rows.append(
                {
                    "run_name": run_name,
                    "model_name": run["model_name"],
                    "level": "object",
                    "component": component,
                    "spearman_rho": np.nan,
                    "p_value": np.nan,
                    "object_fn_auc": safe_auc(objects["false_negative"], predictor),
                }
            )
        for component, predictor in {
            "raw_object_count": objects["annotation_count"],
            "object_distance": objects["distance_ego_m"],
            "inverse_lidar_points": -np.log1p(objects["num_lidar_pts"]),
        }.items():
            component_rows.append(
                {
                    "run_name": run_name,
                    "model_name": run["model_name"],
                    "level": "baseline",
                    "component": component,
                    "spearman_rho": np.nan,
                    "p_value": np.nan,
                    "object_fn_auc": safe_auc(objects["false_negative"], predictor),
                }
            )

        if run["distance_threshold_m"] == 2.0 and run["score_threshold"] == 0.05:
            regression = objects.copy()
            regression["frame_ddi"] = regression[score]
            regression["object_ddi"] = regression[object_score]
            regression["log_distance"] = np.log1p(regression["distance_ego_m"])
            regression["log_count"] = np.log1p(regression["annotation_count"])
            regression["log_box_volume"] = np.log1p(regression["box_volume_m3"])
            fitted = smf.glm(
                "false_negative ~ frame_ddi + object_ddi + log_distance + log_count + "
                "log_box_volume + C(detection_name) + C(location)",
                data=regression,
                family=Binomial(),
            ).fit(cov_type="cluster", cov_kwds={"groups": regression["scene_name"]})
            for term in ("frame_ddi", "object_ddi"):
                interval = fitted.conf_int().loc[term]
                regression_rows.append(
                    {
                        "model_name": run["model_name"],
                        "term": term,
                        "odds_ratio_per_point": np.exp(fitted.params[term]),
                        "ci_low": np.exp(interval.iloc[0]),
                        "ci_high": np.exp(interval.iloc[1]),
                        "p_value": fitted.pvalues[term],
                        "n_objects": int(fitted.nobs),
                        "n_scenes": regression["scene_name"].nunique(),
                    }
                )

            density_frame = frames.dropna(
                subset=[
                    "false_negative_rate",
                    "vru_count",
                    "vehicle_count",
                    "static_count",
                    "distance_ego_mean_m",
                    "location",
                    "scene_name",
                ]
            ).copy()
            density_fit = smf.wls(
                "false_negative_rate ~ vru_count + vehicle_count + static_count + "
                "distance_ego_mean_m + C(location)",
                data=density_frame,
                weights=density_frame["gt_count"].clip(lower=1),
            ).fit(cov_type="cluster", cov_kwds={"groups": density_frame["scene_name"]})
            for term in ("vru_count", "vehicle_count", "static_count"):
                interval = density_fit.conf_int().loc[term]
                density_regression_rows.append(
                    {
                        "model_name": run["model_name"],
                        "term": term,
                        "fn_rate_coefficient_per_object": density_fit.params[term],
                        "ci_low": interval.iloc[0],
                        "ci_high": interval.iloc[1],
                        "p_value": density_fit.pvalues[term],
                        "n_keyframes": int(density_fit.nobs),
                        "n_scenes": density_frame["scene_name"].nunique(),
                    }
                )

    groups = pd.DataFrame(group_rows)
    associations = pd.DataFrame(association_rows)
    components = pd.DataFrame(component_rows)
    regressions = pd.DataFrame(regression_rows)
    density_regressions = pd.DataFrame(density_regression_rows)
    models = pd.DataFrame(model_rows)
    groups.to_csv(args.output_dir / "difficulty_group_results.csv", index=False, encoding="utf-8-sig")
    associations.to_csv(args.output_dir / "continuous_associations.csv", index=False, encoding="utf-8-sig")
    components.to_csv(args.output_dir / "component_and_baseline_validity.csv", index=False, encoding="utf-8-sig")
    regressions.to_csv(args.output_dir / "adjusted_logistic_regression.csv", index=False, encoding="utf-8-sig")
    density_regressions.to_csv(
        args.output_dir / "exploratory_density_regression.csv", index=False, encoding="utf-8-sig"
    )
    models.to_csv(args.output_dir / "model_run_summary.csv", index=False, encoding="utf-8-sig")

    primary = groups.loc[
        (groups["distance_threshold_m"] == 2.0) & (groups["score_threshold"] == 0.05)
    ]
    conclusions = []
    for model_name, part in primary.groupby("model_name"):
        rates, monotonic = monotonic_rates(part, "mean_frame_false_negative_rate")
        conclusions.append(
            {
                "model_name": model_name,
                "easy_fn_rate": rates[0],
                "moderate_fn_rate": rates[1],
                "hard_fn_rate": rates[2],
                "strict_monotonic": monotonic,
            }
        )
    conclusion_df = pd.DataFrame(conclusions)
    conclusion_df.to_csv(args.output_dir / "primary_conclusion.csv", index=False, encoding="utf-8-sig")

    sensitivity = []
    for run_name, part in groups.groupby("run_name"):
        rates, monotonic = monotonic_rates(part, "mean_frame_false_negative_rate")
        sensitivity.append({"run_name": run_name, "strict_monotonic": monotonic, "rates": rates})
    sensitivity_df = pd.DataFrame(
        [{"run_name": row["run_name"], "strict_monotonic": row["strict_monotonic"]} for row in sensitivity]
    )
    sensitivity_df.to_csv(args.output_dir / "sensitivity_conclusion.csv", index=False, encoding="utf-8-sig")

    primary_components = components.loc[
        components["run_name"].str.endswith("_d2_s005")
        & (components["level"] == "frame")
    ]
    incremental_rows = []
    for model_name, part in primary_components.groupby("model_name"):
        composite = part.loc[part["component"] == score, "spearman_rho"]
        single = part.loc[part["component"].isin(frame_components), ["component", "spearman_rho"]]
        strongest = single.sort_values("spearman_rho", ascending=False).iloc[0]
        composite_rho = float(composite.iloc[0]) if len(composite) else np.nan
        incremental_rows.append(
            {
                "model_name": model_name,
                "composite_spearman_rho": composite_rho,
                "strongest_component": strongest["component"],
                "strongest_component_spearman_rho": strongest["spearman_rho"],
                "composite_stronger": bool(composite_rho > strongest["spearman_rho"]),
            }
        )
    incremental = pd.DataFrame(incremental_rows)
    incremental.to_csv(
        args.output_dir / "composite_incremental_validity.csv", index=False, encoding="utf-8-sig"
    )

    fig, ax = plt.subplots(figsize=(8, 4.8))
    for model_name, part in primary.groupby("model_name"):
        indexed = part.set_index("difficulty").reindex(ORDER)
        ax.plot(ORDER, indexed["mean_frame_false_negative_rate"], marker="o", label=model_name)
    ax.set_xlabel("Preregistered difficulty")
    ax.set_ylabel("Mean frame false-negative rate")
    ax.set_ylim(0, 1)
    ax.grid(axis="y", alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(figures / f"{args.modality}_difficulty_vs_fn_rate.png", dpi=180)
    plt.close(fig)

    populated = bool((distribution["keyframes"] > 0).all())
    monotonic_all = bool(len(conclusion_df)) and bool(conclusion_df["strict_monotonic"].all())
    frame_effects = regressions.loc[regressions["term"] == "frame_ddi"]
    positive_adjusted = bool(len(frame_effects)) and bool(
        (
            (frame_effects["odds_ratio_per_point"] > 1)
            & (frame_effects["ci_low"] > 1)
            & (frame_effects["p_value"] < 0.05)
        ).all()
    )
    object_effects = regressions.loc[regressions["term"] == "object_ddi"]
    positive_object_adjusted = bool(len(object_effects)) and bool(
        (
            (object_effects["odds_ratio_per_point"] > 1)
            & (object_effects["ci_low"] > 1)
            & (object_effects["p_value"] < 0.05)
        ).all()
    )
    composite_adds = bool(len(incremental)) and bool(incremental["composite_stronger"].all())
    sensitivity_stable = bool(len(sensitivity_df)) and bool(
        sensitivity_df["strict_monotonic"].mean() >= 0.8
    )
    supported = (
        populated
        and monotonic_all
        and positive_adjusted
        and positive_object_adjusted
        and sensitivity_stable
        and composite_adds
    )

    lines = [
        f"# nuScenes 150-scene {args.modality.upper()} DDI v1 validation",
        "",
        "## Cohort",
        "",
        f"- 150 official validation scenes, {len(samples):,} keyframes",
        f"- Models: {', '.join(sorted(models['model_name'].unique()))}",
        f"- Matching runs: {len(runs)}",
        "",
        "## Preregistered group distribution",
        "",
    ]
    for row in distribution.itertuples(index=False):
        lines.append(f"- {row.difficulty}: {row.keyframes:,} keyframes across {row.scenes} scenes")
    lines.extend(["", "## Primary model results", ""])
    for row in conclusion_df.itertuples(index=False):
        lines.append(
            f"- {row.model_name}: Easy {row.easy_fn_rate:.4f}, Moderate {row.moderate_fn_rate:.4f}, "
            f"Hard {row.hard_fn_rate:.4f}; strict monotonic={row.strict_monotonic}"
        )
    lines.extend(["", "## Adjusted effects", ""])
    for row in regressions.itertuples(index=False):
        lines.append(
            f"- {row.model_name} {row.term}: OR/point={row.odds_ratio_per_point:.4f} "
            f"(95% CI {row.ci_low:.4f}-{row.ci_high:.4f}, p={row.p_value:.4g})"
        )
    lines.extend(
        [
            "",
            "## Frozen decision",
            "",
            f"- All three groups populated: {populated}",
            f"- Strict Easy < Moderate < Hard in every primary model: {monotonic_all}",
            f"- Positive significant adjusted frame-DDI effect in every model: {positive_adjusted}",
            f"- Positive significant adjusted object-DDI effect in every model: {positive_object_adjusted}",
            f"- Monotonic in at least 80% of sensitivity runs: {sensitivity_stable}",
            f"- Composite stronger than every single component: {composite_adds}",
            f"- Overall preregistered support: **{supported}**",
            "",
            "A negative result is retained as evidence against the transferred rule; validation quantiles are not fitted.",
        ]
    )
    (args.output_dir / f"{args.modality}_ddi_v1_validation_report.md").write_text(
        "\n".join(lines) + "\n", encoding="utf-8"
    )
    print(json.dumps({"modality": args.modality, "runs": len(runs), "supported": supported}, indent=2))


if __name__ == "__main__":
    main()
