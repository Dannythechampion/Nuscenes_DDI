#!/usr/bin/env python3
"""Validate perception DDI against multiple nuScenes detection models."""

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


SCORE = "perception_difficulty_score"
COMPONENTS = [
    "count_score",
    "visibility_mean_score",
    "visibility_max_score",
    "lidar_sparsity_mean_score",
    "lidar_sparsity_max_score",
    "truncation_mean_score",
    "truncation_max_score",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--samples-csv", type=Path, required=True)
    parser.add_argument("--manifest-csv", type=Path, required=True)
    parser.add_argument("--matches-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--bootstrap-repetitions", type=int, default=2000)
    return parser.parse_args()


def band(values: pd.Series, low: float, high: float) -> pd.Categorical:
    return pd.cut(
        values,
        bins=[-np.inf, low, high, np.inf],
        labels=["Easy", "Moderate", "Hard"],
        ordered=True,
    )


def scene_bootstrap_mean(
    frame: pd.DataFrame, value: str, repetitions: int, rng: np.random.Generator
) -> tuple[float, float]:
    scenes = frame["scene_name"].drop_duplicates().to_numpy()
    if len(scenes) < 2:
        return np.nan, np.nan
    by_scene = {scene: frame.loc[frame["scene_name"] == scene, value].to_numpy() for scene in scenes}
    estimates = []
    for _ in range(repetitions):
        sampled = rng.choice(scenes, size=len(scenes), replace=True)
        values = np.concatenate([by_scene[scene] for scene in sampled])
        estimates.append(float(np.nanmean(values)))
    return tuple(np.quantile(estimates, [0.025, 0.975]))


def safe_spearman(x: pd.Series, y: pd.Series) -> tuple[float, float]:
    valid = x.notna() & y.notna()
    if valid.sum() < 3 or x[valid].nunique() < 2 or y[valid].nunique() < 2:
        return np.nan, np.nan
    result = spearmanr(x[valid], y[valid])
    return float(result.statistic), float(result.pvalue)


def safe_auc(target: pd.Series, predictor: pd.Series) -> float:
    valid = target.notna() & predictor.notna()
    if target[valid].nunique() < 2:
        return np.nan
    return float(roc_auc_score(target[valid].astype(int), predictor[valid]))


def load_runs(root: Path) -> list[dict]:
    runs = []
    for summary_path in sorted(root.glob("*/matching_summary.json")):
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        run_dir = summary_path.parent
        summary["run_name"] = run_dir.name
        summary["run_dir"] = run_dir
        runs.append(summary)
    if not runs:
        raise RuntimeError(f"No matching runs found below {root}")
    return runs


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    figures_dir = args.output_dir / "figures"
    figures_dir.mkdir(exist_ok=True)
    rng = np.random.default_rng(20260724)

    samples = pd.read_csv(args.samples_csv)
    manifest = pd.read_csv(args.manifest_csv)[
        ["sample_token", "official_split", "location"]
    ]
    samples = samples.merge(manifest, on="sample_token", how="inner", validate="one_to_one")
    train = samples.loc[samples["official_split"] == "train"].copy()
    val = samples.loc[samples["official_split"] == "val"].copy()
    q_low, q_high = train[SCORE].quantile([1 / 3, 2 / 3]).tolist()
    thresholds = pd.DataFrame(
        [
            {"scheme": "train_tertiles", "easy_upper": q_low, "moderate_upper": q_high},
            {"scheme": "fixed_33_67", "easy_upper": 33.3333, "moderate_upper": 66.6667},
        ]
    )
    thresholds.to_csv(args.output_dir / "frozen_thresholds.csv", index=False, encoding="utf-8-sig")

    runs = load_runs(args.matches_root)
    pd.DataFrame(
        [{key: value for key, value in run.items() if key != "run_dir"} for run in runs]
    ).to_csv(args.output_dir / "model_run_summary.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(
        [
            {
                "complete_keyframes": len(samples),
                "complete_scenes": samples["scene_name"].nunique(),
                "train_keyframes": len(train),
                "train_scenes": train["scene_name"].nunique(),
                "validation_keyframes": len(val),
                "validation_scenes": val["scene_name"].nunique(),
                "camera_observations": len(val) * 6,
            }
        ]
    ).to_csv(args.output_dir / "data_scope.csv", index=False, encoding="utf-8-sig")
    group_rows: list[dict] = []
    correlation_rows: list[dict] = []
    component_rows: list[dict] = []
    baseline_rows: list[dict] = []
    regression_rows: list[dict] = []
    frame_outputs: dict[str, pd.DataFrame] = {}

    for run in runs:
        run_dir = run["run_dir"]
        objects = pd.read_csv(run_dir / "object_detection_errors.csv")
        frames = pd.read_csv(run_dir / "frame_detection_errors.csv")
        frames = val.merge(frames, on=["scene_name", "sample_token"], how="left")
        frames["false_positive_count"] = frames["false_positive_count"].fillna(0)
        frames["gt_count"] = frames["gt_count"].fillna(0)
        frames["fp_per_gt_plus_one"] = frames["false_positive_count"] / (frames["gt_count"] + 1)
        objects = objects.merge(
            val[["sample_token", SCORE, "annotation_count", "location"] + COMPONENTS],
            on="sample_token",
            how="inner",
            validate="many_to_one",
        )
        frame_outputs[run["run_name"]] = frames

        for scheme, low, high in thresholds.itertuples(index=False, name=None):
            frames[f"band_{scheme}"] = band(frames[SCORE], low, high)
            objects[f"band_{scheme}"] = band(objects[SCORE], low, high)
            for level in ("Easy", "Moderate", "Hard"):
                frame_part = frames.loc[frames[f"band_{scheme}"] == level]
                object_part = objects.loc[objects[f"band_{scheme}"] == level]
                ci_low, ci_high = scene_bootstrap_mean(
                    frame_part, "false_negative_rate", args.bootstrap_repetitions, rng)
                group_rows.append(
                    {
                        "run_name": run["run_name"],
                        "model_name": run["model_name"],
                        "distance_threshold_m": run["distance_threshold_m"],
                        "score_threshold": run["score_threshold"],
                        "scheme": scheme,
                        "difficulty": level,
                        "keyframes": len(frame_part),
                        "eligible_objects": len(object_part),
                        "pooled_false_negative_rate": object_part["false_negative"].mean(),
                        "mean_frame_false_negative_rate": frame_part["false_negative_rate"].mean(),
                        "frame_fn_rate_ci_low": ci_low,
                        "frame_fn_rate_ci_high": ci_high,
                        "mean_false_positives": frame_part["false_positive_count"].mean(),
                        "mean_translation_error_m": object_part["translation_error_m"].mean(),
                    }
                )

        for outcome in ("false_negative_rate", "fp_per_gt_plus_one", "mean_translation_error_m"):
            rho, p_value = safe_spearman(frames[SCORE], frames[outcome])
            correlation_rows.append(
                {
                    "run_name": run["run_name"],
                    "model_name": run["model_name"],
                    "outcome": outcome,
                    "spearman_rho": rho,
                    "p_value": p_value,
                    "n_keyframes": int((frames[SCORE].notna() & frames[outcome].notna()).sum()),
                }
            )

        for component in [SCORE] + COMPONENTS:
            rho, p_value = safe_spearman(frames[component], frames["false_negative_rate"])
            component_rows.append(
                {
                    "run_name": run["run_name"],
                    "model_name": run["model_name"],
                    "component": component,
                    "spearman_rho_frame_fn_rate": rho,
                    "p_value": p_value,
                    "object_fn_auc": safe_auc(objects["false_negative"], objects[component]),
                }
            )

        predictors = {
            "DDI": objects[SCORE],
            "raw_object_count": objects["annotation_count"],
            "object_distance": objects["distance_ego_m"],
            "inverse_lidar_points": -np.log1p(objects["num_lidar_pts"]),
        }
        for name, predictor in predictors.items():
            baseline_rows.append(
                {
                    "run_name": run["run_name"],
                    "model_name": run["model_name"],
                    "predictor": name,
                    "object_fn_auc": safe_auc(objects["false_negative"], predictor),
                }
            )

        if run["distance_threshold_m"] == 2.0 and run["score_threshold"] == 0.05:
            model_frame = objects.copy()
            model_frame["ddi_per_10"] = model_frame[SCORE] / 10.0
            model_frame["log_distance"] = np.log1p(model_frame["distance_ego_m"])
            model_frame["log_count"] = np.log1p(model_frame["annotation_count"])
            model_frame["log_box_volume"] = np.log1p(model_frame["box_volume_m3"])
            fitted = smf.glm(
                "false_negative ~ ddi_per_10 + log_distance + log_count + "
                "log_box_volume + C(detection_name) + C(location)",
                data=model_frame,
                family=Binomial(),
            ).fit(cov_type="cluster", cov_kwds={"groups": model_frame["scene_name"]})
            coefficient = fitted.params["ddi_per_10"]
            interval = fitted.conf_int().loc["ddi_per_10"]
            regression_rows.append(
                {
                    "model_name": run["model_name"],
                    "ddi_odds_ratio_per_10_points": np.exp(coefficient),
                    "ci_low": np.exp(interval.iloc[0]),
                    "ci_high": np.exp(interval.iloc[1]),
                    "p_value": fitted.pvalues["ddi_per_10"],
                    "n_objects": int(fitted.nobs),
                    "n_scenes": model_frame["scene_name"].nunique(),
                }
            )

    groups = pd.DataFrame(group_rows)
    correlations = pd.DataFrame(correlation_rows)
    components = pd.DataFrame(component_rows)
    baselines = pd.DataFrame(baseline_rows)
    regressions = pd.DataFrame(regression_rows)
    groups.to_csv(args.output_dir / "difficulty_group_results.csv", index=False, encoding="utf-8-sig")
    correlations.to_csv(args.output_dir / "continuous_associations.csv", index=False, encoding="utf-8-sig")
    components.to_csv(args.output_dir / "component_validity.csv", index=False, encoding="utf-8-sig")
    baselines.to_csv(args.output_dir / "baseline_comparison.csv", index=False, encoding="utf-8-sig")
    regressions.to_csv(args.output_dir / "adjusted_logistic_regression.csv", index=False, encoding="utf-8-sig")

    primary = groups.loc[
        (groups["distance_threshold_m"] == 2.0)
        & (groups["score_threshold"] == 0.05)
        & (groups["scheme"] == "train_tertiles")
    ].copy()
    fig, ax = plt.subplots(figsize=(8, 4.8))
    for model, part in primary.groupby("model_name"):
        part = part.set_index("difficulty").reindex(["Easy", "Moderate", "Hard"])
        ax.plot(part.index, part["mean_frame_false_negative_rate"], marker="o", label=model)
    ax.set_ylabel("Mean frame false-negative rate")
    ax.set_xlabel("Frozen train-tertile difficulty")
    ax.set_ylim(0, 1)
    ax.grid(axis="y", alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(figures_dir / "difficulty_vs_false_negative_rate.png", dpi=180)
    plt.close(fig)

    primary_runs = [
        run for run in runs
        if run["distance_threshold_m"] == 2.0 and run["score_threshold"] == 0.05
    ]
    cross_model_rho = np.nan
    if len(primary_runs) >= 2:
        first = frame_outputs[primary_runs[0]["run_name"]][["sample_token", "false_negative_rate"]]
        second = frame_outputs[primary_runs[1]["run_name"]][["sample_token", "false_negative_rate"]]
        paired = first.merge(second, on="sample_token", suffixes=("_a", "_b"))
        cross_model_rho, _ = safe_spearman(
            paired["false_negative_rate_a"], paired["false_negative_rate_b"])

    primary_summary = []
    for model, part in primary.groupby("model_name"):
        ordered = part.set_index("difficulty").reindex(["Easy", "Moderate", "Hard"])
        rates = ordered["mean_frame_false_negative_rate"].tolist()
        primary_summary.append(
            {
                "model_name": model,
                "easy_fn_rate": rates[0],
                "moderate_fn_rate": rates[1],
                "hard_fn_rate": rates[2],
                "strict_monotonic": bool(rates[0] < rates[1] < rates[2]),
            }
        )
    pd.DataFrame(primary_summary).to_csv(
        args.output_dir / "primary_conclusion.csv", index=False, encoding="utf-8-sig")

    all_monotonic = bool(primary_summary) and all(
        item["strict_monotonic"] for item in primary_summary
    )
    positive_adjusted = bool(len(regressions)) and bool(
        (regressions["ddi_odds_ratio_per_10_points"] > 1).all()
    )
    verdict = "지지" if all_monotonic and positive_adjusted else "미지지"
    sensitivity_trends = []
    for run_name, part in groups.loc[groups["scheme"] == "train_tertiles"].groupby("run_name"):
        ordered = part.set_index("difficulty").reindex(["Easy", "Moderate", "Hard"])
        rates = ordered["mean_frame_false_negative_rate"].tolist()
        sensitivity_trends.append(bool(rates[0] < rates[1] < rates[2]))
    fixed_primary = groups.loc[
        (groups["run_name"] == primary_runs[0]["run_name"])
        & (groups["scheme"] == "fixed_33_67")
    ].set_index("difficulty")

    lines = [
        "# nuScenes 인지 DDI 모델 검증 보고서",
        "",
        "## 1. 검증 질문",
        "",
        "제안한 인지 DDI가 높을수록 서로 다른 자율주행 인지 모델의 실제 검출 오류가 일관되게 증가하는지 검증한다.",
        "",
        "## 2. 데이터와 분할",
        "",
        f"- 완전 keyframe: {len(samples):,}개, {samples['scene_name'].nunique()}개 장면",
        f"- 기준 개발용 공식 train: {len(train):,}개 keyframe",
        f"- 최종 모델 검증용 공식 validation: {len(val):,}개 keyframe, {val['scene_name'].nunique()}개 장면",
        "- validation 카메라 추론: keyframe당 6개 카메라",
        "- intermediate sweep는 사용하지 않았다.",
        "",
        "## 3. 사전 동결한 난이도 경계",
        "",
        f"- train 분포 3분위 경계: Easy <= {q_low:.2f}, Moderate <= {q_high:.2f}, Hard > {q_high:.2f}",
        "- 이 경계는 validation 모델 오류를 보기 전에 train DDI 분포만으로 정했다.",
        "- 발표안의 고정 33.33/66.67 경계도 비교 기준으로 유지했다.",
        "",
        "## 4. 모델과 오류 정의",
        "",
        "- 모델: FCOS3D, PGD의 공식 nuScenes monocular fine-tuned checkpoint",
        "- 주 분석: confidence 0.05, 동일 클래스 중심거리 2m 이내를 true positive로 매칭",
        "- 지표: false-negative rate, false positives, matched-box translation error",
        "- 민감도 분석: 중심거리 1m/4m, confidence 0.10/0.20",
        "",
        "## 5. 주 결과",
        "",
    ]
    for item in primary_summary:
        lines.append(
            f"- {item['model_name']}: Easy {item['easy_fn_rate']:.3f}, "
            f"Moderate {item['moderate_fn_rate']:.3f}, Hard {item['hard_fn_rate']:.3f}; "
            f"엄격한 단조 증가={'충족' if item['strict_monotonic'] else '미충족'}"
        )
    for row in regressions.itertuples(index=False):
        lines.append(
            f"- {row.model_name} 조정모형: DDI 10점 증가 OR={row.ddi_odds_ratio_per_10_points:.3f} "
            f"(95% CI {row.ci_low:.3f}-{row.ci_high:.3f}, p={row.p_value:.4g})"
        )
    primary_correlations = correlations.loc[
        correlations["run_name"].isin([run["run_name"] for run in primary_runs])
        & (correlations["outcome"] == "false_negative_rate")
    ]
    for row in primary_correlations.itertuples(index=False):
        lines.append(
            f"- {row.model_name} 연속 DDI-frame FN Spearman rho={row.spearman_rho:.3f} "
            f"(p={row.p_value:.4g})"
        )
    lines.extend(
        [
            f"- 두 모델 frame false-negative rate의 Spearman rho: {cross_model_rho:.3f}",
            f"- 사전 정의된 수용 기준에 대한 현재 판정: **{verdict}**",
            f"- 민감도 조건 중 엄격한 Easy < Moderate < Hard 충족: {sum(sensitivity_trends)}/{len(sensitivity_trends)}개",
            f"- 고정 33.33/66.67 경계의 validation keyframe 수: Easy {int(fixed_primary.at['Easy', 'keyframes'])}, Moderate {int(fixed_primary.at['Moderate', 'keyframes'])}, Hard {int(fixed_primary.at['Hard', 'keyframes'])}",
            "",
            "### 단순 기준선 비교",
            "",
        ]
    )
    primary_baselines = baselines.loc[
        baselines["run_name"].isin([run["run_name"] for run in primary_runs])
    ]
    for row in primary_baselines.itertuples(index=False):
        lines.append(f"- {row.model_name} / {row.predictor}: object FN AUC={row.object_fn_auc:.3f}")
    lines.extend(
        [
            "",
            "### 지표별 해석",
            "",
            "주 분석에서 각 구성요소와 frame FN rate의 Spearman rho는 다음과 같다.",
            "",
        ]
    )
    primary_components = components.loc[
        components["run_name"].isin([run["run_name"] for run in primary_runs])
    ]
    for component in [SCORE] + COMPONENTS:
        part = primary_components.loc[primary_components["component"] == component]
        values = ", ".join(
            f"{row.model_name} {row.spearman_rho_frame_fn_rate:.3f}"
            for row in part.itertuples(index=False)
        )
        lines.append(f"- {component}: {values}")
    lines.extend(
        [
            "",
            "평균 가시성 난이도는 두 모델에서 일관된 양의 관계를 보였다. 반면 객체 수는 거의 무관했고, truncation proxy는 두 모델 모두 역방향이었다. LiDAR 희소성 평균은 양의 관계였지만 카메라 모델 결과이므로 거리 교란을 포함한 간접 신호로만 해석해야 한다. 현재 합성 점수는 이 상반된 구성요소와 포화된 maximum 항을 평균하여 유효 신호를 상쇄했다.",
            "",
            "## 6. 해석 원칙",
            "",
            "- 두 모델에서 Easy < Moderate < Hard가 반복되고 연속 DDI 효과도 양수이면 단계 분류의 예측 타당성을 지지한다.",
            "- 한 모델에서만 성립하거나 민감도 조건에 따라 방향이 바뀌면 모델 일반화 또는 강건성이 부족하다.",
            "- 단계 경계는 실패하지만 연속 DDI와 오류가 관련되면 연속 지표만 유지하는 결론이 가능하다.",
            "- DDI가 단순 객체 수·거리 기준선보다 낫지 않으면 현재 합성 방식의 추가 타당성은 입증되지 않는다.",
            "",
            "## 7. 한계",
            "",
            "- 현재 자료는 trainval01 shard의 85개 완전 장면이며 전체 nuScenes trainval 850개 장면이 아니다.",
            "- 검증 모델 둘 다 monocular 계열이므로 센서 양식이 다른 LiDAR·fusion 모델로 외적 검증이 필요하다.",
            "- 따라서 이번 결과는 camera-aligned 지표와 presentation_legacy_v0 기준선의 검증이며, LiDAR 희소성을 포함한 통합 인지 DDI의 완전 검증은 아니다.",
            "- keyframe-only 실험이므로 temporal sweep 활용 난이도는 검증하지 않는다.",
            "- train 3분위 경계는 객관적 정답이 아니라 오류를 보지 않고 동결한 잠정 운영 경계다.",
            "",
            "## 8. 산출물",
            "",
            "세부 수치와 재현 가능한 결과는 같은 폴더의 CSV 및 figures 디렉터리에 저장했다.",
        ]
    )
    (args.output_dir / "인지_DDI_모델_검증_보고서.md").write_text(
        "\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps({"train_thresholds": [q_low, q_high], "runs": len(runs)}, indent=2))


if __name__ == "__main__":
    main()
