"""Match official-format nuScenes predictions to object-level DDI features.

Predictions are processed in descending confidence order. Each prediction can
match at most one ground-truth object of the same detection class within the
configured center-distance threshold, following the nuScenes AP matching idea.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd
from nuscenes.eval.common.config import config_factory
from nuscenes.eval.detection.utils import category_to_detection_name
from nuscenes.nuscenes import NuScenes
from pyquaternion import Quaternion


def yaw_from_quaternion(rotation: list[float]) -> float:
    matrix = Quaternion(rotation).rotation_matrix
    return math.atan2(matrix[1, 0], matrix[0, 0])


def angle_error(first: float, second: float, period: float = 2 * math.pi) -> float:
    difference = (first - second + period / 2) % period - period / 2
    return abs(difference)


def scale_error(gt_size: list[float], pred_size: list[float]) -> float:
    gt = np.asarray(gt_size, dtype=float)
    pred = np.asarray(pred_size, dtype=float)
    intersection = np.minimum(gt, pred).prod()
    union = np.maximum(gt, pred).prod()
    return float(1.0 - intersection / union) if union > 0 else np.nan


def load_predictions(path: Path) -> dict[str, list[dict]]:
    content = json.loads(path.read_text(encoding="utf-8"))
    results = content.get("results")
    if not isinstance(results, dict):
        raise ValueError("Prediction JSON must contain a 'results' object keyed by sample token.")
    return results


def build_gt_rows(nusc: NuScenes, features: pd.DataFrame, class_range: dict[str, float]) -> pd.DataFrame:
    feature_by_token = features.set_index("annotation_token", drop=False)
    rows = []

    for annotation_token in feature_by_token.index:
        ann = nusc.get("sample_annotation", annotation_token)
        detection_name = category_to_detection_name(ann["category_name"])
        if detection_name is None:
            continue
        feature = feature_by_token.loc[ann["token"]].to_dict()
        eligible = (
            ann["num_lidar_pts"] + ann["num_radar_pts"] > 0
            and feature["distance_ego_m"] < class_range[detection_name]
        )
        if not eligible:
            continue
        rows.append(
            {
                **feature,
                "detection_name": detection_name,
                "gt_translation": ann["translation"],
                "gt_size": ann["size"],
                "gt_rotation": ann["rotation"],
            }
        )
    return pd.DataFrame(rows)


def filter_predictions(
    predictions: list[dict],
    class_range: dict[str, float],
    ego_xy: np.ndarray,
    score_threshold: float,
) -> list[dict]:
    filtered = []
    for prediction in predictions:
        if float(prediction.get("detection_score", 0.0)) < score_threshold:
            continue
        detection_name = prediction.get("detection_name")
        if detection_name not in class_range:
            continue
        translation = np.asarray(prediction["translation"], dtype=float)
        if np.linalg.norm(translation[:2] - ego_xy) >= class_range[detection_name]:
            continue
        filtered.append(prediction)
    return sorted(filtered, key=lambda item: float(item.get("detection_score", 0.0)), reverse=True)


def match_sample(
    nusc: NuScenes,
    sample_token: str,
    gt_indices: list[int],
    gt_df: pd.DataFrame,
    predictions: list[dict],
    class_range: dict[str, float],
    threshold: float,
    score_threshold: float,
) -> tuple[list[dict], list[dict]]:
    sample = nusc.get("sample", sample_token)
    lidar_data = nusc.get("sample_data", sample["data"]["LIDAR_TOP"])
    ego_pose = nusc.get("ego_pose", lidar_data["ego_pose_token"])
    ego_xy = np.asarray(ego_pose["translation"][:2], dtype=float)
    predictions = filter_predictions(predictions, class_range, ego_xy, score_threshold)

    unmatched_gt = set(gt_indices)
    matches: dict[int, dict] = {}
    false_positives: list[dict] = []

    for prediction in predictions:
        pred_xy = np.asarray(prediction["translation"][:2], dtype=float)
        candidates = [
            index
            for index in unmatched_gt
            if gt_df.at[index, "detection_name"] == prediction["detection_name"]
        ]
        if not candidates:
            false_positives.append(prediction)
            continue

        distances = {
            index: float(np.linalg.norm(np.asarray(gt_df.at[index, "gt_translation"][:2]) - pred_xy))
            for index in candidates
        }
        best_index = min(distances, key=distances.get)
        if distances[best_index] >= threshold:
            false_positives.append(prediction)
            continue

        unmatched_gt.remove(best_index)
        gt_yaw = yaw_from_quaternion(gt_df.at[best_index, "gt_rotation"])
        pred_yaw = yaw_from_quaternion(prediction["rotation"])
        period = math.pi if prediction["detection_name"] == "barrier" else 2 * math.pi
        matches[best_index] = {
            "pred_detection_score": float(prediction.get("detection_score", np.nan)),
            "translation_error_m": distances[best_index],
            "scale_error": scale_error(gt_df.at[best_index, "gt_size"], prediction["size"]),
            "orientation_error_rad": angle_error(gt_yaw, pred_yaw, period),
        }

    object_results = []
    for index in gt_indices:
        row = gt_df.loc[index].drop(labels=["gt_translation", "gt_size", "gt_rotation"]).to_dict()
        matched = index in matches
        row.update(
            {
                "matched": matched,
                "false_negative": not matched,
                "match_distance_threshold_m": threshold,
                **matches.get(
                    index,
                    {
                        "pred_detection_score": np.nan,
                        "translation_error_m": np.nan,
                        "scale_error": np.nan,
                        "orientation_error_rad": np.nan,
                    },
                ),
            }
        )
        object_results.append(row)

    fp_rows = [
        {
            "sample_token": sample_token,
            "detection_name": prediction["detection_name"],
            "detection_score": float(prediction.get("detection_score", np.nan)),
        }
        for prediction in false_positives
    ]
    return object_results, fp_rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataroot", required=True)
    parser.add_argument("--version", default="v1.0-trainval")
    parser.add_argument("--prediction-json", required=True)
    parser.add_argument("--objects-csv", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--model-name", required=True)
    parser.add_argument("--distance-threshold", type=float, default=2.0)
    parser.add_argument("--score-threshold", type=float, default=0.05)
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    nusc = NuScenes(version=args.version, dataroot=args.dataroot, verbose=False)
    config = config_factory("detection_cvpr_2019")
    features = pd.read_csv(args.objects_csv)
    gt_df = build_gt_rows(nusc, features, config.class_range)
    predictions = load_predictions(Path(args.prediction_json))

    gt_indices_by_sample = gt_df.groupby("sample_token").groups
    object_rows: list[dict] = []
    fp_rows: list[dict] = []
    # The prediction JSON defines the evaluated split. Including feature-only
    # tokens would incorrectly count non-inferred train samples as all-FN.
    evaluation_tokens = sorted(predictions)
    scene_by_sample: dict[str, str] = {}
    for sample_token in evaluation_tokens:
        sample = nusc.get("sample", sample_token)
        scene_by_sample[sample_token] = nusc.get("scene", sample["scene_token"])["name"]
        indices = list(gt_indices_by_sample.get(sample_token, []))
        matched, fps = match_sample(
            nusc,
            sample_token,
            indices,
            gt_df,
            predictions.get(sample_token, []),
            config.class_range,
            args.distance_threshold,
            args.score_threshold,
        )
        object_rows.extend(matched)
        fp_rows.extend(fps)

    object_errors = pd.DataFrame(object_rows)
    false_positives = pd.DataFrame(fp_rows, columns=["sample_token", "detection_name", "detection_score"])
    frame_errors = object_errors.groupby(["scene_name", "sample_token"], as_index=False).agg(
        gt_count=("annotation_token", "size"),
        false_negative_count=("false_negative", "sum"),
        false_negative_rate=("false_negative", "mean"),
        matched_count=("matched", "sum"),
        mean_translation_error_m=("translation_error_m", "mean"),
    )
    fp_counts = false_positives.groupby("sample_token").size().rename("false_positive_count")
    frame_base = pd.DataFrame(
        {
            "sample_token": evaluation_tokens,
            "scene_name": [scene_by_sample[token] for token in evaluation_tokens],
        }
    )
    frame_errors = frame_base.merge(frame_errors, on=["scene_name", "sample_token"], how="left")
    frame_errors = frame_errors.merge(fp_counts, on="sample_token", how="left")
    for column in ("gt_count", "false_negative_count", "matched_count"):
        frame_errors[column] = frame_errors[column].fillna(0).astype(int)
    frame_errors["false_positive_count"] = frame_errors["false_positive_count"].fillna(0).astype(int)

    for frame in (object_errors, false_positives, frame_errors):
        frame.insert(0, "model_name", args.model_name)
        frame.insert(1, "score_threshold", args.score_threshold)

    object_errors.to_csv(output_dir / "object_detection_errors.csv", index=False, encoding="utf-8-sig")
    false_positives.to_csv(output_dir / "false_positive_predictions.csv", index=False, encoding="utf-8-sig")
    frame_errors.to_csv(output_dir / "frame_detection_errors.csv", index=False, encoding="utf-8-sig")

    summary = {
        "model_name": args.model_name,
        "eligible_gt_objects": len(object_errors),
        "false_negative_rate": float(object_errors["false_negative"].mean()),
        "false_positives": len(false_positives),
        "distance_threshold_m": args.distance_threshold,
        "score_threshold": args.score_threshold,
    }
    (output_dir / "matching_summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
