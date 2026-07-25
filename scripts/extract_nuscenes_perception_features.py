"""Extract nuScenes perception-difficulty proxy features.

This script computes the perception side of the team's DDI proposal from
nuScenes metadata:

- object / VRU / vehicle / static counts
- visibility difficulty from official nuScenes visibility tokens
- LiDAR sparsity difficulty from num_lidar_pts
- camera truncation proxy by projecting 3D boxes into the 6 camera images

Outputs both sample-level and scene-level CSV files.
"""

from __future__ import annotations

import argparse
import math
from collections import Counter
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
from nuscenes.eval.common.config import config_factory
from nuscenes.eval.detection.utils import category_to_detection_name
from nuscenes.nuscenes import NuScenes
from nuscenes.utils.splits import create_splits_scenes
from nuscenes.utils.geometry_utils import BoxVisibility, view_points


CAMERA_CHANNELS = (
    "CAM_FRONT",
    "CAM_FRONT_RIGHT",
    "CAM_BACK_RIGHT",
    "CAM_BACK",
    "CAM_BACK_LEFT",
    "CAM_FRONT_LEFT",
)

VRU_PREFIXES = ("human.pedestrian", "vehicle.bicycle", "vehicle.motorcycle")
VEHICLE_PREFIXES = ("vehicle.car", "vehicle.bus", "vehicle.truck", "vehicle.trailer", "vehicle.construction")
STATIC_PREFIXES = ("movable_object.trafficcone", "movable_object.barrier", "static_object")

VISIBILITY_SCORE = {
    "1": 10.0,  # 0-40% visible
    "2": 7.0,   # 40-60% visible
    "3": 3.0,   # 60-80% visible
    "4": 0.0,   # 80-100% visible
}

# Official nuScenes visibility bins converted to hidden-fraction midpoint severity.
VISIBILITY_PRIOR_SCORE = {
    "1": 8.0,  # 0-40% visible -> midpoint 20% visible -> 80% hidden.
    "2": 5.0,
    "3": 3.0,
    "4": 1.0,
}

VISIBILITY_PRIOR_LEVEL = {
    "1": 2,  # Hard: at most 40% visible.
    "2": 1,  # Moderate: 40-60% visible.
    "3": 1,  # Moderate: 60-80% visible.
    "4": 0,  # Easy: 80-100% visible.
}


def score_count(weighted_complexity: float, raw_count: int) -> float:
    if raw_count >= 40:
        return 10.0
    if weighted_complexity <= 5:
        return 0.0
    if weighted_complexity <= 15:
        return 5.0
    return 10.0


def score_lidar_points(num_lidar_pts: int) -> float:
    if num_lidar_pts >= 50:
        return 0.0
    if num_lidar_pts >= 20:
        return 3.0
    if num_lidar_pts >= 5:
        return 7.0
    return 10.0


def score_truncation(truncation_ratio: float | None) -> float:
    if truncation_ratio is None or math.isnan(truncation_ratio):
        return np.nan
    if truncation_ratio <= 0.05:
        return 0.0
    if truncation_ratio <= 0.30:
        return 3.0
    if truncation_ratio <= 0.50:
        return 7.0
    return 10.0


def score_kitti_truncation(truncation_ratio: float | None) -> float:
    """Map KITTI's 15/30/50% truncation boundaries to a 0-10 severity scale."""
    if truncation_ratio is None or math.isnan(truncation_ratio):
        return np.nan
    if truncation_ratio <= 0.15:
        return 0.0
    if truncation_ratio <= 0.30:
        return 10.0 / 3.0
    if truncation_ratio <= 0.50:
        return 20.0 / 3.0
    return 10.0


def kitti_truncation_level(truncation_ratio: float | None) -> float:
    if truncation_ratio is None or math.isnan(truncation_ratio):
        return np.nan
    if truncation_ratio <= 0.15:
        return 0.0
    if truncation_ratio <= 0.30:
        return 1.0
    return 2.0


def lidar_point_level(num_lidar_pts: int) -> int:
    """Use Waymo's five-point boundary plus the physical zero-return case."""
    if num_lidar_pts == 0:
        return 2
    if num_lidar_pts <= 5:
        return 1
    return 0


def ordinal_name(level: int) -> str:
    return ("Easy", "Moderate", "Hard")[level]


def quantile_or_nan(values: Iterable[float], quantile: float) -> float:
    arr = np.array([v for v in values if not pd.isna(v)], dtype=float)
    return float(np.quantile(arr, quantile)) if arr.size else np.nan


def mean_components(*values: float) -> float:
    valid = [value for value in values if not pd.isna(value)]
    return float(np.mean(valid)) if valid else np.nan


def ego_speed_mps(nusc: NuScenes, sample: dict) -> float:
    """Estimate ego speed from adjacent keyframe LiDAR ego poses."""
    neighbors = []
    for key in ("prev", "next"):
        token = sample.get(key)
        if not token:
            continue
        other = nusc.get("sample", token)
        sample_data = nusc.get("sample_data", other["data"]["LIDAR_TOP"])
        pose = nusc.get("ego_pose", sample_data["ego_pose_token"])
        neighbors.append((other["timestamp"], np.array(pose["translation"][:2], dtype=float)))
    current_data = nusc.get("sample_data", sample["data"]["LIDAR_TOP"])
    current_pose = nusc.get("ego_pose", current_data["ego_pose_token"])
    current = (sample["timestamp"], np.array(current_pose["translation"][:2], dtype=float))
    if len(neighbors) == 2:
        before, after = sorted(neighbors, key=lambda item: item[0])
        seconds = (after[0] - before[0]) / 1e6
        return float(np.linalg.norm(after[1] - before[1]) / seconds) if seconds > 0 else np.nan
    if len(neighbors) == 1:
        seconds = abs(neighbors[0][0] - current[0]) / 1e6
        return float(np.linalg.norm(neighbors[0][1] - current[1]) / seconds) if seconds > 0 else np.nan
    return np.nan


def category_group(category_name: str) -> str:
    if category_name.startswith(VRU_PREFIXES):
        return "vru"
    if category_name.startswith(VEHICLE_PREFIXES):
        return "vehicle"
    if category_name.startswith(STATIC_PREFIXES):
        return "static"
    return "other"


def clipped_bbox_area(points: np.ndarray, width: int, height: int) -> tuple[float, float] | None:
    """Return full and image-clipped 2D bbox areas for projected points."""
    if points.shape[1] == 0:
        return None
    x_min, y_min = points[:2, :].min(axis=1)
    x_max, y_max = points[:2, :].max(axis=1)
    full_w = max(0.0, x_max - x_min)
    full_h = max(0.0, y_max - y_min)
    full_area = full_w * full_h
    if full_area <= 0:
        return None

    clip_x_min = min(max(x_min, 0.0), float(width))
    clip_y_min = min(max(y_min, 0.0), float(height))
    clip_x_max = min(max(x_max, 0.0), float(width))
    clip_y_max = min(max(y_max, 0.0), float(height))
    inside_area = max(0.0, clip_x_max - clip_x_min) * max(0.0, clip_y_max - clip_y_min)
    return full_area, inside_area


def best_camera_truncation(nusc: NuScenes, sample: dict, ann_token: str) -> float | None:
    """Estimate truncation as 1 - max visible projected 2D bbox area ratio."""
    best_inside_ratio = None

    for channel in CAMERA_CHANNELS:
        sample_data_token = sample["data"].get(channel)
        if not sample_data_token:
            continue

        sample_data = nusc.get("sample_data", sample_data_token)
        _, boxes, camera_intrinsic = nusc.get_sample_data(
            sample_data_token,
            box_vis_level=BoxVisibility.NONE,
            selected_anntokens=[ann_token],
        )
        if not boxes:
            continue

        box = boxes[0]
        corners_3d = box.corners()
        in_front = corners_3d[2, :] > 0.1
        if not np.any(in_front):
            continue

        projected = view_points(corners_3d[:, in_front], np.array(camera_intrinsic), normalize=True)
        area_pair = clipped_bbox_area(projected, sample_data["width"], sample_data["height"])
        if area_pair is None:
            continue
        full_area, inside_area = area_pair
        inside_ratio = inside_area / full_area
        best_inside_ratio = inside_ratio if best_inside_ratio is None else max(best_inside_ratio, inside_ratio)

    if best_inside_ratio is None:
        return None
    return float(1.0 - min(max(best_inside_ratio, 0.0), 1.0))


def mean_or_nan(values: Iterable[float]) -> float:
    arr = np.array([v for v in values if not pd.isna(v)], dtype=float)
    return float(arr.mean()) if arr.size else np.nan


def max_or_nan(values: Iterable[float]) -> float:
    arr = np.array([v for v in values if not pd.isna(v)], dtype=float)
    return float(arr.max()) if arr.size else np.nan


def perception_score(row: dict) -> float:
    components = [
        row["count_score"],
        row["visibility_mean_score"],
        row["visibility_max_score"],
        row["lidar_sparsity_mean_score"],
        row["lidar_sparsity_max_score"],
        row["truncation_mean_score"],
        row["truncation_max_score"],
    ]
    valid = [v for v in components if not pd.isna(v)]
    return float(np.mean(valid) * 10.0) if valid else np.nan


def sample_has_keyframe_files(nusc: NuScenes, sample: dict) -> bool:
    required_channels = (
        "CAM_FRONT",
        "CAM_FRONT_RIGHT",
        "CAM_BACK_RIGHT",
        "CAM_BACK",
        "CAM_BACK_LEFT",
        "CAM_FRONT_LEFT",
        "LIDAR_TOP",
    )
    dataroot = Path(nusc.dataroot)
    return all(
        channel in sample["data"]
        and (dataroot / nusc.get("sample_data", sample["data"][channel])["filename"]).is_file()
        for channel in required_channels
    )


def extract_rows(
    nusc: NuScenes,
    include_truncation: bool,
    require_existing_keyframes: bool = False,
    selected_scene_names: set[str] | None = None,
) -> tuple[list[dict], list[dict], int]:
    scene_by_token = {scene["token"]: scene for scene in nusc.scene}
    detection_class_range = config_factory("detection_cvpr_2019").class_range
    sample_rows: list[dict] = []
    object_rows: list[dict] = []
    skipped_missing_keyframes = 0

    for sample in nusc.sample:
        scene = scene_by_token[sample["scene_token"]]
        if selected_scene_names is not None and scene["name"] not in selected_scene_names:
            continue
        if require_existing_keyframes and not sample_has_keyframe_files(nusc, sample):
            skipped_missing_keyframes += 1
            continue
        annotations = [nusc.get("sample_annotation", token) for token in sample["anns"]]

        groups = [category_group(ann["category_name"]) for ann in annotations]
        raw_count = len(annotations)
        vru_count = groups.count("vru")
        vehicle_count = groups.count("vehicle")
        static_count = groups.count("static")
        other_count = groups.count("other")
        weighted_complexity = 2.0 * vru_count + 1.0 * vehicle_count + 0.5 * static_count + 0.5 * other_count

        visibility_scores = [VISIBILITY_SCORE.get(ann["visibility_token"], np.nan) for ann in annotations]
        visibility_prior_scores = [
            VISIBILITY_PRIOR_SCORE.get(ann["visibility_token"], np.nan) for ann in annotations
        ]
        lidar_scores = [score_lidar_points(ann["num_lidar_pts"]) for ann in annotations]

        truncation_by_token: dict[str, float | None] = {}
        if include_truncation:
            for ann in annotations:
                truncation_by_token[ann["token"]] = best_camera_truncation(nusc, sample, ann["token"])

        truncation_ratios = [value for value in truncation_by_token.values() if value is not None]

        truncation_scores = [score_truncation(v) for v in truncation_ratios]

        lidar_sample_data = nusc.get("sample_data", sample["data"]["LIDAR_TOP"])
        ego_pose = nusc.get("ego_pose", lidar_sample_data["ego_pose_token"])
        ego_xy = np.array(ego_pose["translation"][:2], dtype=float)
        ego_speed = ego_speed_mps(nusc, sample)
        annotation_distances = [
            float(np.linalg.norm(np.array(ann["translation"][:2], dtype=float) - ego_xy))
            for ann in annotations
        ]
        moving_vehicle_count_8m = 0
        for ann, group, distance in zip(annotations, groups, annotation_distances, strict=True):
            attributes = {
                nusc.get("attribute", token)["name"] for token in ann["attribute_tokens"]
            }
            if group == "vehicle" and distance < 8.0 and "vehicle.moving" in attributes:
                moving_vehicle_count_8m += 1
        nuplan_near_multiple_vehicles = bool(
            moving_vehicle_count_8m > 6 and not pd.isna(ego_speed) and ego_speed > 6.0
        )
        density_prior_score = min(10.0, moving_vehicle_count_8m / 7.0 * 10.0)

        for ann, group, visibility_score, visibility_prior_score, lidar_score in zip(
            annotations,
            groups,
            visibility_scores,
            visibility_prior_scores,
            lidar_scores,
            strict=True,
        ):
            ann_xy = np.array(ann["translation"][:2], dtype=float)
            distance_ego_m = float(np.linalg.norm(ann_xy - ego_xy))
            detection_name = category_to_detection_name(ann["category_name"])
            official_eval_eligible = (
                detection_name is not None
                and ann["num_lidar_pts"] + ann["num_radar_pts"] > 0
                and distance_ego_m < detection_class_range[detection_name]
            )
            truncation_ratio = truncation_by_token.get(ann["token"])
            visibility_level = VISIBILITY_PRIOR_LEVEL.get(ann["visibility_token"], 2)
            truncation_level = kitti_truncation_level(truncation_ratio)
            camera_object_level = (
                visibility_level
                if pd.isna(truncation_level)
                else max(visibility_level, int(truncation_level))
            )
            lidar_level = lidar_point_level(ann["num_lidar_pts"])
            size_w, size_l, size_h = ann["size"]
            object_rows.append(
                {
                    "scene_name": scene["name"],
                    "sample_token": sample["token"],
                    "annotation_token": ann["token"],
                    "instance_token": ann["instance_token"],
                    "category_name": ann["category_name"],
                    "category_group": group,
                    "detection_name": detection_name,
                    "official_eval_eligible": official_eval_eligible,
                    "visibility_token": ann["visibility_token"],
                    "visibility_score": visibility_score,
                    "camera_occlusion_prior_score": visibility_prior_score,
                    "camera_occlusion_prior_level": visibility_level,
                    "num_lidar_pts": ann["num_lidar_pts"],
                    "num_radar_pts": ann["num_radar_pts"],
                    "num_sensor_pts": ann["num_lidar_pts"] + ann["num_radar_pts"],
                    "official_eval_has_point": ann["num_lidar_pts"] + ann["num_radar_pts"] > 0,
                    "lidar_sparsity_score": lidar_score,
                    "distance_ego_m": distance_ego_m,
                    "box_width_m": size_w,
                    "box_length_m": size_l,
                    "box_height_m": size_h,
                    "box_volume_m3": size_w * size_l * size_h,
                    "truncation_ratio": truncation_ratio,
                    "truncation_score": score_truncation(truncation_ratio),
                    "camera_truncation_kitti_score": score_kitti_truncation(truncation_ratio),
                    "camera_truncation_kitti_level": truncation_level,
                    "camera_object_ddi_prior": mean_components(
                        visibility_prior_score, score_kitti_truncation(truncation_ratio)
                    ),
                    "camera_object_difficulty_level": camera_object_level,
                    "camera_object_difficulty": ordinal_name(camera_object_level),
                    "lidar_object_ddi_prior": lidar_level * 5.0,
                    "lidar_object_difficulty_level": lidar_level,
                    "lidar_object_difficulty": ordinal_name(lidar_level),
                }
            )

        visibility_counts = Counter(ann["visibility_token"] for ann in annotations)
        lidar_point_values = np.array([ann["num_lidar_pts"] for ann in annotations], dtype=float)
        sensor_point_values = np.array(
            [ann["num_lidar_pts"] + ann["num_radar_pts"] for ann in annotations], dtype=float
        )
        distances = np.array(annotation_distances, dtype=float)
        detection_names = [category_to_detection_name(ann["category_name"]) for ann in annotations]
        eligible_flags = [
            name is not None
            and ann["num_lidar_pts"] + ann["num_radar_pts"] > 0
            and distance < detection_class_range[name]
            for ann, name, distance in zip(annotations, detection_names, distances, strict=True)
        ]
        detection_class_count = sum(name is not None for name in detection_names)
        official_eval_eligible_count = sum(eligible_flags)
        eligible_visibility_prior = [
            score for score, eligible in zip(visibility_prior_scores, eligible_flags, strict=True)
            if eligible
        ]
        eligible_truncation_prior = [
            score_kitti_truncation(truncation_by_token.get(ann["token"]))
            for ann, eligible in zip(annotations, eligible_flags, strict=True)
            if eligible
        ]
        eligible_lidar_points = np.array(
            [ann["num_lidar_pts"] for ann, eligible in zip(annotations, eligible_flags, strict=True) if eligible],
            dtype=float,
        )
        camera_occlusion_p75 = quantile_or_nan(eligible_visibility_prior, 0.75)
        camera_truncation_p75 = quantile_or_nan(eligible_truncation_prior, 0.75)
        lidar_le5_prop = (
            float((eligible_lidar_points <= 5).mean()) if eligible_lidar_points.size else np.nan
        )
        eligible_camera_levels = [
            max(
                VISIBILITY_PRIOR_LEVEL.get(ann["visibility_token"], 2),
                int(kitti_truncation_level(truncation_by_token.get(ann["token"])))
                if not pd.isna(kitti_truncation_level(truncation_by_token.get(ann["token"])))
                else 0,
            )
            for ann, eligible in zip(annotations, eligible_flags, strict=True)
            if eligible
        ]
        eligible_lidar_levels = [
            lidar_point_level(ann["num_lidar_pts"])
            for ann, eligible in zip(annotations, eligible_flags, strict=True)
            if eligible
        ]
        camera_frame_level = int(
            max(
                2 if nuplan_near_multiple_vehicles else 0,
                math.ceil(quantile_or_nan(eligible_camera_levels, 0.75))
                if eligible_camera_levels else 0,
            )
        )
        lidar_frame_level = int(
            max(
                2 if nuplan_near_multiple_vehicles else 0,
                math.ceil(quantile_or_nan(eligible_lidar_levels, 0.75))
                if eligible_lidar_levels else 0,
            )
        )
        camera_ddi_prior = mean_components(
            density_prior_score, camera_occlusion_p75, camera_truncation_p75
        )
        lidar_ddi_prior = mean_components(
            density_prior_score, lidar_le5_prop * 10.0
        )

        row = {
            "scene_name": scene["name"],
            "scene_description": scene["description"],
            "sample_token": sample["token"],
            "timestamp": sample["timestamp"],
            "annotation_count": raw_count,
            "official_detection_class_count": detection_class_count,
            "official_eval_eligible_count": official_eval_eligible_count,
            "official_eval_eligible_prop": official_eval_eligible_count / raw_count if raw_count else np.nan,
            "vru_count": vru_count,
            "vehicle_count": vehicle_count,
            "static_count": static_count,
            "other_count": other_count,
            "weighted_complexity": weighted_complexity,
            "count_score": score_count(weighted_complexity, raw_count),
            "ego_speed_mps": ego_speed,
            "moving_vehicle_count_8m": moving_vehicle_count_8m,
            "nuplan_near_multiple_vehicles": nuplan_near_multiple_vehicles,
            "density_prior_score": density_prior_score,
            "visibility_mean_score": mean_or_nan(visibility_scores),
            "visibility_max_score": max_or_nan(visibility_scores),
            **{
                f"visibility_token_{token}_prop": visibility_counts[token] / raw_count if raw_count else np.nan
                for token in ("1", "2", "3", "4")
            },
            "lidar_sparsity_mean_score": mean_or_nan(lidar_scores),
            "lidar_sparsity_max_score": max_or_nan(lidar_scores),
            "lidar_points_median": float(np.median(lidar_point_values)) if raw_count else np.nan,
            "lidar_points_p10": float(np.quantile(lidar_point_values, 0.10)) if raw_count else np.nan,
            "lidar_zero_point_prop": float((lidar_point_values == 0).mean()) if raw_count else np.nan,
            "official_zero_sensor_point_prop": float((sensor_point_values == 0).mean()) if raw_count else np.nan,
            "distance_ego_mean_m": float(distances.mean()) if raw_count else np.nan,
            "distance_ego_median_m": float(np.median(distances)) if raw_count else np.nan,
            "distance_over_40m_prop": float((distances > 40.0).mean()) if raw_count else np.nan,
            "truncation_mean_ratio": mean_or_nan(truncation_ratios),
            "truncation_max_ratio": max_or_nan(truncation_ratios),
            "truncation_mean_score": mean_or_nan(truncation_scores),
            "truncation_max_score": max_or_nan(truncation_scores),
            "camera_occlusion_p75_prior_score": camera_occlusion_p75,
            "camera_truncation_p75_kitti_score": camera_truncation_p75,
            "lidar_le5_point_prop": lidar_le5_prop,
            "camera_ddi_prior_v1": camera_ddi_prior,
            "lidar_ddi_prior_v1": lidar_ddi_prior,
            "camera_difficulty_prior_level_v1": camera_frame_level,
            "camera_difficulty_prior_v1": ordinal_name(camera_frame_level),
            "lidar_difficulty_prior_level_v1": lidar_frame_level,
            "lidar_difficulty_prior_v1": ordinal_name(lidar_frame_level),
        }
        row["score_version"] = "presentation_legacy_v0"
        row["perception_difficulty_score"] = perception_score(row)
        sample_rows.append(row)

    return sample_rows, object_rows, skipped_missing_keyframes


def aggregate_scene_rows(sample_df: pd.DataFrame) -> pd.DataFrame:
    numeric_cols = sample_df.select_dtypes(include=[np.number]).columns.tolist()
    grouped = sample_df.groupby(["scene_name", "scene_description"], dropna=False)
    scene_df = grouped[numeric_cols].agg(["mean", "max"])
    scene_df.columns = [f"{col}_{agg}" for col, agg in scene_df.columns]
    scene_df = scene_df.reset_index()
    return scene_df.sort_values("perception_difficulty_score_mean", ascending=False)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataroot", default="data/nuscenes")
    parser.add_argument("--version", default="v1.0-mini")
    parser.add_argument("--split", choices=("train", "val"))
    parser.add_argument("--output-dir", default="outputs/perception")
    parser.add_argument("--skip-truncation", action="store_true", help="Skip camera projection for faster metadata-only extraction.")
    parser.add_argument(
        "--require-existing-keyframes",
        action="store_true",
        help="Process only samples whose 6 camera and LIDAR_TOP keyframe files exist.",
    )
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    nusc = NuScenes(version=args.version, dataroot=args.dataroot, verbose=True)
    split_scene_names = None
    if args.split:
        split_scene_names = set(create_splits_scenes()[args.split])
        split_sample_count = sum(
            1
            for sample in nusc.sample
            if nusc.get("scene", sample["scene_token"])["name"] in split_scene_names
        )
        print(
            f"Selected official {args.split} split: "
            f"{len(split_scene_names)} scenes, {split_sample_count} keyframes"
        )
    sample_rows, object_rows, skipped_missing_keyframes = extract_rows(
        nusc,
        include_truncation=not args.skip_truncation,
        require_existing_keyframes=args.require_existing_keyframes,
        selected_scene_names=split_scene_names,
    )
    sample_df = pd.DataFrame(sample_rows).sort_values("perception_difficulty_score", ascending=False)
    object_df = pd.DataFrame(object_rows)
    scene_df = aggregate_scene_rows(sample_df)

    sample_path = output_dir / f"{args.version}_perception_samples.csv"
    object_path = output_dir / f"{args.version}_perception_objects.csv"
    scene_path = output_dir / f"{args.version}_perception_scenes.csv"
    sample_df.to_csv(sample_path, index=False, encoding="utf-8-sig")
    object_df.to_csv(object_path, index=False, encoding="utf-8-sig")
    scene_df.to_csv(scene_path, index=False, encoding="utf-8-sig")

    print(f"Wrote {len(sample_df)} sample rows to {sample_path}")
    print(f"Wrote {len(object_df)} object rows to {object_path}")
    print(f"Wrote {len(scene_df)} scene rows to {scene_path}")
    print(f"Skipped {skipped_missing_keyframes} samples with missing keyframe files")
    print("\nTop scene-level perception difficulty:")
    print(scene_df[["scene_name", "perception_difficulty_score_mean", "annotation_count_mean", "vru_count_mean", "vehicle_count_mean"]].head(10).to_string(index=False))


if __name__ == "__main__":
    main()
