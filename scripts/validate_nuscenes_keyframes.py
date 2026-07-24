"""Validate a keyframe-only nuScenes trainval package before research use."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

from nuscenes.nuscenes import NuScenes


CAMERAS = (
    "CAM_FRONT",
    "CAM_FRONT_RIGHT",
    "CAM_BACK_RIGHT",
    "CAM_BACK",
    "CAM_BACK_LEFT",
    "CAM_FRONT_LEFT",
)
RADARS = (
    "RADAR_FRONT",
    "RADAR_FRONT_LEFT",
    "RADAR_FRONT_RIGHT",
    "RADAR_BACK_LEFT",
    "RADAR_BACK_RIGHT",
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataroot", required=True)
    parser.add_argument("--version", default="v1.0-trainval")
    parser.add_argument("--output", required=True)
    parser.add_argument("--require-radar", action="store_true")
    args = parser.parse_args()

    dataroot = Path(args.dataroot)
    required_channels = (*CAMERAS, "LIDAR_TOP", *(RADARS if args.require_radar else ()))
    nusc = NuScenes(version=args.version, dataroot=str(dataroot), verbose=False)

    missing_channels: Counter[str] = Counter()
    non_keyframe_links: Counter[str] = Counter()
    missing_files: Counter[str] = Counter()
    zero_byte_files: Counter[str] = Counter()
    checked_bytes: Counter[str] = Counter()
    complete_samples_by_scene: Counter[str] = Counter()
    total_samples_by_scene: Counter[str] = Counter()
    missing_examples: list[str] = []

    for sample in nusc.sample:
        total_samples_by_scene[sample["scene_token"]] += 1
        sample_complete = True
        for channel in required_channels:
            token = sample["data"].get(channel)
            if token is None:
                missing_channels[channel] += 1
                sample_complete = False
                continue

            sample_data = nusc.get("sample_data", token)
            if not sample_data["is_key_frame"]:
                non_keyframe_links[channel] += 1

            path = dataroot / sample_data["filename"]
            if not path.is_file():
                missing_files[channel] += 1
                sample_complete = False
                if len(missing_examples) < 100:
                    missing_examples.append(str(path))
                continue
            size = path.stat().st_size
            checked_bytes[channel] += size
            if size == 0:
                zero_byte_files[channel] += 1
                sample_complete = False

        if sample_complete:
            complete_samples_by_scene[sample["scene_token"]] += 1

    expected_counts = {
        "v1.0-trainval": {"scenes": 850, "samples": 34149},
        "v1.0-mini": {"scenes": 10, "samples": 404},
    }.get(args.version)
    observed_counts = {
        "scenes": len(nusc.scene),
        "samples": len(nusc.sample),
        "annotations": len(nusc.sample_annotation),
    }
    count_checks = {
        key: observed_counts[key] == expected
        for key, expected in (expected_counts or {}).items()
    }

    complete_sample_count = sum(complete_samples_by_scene.values())
    available_scene_count = sum(count > 0 for count in complete_samples_by_scene.values())
    complete_scene_count = sum(
        complete_samples_by_scene[token] == total
        for token, total in total_samples_by_scene.items()
    )
    partial_scene_count = sum(
        0 < complete_samples_by_scene[token] < total
        for token, total in total_samples_by_scene.items()
    )

    maps_present = (dataroot / "maps").is_dir()
    passed = (
        all(count_checks.values())
        and not missing_channels
        and not non_keyframe_links
        and not missing_files
        and not zero_byte_files
        and maps_present
    )

    report = {
        "passed": passed,
        "dataroot": str(dataroot.resolve()),
        "version": args.version,
        "required_channels": list(required_channels),
        "expected_counts": expected_counts,
        "observed_counts": observed_counts,
        "count_checks": count_checks,
        "keyframe_coverage": {
            "complete_samples": complete_sample_count,
            "missing_samples": len(nusc.sample) - complete_sample_count,
            "complete_sample_pct": round(complete_sample_count / len(nusc.sample) * 100, 3),
            "available_scenes": available_scene_count,
            "complete_scenes": complete_scene_count,
            "partial_scenes": partial_scene_count,
        },
        "maps_present": maps_present,
        "missing_channels": dict(missing_channels),
        "non_keyframe_links": dict(non_keyframe_links),
        "missing_files": dict(missing_files),
        "zero_byte_files": dict(zero_byte_files),
        "checked_size_gib_by_channel": {
            channel: round(size / (1024**3), 3) for channel, size in checked_bytes.items()
        },
        "missing_file_examples": missing_examples,
    }

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))
    raise SystemExit(0 if passed else 1)


if __name__ == "__main__":
    main()
