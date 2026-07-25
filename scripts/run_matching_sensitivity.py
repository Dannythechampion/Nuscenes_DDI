#!/usr/bin/env python3
"""Run the frozen nuScenes matching sensitivity matrix for multiple models."""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


CONDITIONS = (
    (1.0, 0.05),
    (2.0, 0.05),
    (2.0, 0.10),
    (2.0, 0.20),
    (4.0, 0.05),
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataroot", required=True)
    parser.add_argument("--version", default="v1.0-trainval")
    parser.add_argument("--objects-csv", required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument(
        "--model",
        action="append",
        required=True,
        metavar="NAME=PREDICTION_JSON",
    )
    return parser.parse_args()


def slug(model: str, distance: float, score: float) -> str:
    score_text = f"{score:.2f}".replace(".", "")
    return f"{model.lower()}_d{distance:g}_s{score_text}"


def main() -> None:
    args = parse_args()
    matcher = Path(__file__).with_name("match_nuscenes_detection_errors.py")
    models = []
    for spec in args.model:
        if "=" not in spec:
            raise ValueError(f"Expected NAME=PREDICTION_JSON, got {spec!r}")
        name, prediction = spec.split("=", 1)
        models.append((name, prediction))

    for name, prediction in models:
        for distance, score in CONDITIONS:
            output_dir = args.output_root / slug(name, distance, score)
            summary = output_dir / "matching_summary.json"
            if summary.is_file():
                print(f"skip existing {summary}", flush=True)
                continue
            command = [
                sys.executable,
                str(matcher),
                "--dataroot",
                args.dataroot,
                "--version",
                args.version,
                "--prediction-json",
                prediction,
                "--objects-csv",
                args.objects_csv,
                "--output-dir",
                str(output_dir),
                "--model-name",
                name,
                "--distance-threshold",
                str(distance),
                "--score-threshold",
                str(score),
            ]
            print(f"run {output_dir.name}", flush=True)
            subprocess.run(command, check=True)


if __name__ == "__main__":
    main()
