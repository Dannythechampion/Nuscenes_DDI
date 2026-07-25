#!/usr/bin/env python3
"""Create MMDetection3D nuScenes infos and retain an explicit token subset."""

from __future__ import annotations

import argparse
import csv
import gc
from pathlib import Path

import mmengine


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--tokens", type=Path, required=True)
    parser.add_argument("--all-samples-csv", type=Path, required=True)
    parser.add_argument("--prefix", default="nuscenes_keyframes")
    parser.add_argument(
        "--max-sweeps",
        type=int,
        default=0,
        help="Historical LiDAR sweeps in generated infos (0 for camera keyframe models).",
    )
    return parser.parse_args()


def sample_token(info: dict) -> str:
    for key in ("token", "sample_token"):
        value = info.get(key)
        if value:
            return str(value)
    raise KeyError(f"No sample token field in info keys: {sorted(info)}")


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    raw_dir = args.output_dir / "raw"
    v2_dir = args.output_dir / "v2"
    raw_dir.mkdir(exist_ok=True)
    v2_dir.mkdir(exist_ok=True)

    from nuscenes.nuscenes import NuScenes
    from nuscenes.utils import splits
    from tools.dataset_converters.nuscenes_converter import _fill_trainval_infos
    from tools.dataset_converters.update_infos_to_v2 import update_pkl_infos

    raw_prefix = raw_dir / args.prefix
    raw_train = raw_dir / f"{args.prefix}_infos_train.pkl"
    raw_val = raw_dir / f"{args.prefix}_infos_val.pkl"
    v2_train = v2_dir / raw_train.name
    v2_val = v2_dir / raw_val.name

    if not raw_val.exists():
        with args.all_samples_csv.open(encoding="utf-8-sig", newline="") as handle:
            complete_tokens = {row["sample_token"] for row in csv.DictReader(handle)}
        nusc = NuScenes(
            version="v1.0-trainval", dataroot=str(args.data_root), verbose=True)
        nusc.sample = [
            sample for sample in nusc.sample if sample["token"] in complete_tokens
        ]
        # NuScenes.get() indexes directly into each table. Rebuild the sample
        # index after filtering so annotation/velocity lookups remain correct.
        nusc._token2ind["sample"] = {
            sample["token"]: index for index, sample in enumerate(nusc.sample)
        }
        scene_names = {scene["token"]: scene["name"] for scene in nusc.scene}
        train_names = set(splits.train)
        val_names = set(splits.val)
        present_scenes = {sample["scene_token"] for sample in nusc.sample}
        train_scenes = {
            token for token in present_scenes if scene_names[token] in train_names
        }
        val_scenes = {
            token for token in present_scenes if scene_names[token] in val_names
        }
        train_infos, val_infos = _fill_trainval_infos(
            nusc,
            train_scenes,
            val_scenes,
            test=False,
            max_sweeps=args.max_sweeps,
        )
        metadata = {"version": "v1.0-trainval"}
        mmengine.dump({"infos": train_infos, "metadata": metadata}, raw_train)
        mmengine.dump({"infos": val_infos, "metadata": metadata}, raw_val)
        print(
            f"raw_train={len(train_infos)} raw_val={len(val_infos)} "
            f"complete={len(complete_tokens)}"
        )
        del nusc, train_infos, val_infos
        gc.collect()
    raw_train_payload = mmengine.load(raw_train)
    if not v2_train.exists() and raw_train_payload["infos"]:
        update_pkl_infos("nuscenes", out_dir=str(v2_dir), pkl_path=str(raw_train))
    if not v2_val.exists():
        update_pkl_infos("nuscenes", out_dir=str(v2_dir), pkl_path=str(raw_val))

    allowed = {
        line.strip() for line in args.tokens.read_text(encoding="utf-8").splitlines()
        if line.strip()
    }
    payload = mmengine.load(v2_val)
    rows = payload["data_list"]
    print(
        "first_v2_identity="
        + repr({key: rows[0].get(key) for key in ("sample_idx", "token", "timestamp")})
    )
    kept = [row for row in rows if sample_token(row) in allowed]
    missing = allowed - {sample_token(row) for row in kept}
    if missing:
        raise RuntimeError(f"Missing {len(missing)} requested validation tokens")

    payload["data_list"] = kept
    filtered = v2_dir / f"{args.prefix}_infos_val_subset.pkl"
    mmengine.dump(payload, filtered)
    print(f"full_val={len(rows)} subset_val={len(kept)} output={filtered}")


if __name__ == "__main__":
    main()
