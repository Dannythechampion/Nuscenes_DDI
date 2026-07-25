"""Build fixed train/validation manifests for complete nuScenes keyframes."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
from nuscenes.nuscenes import NuScenes
from nuscenes.utils.splits import create_splits_scenes


REQUIRED_CHANNELS = (
    "CAM_FRONT",
    "CAM_FRONT_RIGHT",
    "CAM_BACK_RIGHT",
    "CAM_BACK",
    "CAM_BACK_LEFT",
    "CAM_FRONT_LEFT",
    "LIDAR_TOP",
)


def sample_is_complete(nusc: NuScenes, dataroot: Path, sample: dict) -> bool:
    return all(
        channel in sample["data"]
        and (dataroot / nusc.get("sample_data", sample["data"][channel])["filename"]).is_file()
        for channel in REQUIRED_CHANNELS
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataroot", required=True)
    parser.add_argument("--version", default="v1.0-trainval")
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()

    dataroot = Path(args.dataroot)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    nusc = NuScenes(version=args.version, dataroot=str(dataroot), verbose=False)
    split_scenes = create_splits_scenes()
    split_by_scene = {
        scene_name: split
        for split in ("train", "val")
        for scene_name in split_scenes[split]
    }
    scene_by_token = {scene["token"]: scene for scene in nusc.scene}
    log_by_token = {log["token"]: log for log in nusc.log}

    sample_rows: list[dict] = []
    for sample in nusc.sample:
        if not sample_is_complete(nusc, dataroot, sample):
            continue
        scene = scene_by_token[sample["scene_token"]]
        location = log_by_token[scene["log_token"]]["location"]
        sample_rows.append(
            {
                "official_split": split_by_scene.get(scene["name"], "other"),
                "scene_name": scene["name"],
                "scene_token": scene["token"],
                "sample_token": sample["token"],
                "timestamp": sample["timestamp"],
                "first_sample_token": scene["first_sample_token"],
                "last_sample_token": scene["last_sample_token"],
                "description": scene["description"],
                "location": location,
            }
        )

    sample_df = pd.DataFrame(sample_rows).sort_values(
        ["official_split", "scene_name", "timestamp"]
    )
    scene_df = (
        sample_df.groupby(
            ["official_split", "scene_name", "scene_token", "description", "location"],
            as_index=False,
        )
        .agg(
            sample_count=("sample_token", "count"),
            first_timestamp=("timestamp", "min"),
            last_timestamp=("timestamp", "max"),
        )
        .sort_values(["official_split", "scene_name"])
    )

    sample_path = output_dir / "complete_keyframe_samples.csv"
    scene_path = output_dir / "complete_keyframe_scenes.csv"
    val_token_path = output_dir / "validation_sample_tokens.txt"
    sample_df.to_csv(sample_path, index=False, encoding="utf-8-sig")
    scene_df.to_csv(scene_path, index=False, encoding="utf-8-sig")
    val_tokens = sample_df.loc[sample_df["official_split"] == "val", "sample_token"]
    val_scene_count = int(
        scene_df.loc[scene_df["official_split"] == "val", "scene_name"].nunique()
    )
    if val_scene_count != 150 or len(val_tokens) != 6019:
        raise RuntimeError(
            "Official validation cohort is incomplete: "
            f"{val_scene_count}/150 scenes, {len(val_tokens)}/6019 keyframes"
        )
    val_token_path.write_text("\n".join(val_tokens) + "\n", encoding="ascii")

    print(scene_df.groupby("official_split")["scene_name"].count().to_string())
    print(sample_df.groupby("official_split")["sample_token"].count().to_string())
    print(f"Wrote {len(sample_df)} samples to {sample_path}")
    print(f"Wrote {len(scene_df)} scenes to {scene_path}")
    print(f"Wrote {len(val_tokens)} validation tokens to {val_token_path}")


if __name__ == "__main__":
    main()
