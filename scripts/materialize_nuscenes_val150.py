"""Materialize the official 150-scene nuScenes validation sensor subset.

The public AWS mirror stores trainval data in ten archives.  This script downloads
one archive at a time and extracts only the camera keyframes, LiDAR keyframes, and
    optionally the ten historical LiDAR sweeps required by official nuScenes baselines.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import tarfile
import time
import urllib.error
import urllib.request
from pathlib import Path, PurePosixPath


AWS_BASE = "https://d36yt3mvayqw5m.cloudfront.net/public/v1.0"
USER_AGENT = "nuScenes-DDI-val150-materializer/1.0"


def load_json(path: Path) -> list[dict]:
    with path.open("r", encoding="utf-8") as source:
        return json.load(source)


def official_val_scene_names() -> set[str]:
    try:
        from nuscenes.utils.splits import create_splits_scenes
    except ImportError as error:
        raise SystemExit(
            "nuscenes-devkit is required. Run this script in the project WSL environment."
        ) from error
    names = set(create_splits_scenes()["val"])
    if len(names) != 150:
        raise RuntimeError(f"Expected 150 official validation scenes, found {len(names)}")
    return names


def build_requirements(metadata_root: Path, historical_sweeps: int) -> dict[str, set[str]]:
    version_root = metadata_root / "v1.0-trainval"
    scenes = load_json(version_root / "scene.json")
    samples = load_json(version_root / "sample.json")
    sample_data = load_json(version_root / "sample_data.json")

    val_names = official_val_scene_names()
    val_scene_tokens = {row["token"] for row in scenes if row["name"] in val_names}
    if len(val_scene_tokens) != 150:
        raise RuntimeError(
            f"Metadata contains {len(val_scene_tokens)}/150 official validation scenes"
        )

    val_sample_tokens = {
        row["token"] for row in samples if row["scene_token"] in val_scene_tokens
    }
    by_token = {row["token"]: row for row in sample_data}

    camera_keyframes: set[str] = set()
    lidar_keyframes: set[str] = set()
    lidar_sweeps: set[str] = set()
    for row in sample_data:
        if row["sample_token"] not in val_sample_tokens or not row["is_key_frame"]:
            continue
        filename = PurePosixPath(row["filename"]).as_posix()
        channel = row["filename"].split("/")[1]
        if channel.startswith("CAM_"):
            camera_keyframes.add(filename)
        elif channel == "LIDAR_TOP":
            lidar_keyframes.add(filename)
            previous = row["prev"]
            for _ in range(historical_sweeps):
                if not previous:
                    break
                sweep = by_token[previous]
                lidar_sweeps.add(PurePosixPath(sweep["filename"]).as_posix())
                previous = sweep["prev"]

    expected_samples = len(val_sample_tokens)
    if len(camera_keyframes) != expected_samples * 6:
        raise RuntimeError(
            f"Expected {expected_samples * 6} camera keyframes, found {len(camera_keyframes)}"
        )
    if len(lidar_keyframes) != expected_samples:
        raise RuntimeError(
            f"Expected {expected_samples} LiDAR keyframes, found {len(lidar_keyframes)}"
        )

    return {
        "scene_names": val_names,
        "scene_tokens": val_scene_tokens,
        "sample_tokens": val_sample_tokens,
        "camera_keyframes": camera_keyframes,
        "lidar_keyframes": lidar_keyframes,
        "lidar_sweeps": lidar_sweeps,
    }


def copy_metadata(metadata_root: Path, output_root: Path) -> None:
    output_root.mkdir(parents=True, exist_ok=True)
    for name in ("v1.0-trainval", "maps"):
        source = metadata_root / name
        destination = output_root / name
        if not source.exists():
            raise FileNotFoundError(source)
        if not destination.exists():
            shutil.copytree(source, destination)


def indexed_files(root: Path, top_level_names: tuple[str, ...]) -> set[str]:
    present: set[str] = set()
    for top_level in top_level_names:
        directory = root / top_level
        if not directory.is_dir():
            continue
        for parent, _, filenames in os.walk(directory):
            parent_path = Path(parent)
            for filename in filenames:
                present.add((parent_path / filename).relative_to(root).as_posix())
    return present


def seed_existing_files(seed_root: Path, output_root: Path, required: set[str]) -> int:
    linked = 0
    available = indexed_files(seed_root, ("samples", "sweeps"))
    existing = indexed_files(output_root, ("samples", "sweeps"))
    for relative in sorted(required & available - existing):
        source = seed_root / Path(relative)
        destination = output_root / Path(relative)
        destination.parent.mkdir(parents=True, exist_ok=True)
        try:
            os.link(source, destination)
        except OSError:
            shutil.copy2(source, destination)
        linked += 1
    return linked


def remote_size(url: str) -> int:
    request = urllib.request.Request(url, method="HEAD", headers={"User-Agent": USER_AGENT})
    with urllib.request.urlopen(request, timeout=60) as response:
        return int(response.headers["Content-Length"])


def download_resumable(url: str, destination: Path, retries: int = 30) -> None:
    expected = remote_size(url)
    partial = destination.with_suffix(destination.suffix + ".partial")
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.is_file() and destination.stat().st_size == expected:
        return
    if destination.exists():
        destination.unlink()
    if partial.exists() and partial.stat().st_size > expected:
        partial.unlink()

    for attempt in range(1, retries + 1):
        offset = partial.stat().st_size if partial.exists() else 0
        if offset == expected:
            os.replace(partial, destination)
            return
        headers = {"User-Agent": USER_AGENT, "Range": f"bytes={offset}-"}
        request = urllib.request.Request(url, headers=headers)
        started = time.monotonic()
        last_report = started
        last_offset = offset
        try:
            with urllib.request.urlopen(request, timeout=120) as response, partial.open("ab") as out:
                if offset and response.status != 206:
                    raise RuntimeError(f"Server ignored Range request at byte {offset}")
                while True:
                    chunk = response.read(8 * 1024 * 1024)
                    if not chunk:
                        break
                    out.write(chunk)
                    offset += len(chunk)
                    now = time.monotonic()
                    if now - last_report >= 15:
                        speed = (offset - last_offset) / (now - last_report) / 1024 / 1024
                        print(
                            f"  {offset / expected:6.2%}  {offset:,}/{expected:,}  {speed:5.1f} MiB/s",
                            flush=True,
                        )
                        last_report = now
                        last_offset = offset
        except (OSError, urllib.error.URLError, RuntimeError) as error:
            if attempt == retries:
                raise
            print(f"  retry {attempt}/{retries}: {error}", flush=True)
            time.sleep(min(attempt * 5, 60))
            continue

        if partial.stat().st_size == expected:
            os.replace(partial, destination)
            return
        print(
            f"  retry {attempt}/{retries}: received {partial.stat().st_size:,}/{expected:,}",
            flush=True,
        )
    raise RuntimeError(f"Download did not complete: {url}")


def normalize_member_name(name: str) -> str:
    normalized = PurePosixPath(name.lstrip("./")).as_posix()
    if normalized.startswith("../") or normalized == "..":
        raise RuntimeError(f"Unsafe archive member: {name}")
    return normalized


def extract_selected(archive: Path, output_root: Path, required: set[str]) -> int:
    missing = {name for name in required if not (output_root / Path(name)).is_file()}
    if not missing:
        return 0
    extracted = 0
    with tarfile.open(archive, "r:gz") as tar:
        for member in tar:
            if not member.isfile():
                continue
            relative = normalize_member_name(member.name)
            if relative not in missing:
                continue
            destination = output_root / Path(relative)
            destination.parent.mkdir(parents=True, exist_ok=True)
            temporary = destination.with_suffix(destination.suffix + ".partial")
            source = tar.extractfile(member)
            if source is None:
                raise RuntimeError(f"Unable to read archive member: {member.name}")
            with source, temporary.open("wb") as out:
                shutil.copyfileobj(source, out, length=8 * 1024 * 1024)
            if temporary.stat().st_size != member.size:
                raise RuntimeError(f"Incomplete extraction: {member.name}")
            os.replace(temporary, destination)
            missing.remove(relative)
            extracted += 1
    return extracted


def extract_selected_fast(archive: Path, output_root: Path, required: set[str]) -> int:
    """Use native tar for batched extraction on mounted Windows filesystems."""
    present = indexed_files(output_root, ("samples", "sweeps"))
    missing = required - present
    if not missing:
        return 0
    listing = subprocess.run(
        ["tar", "-tzf", str(archive)],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.splitlines()
    selected = [name for name in listing if normalize_member_name(name) in missing]
    if not selected:
        return 0
    member_list = archive.with_suffix(archive.suffix + ".members.txt")
    member_list.write_text("\n".join(selected) + "\n", encoding="utf-8")
    try:
        subprocess.run(
            ["tar", "-xzf", str(archive), "-C", str(output_root), "-T", str(member_list)],
            check=True,
        )
    finally:
        member_list.unlink(missing_ok=True)
    return len(selected)


def validate_files(output_root: Path, required: set[str]) -> tuple[int, list[str]]:
    present = indexed_files(output_root, ("samples", "sweeps"))
    missing = sorted(required - present)
    return len(required) - len(missing), missing


def process_archives(
    kind: str,
    output_root: Path,
    downloads_root: Path,
    required: set[str],
    start: int,
    end: int,
    keep_archives: bool,
) -> None:
    suffix = "keyframes" if kind == "keyframes" else "blobs_lidar"
    for index in range(start, end + 1):
        filename = f"v1.0-trainval{index:02d}_{suffix}.tgz"
        archive = downloads_root / filename
        url = f"{AWS_BASE}/{filename}"
        present, missing = validate_files(output_root, required)
        if not missing:
            print(f"All {present:,} required {kind} files are already present.")
            return
        print(f"[{index:02d}/10] {kind}: {len(missing):,} files still missing", flush=True)
        download_resumable(url, archive)
        extracted = extract_selected_fast(archive, output_root, required)
        print(f"  extracted {extracted:,} selected files", flush=True)
        if not keep_archives:
            archive.unlink(missing_ok=True)


def write_manifest(output_root: Path, requirements: dict[str, set[str]], sweeps: int) -> None:
    keyframes = requirements["camera_keyframes"] | requirements["lidar_keyframes"]
    keyframe_present, keyframe_missing = validate_files(output_root, keyframes)
    sweep_present, sweep_missing = validate_files(output_root, requirements["lidar_sweeps"])
    report = {
        "split": "val",
        "scenes": len(requirements["scene_tokens"]),
        "samples": len(requirements["sample_tokens"]),
        "camera_keyframes_required": len(requirements["camera_keyframes"]),
        "lidar_keyframes_required": len(requirements["lidar_keyframes"]),
        "keyframes_present": keyframe_present,
        "keyframes_missing": len(keyframe_missing),
        "historical_sweeps_per_keyframe": sweeps,
        "unique_lidar_sweeps_required": len(requirements["lidar_sweeps"]),
        "lidar_sweeps_present": sweep_present,
        "lidar_sweeps_missing": len(sweep_missing),
        "complete_for_camera_keyframe_models": not keyframe_missing,
        "complete_for_lidar_10_frame_models": not keyframe_missing and not sweep_missing,
        "missing_examples": (keyframe_missing + sweep_missing)[:20],
    }
    path = output_root / "val150_materialization.json"
    with path.open("w", encoding="utf-8") as destination:
        json.dump(report, destination, indent=2, ensure_ascii=False)
    print(json.dumps(report, indent=2, ensure_ascii=False))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--metadata-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--downloads-root", type=Path, required=True)
    parser.add_argument("--seed-root", type=Path)
    parser.add_argument(
        "--kind", choices=("keyframes", "lidar-sweeps", "all", "validate"), default="all"
    )
    parser.add_argument("--historical-sweeps", type=int, default=10)
    parser.add_argument("--archive-start", type=int, default=1)
    parser.add_argument("--archive-end", type=int, default=10)
    parser.add_argument("--keep-archives", action="store_true")
    args = parser.parse_args()

    requirements = build_requirements(args.metadata_root, args.historical_sweeps)
    copy_metadata(args.metadata_root, args.output_root)
    keyframes = requirements["camera_keyframes"] | requirements["lidar_keyframes"]
    if args.seed_root:
        linked = seed_existing_files(
            args.seed_root,
            args.output_root,
            keyframes | requirements["lidar_sweeps"],
        )
        print(f"Seeded {linked:,} existing files from {args.seed_root}")

    if args.kind in ("keyframes", "all"):
        process_archives(
            "keyframes",
            args.output_root,
            args.downloads_root,
            keyframes,
            args.archive_start,
            args.archive_end,
            args.keep_archives,
        )
    if args.kind in ("lidar-sweeps", "all"):
        process_archives(
            "lidar-sweeps",
            args.output_root,
            args.downloads_root,
            requirements["lidar_sweeps"],
            args.archive_start,
            args.archive_end,
            args.keep_archives,
        )
    write_manifest(args.output_root, requirements, args.historical_sweeps)


if __name__ == "__main__":
    main()
