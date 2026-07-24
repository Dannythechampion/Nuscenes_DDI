"""Resume fixed byte-range parts with validation and bounded parallelism."""

from __future__ import annotations

import argparse
import concurrent.futures
import math
import time
from pathlib import Path

import requests


def download_part(
    url: str,
    path: Path,
    segment_start: int,
    segment_end: int,
    retries: int,
) -> dict:
    expected_size = segment_end - segment_start + 1
    if path.exists() and path.stat().st_size > expected_size:
        raise ValueError(f"{path} is larger than its assigned byte range.")

    for attempt in range(1, retries + 1):
        current_size = path.stat().st_size if path.exists() else 0
        if current_size == expected_size:
            return {"path": str(path), "bytes": current_size, "complete": True}

        request_start = segment_start + current_size
        headers = {
            "Range": f"bytes={request_start}-{segment_end}",
            "User-Agent": "nuScenes-DDI-research-downloader/1.0",
        }
        try:
            with requests.get(url, headers=headers, stream=True, timeout=(30, 60)) as response:
                if response.status != 206:
                    raise RuntimeError(f"Expected HTTP 206 for {path.name}, got {response.status}.")
                content_range = response.headers.get("Content-Range", "")
                if not content_range.startswith(f"bytes {request_start}-{segment_end}/"):
                    raise RuntimeError(f"Unexpected Content-Range for {path.name}: {content_range}")

                path.parent.mkdir(parents=True, exist_ok=True)
                with path.open("ab") as output:
                    for chunk in response.iter_content(chunk_size=1024 * 1024):
                        if not chunk:
                            continue
                        output.write(chunk)
        except Exception as error:
            if attempt == retries:
                raise
            print(f"{path.name}: retry {attempt}/{retries} after {error}", flush=True)
            time.sleep(min(10 * attempt, 60))

    final_size = path.stat().st_size
    return {"path": str(path), "bytes": final_size, "complete": final_size == expected_size}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--url", required=True)
    parser.add_argument("--parts-dir", required=True)
    parser.add_argument("--base-offset", type=int, required=True)
    parser.add_argument("--total-bytes", type=int, required=True)
    parser.add_argument("--part-count", type=int, default=4)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--retries", type=int, default=20)
    args = parser.parse_args()

    parts_dir = Path(args.parts_dir)
    remaining = args.total_bytes - args.base_offset
    segment_size = math.ceil(remaining / args.part_count)
    tasks = []
    for index in range(args.part_count):
        start = args.base_offset + index * segment_size
        end = min(args.total_bytes - 1, start + segment_size - 1)
        path = parts_dir / f"part-{index + 1:02d}.bin"
        tasks.append((args.url, path, start, end, args.retries))
        print(
            f"{path.name}: range={start}-{end}, existing={path.stat().st_size if path.exists() else 0}",
            flush=True,
        )

    results = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=args.workers) as executor:
        futures = [executor.submit(download_part, *task) for task in tasks]
        for future in concurrent.futures.as_completed(futures):
            result = future.result()
            results.append(result)
            print(result, flush=True)

    if not all(result["complete"] for result in results):
        raise SystemExit("At least one byte-range part is incomplete.")


if __name__ == "__main__":
    main()
