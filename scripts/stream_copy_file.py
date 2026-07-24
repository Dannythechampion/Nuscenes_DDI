"""Copy a large file sequentially with resumable progress and atomic replacement."""

from __future__ import annotations

import argparse
import os
import time
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", required=True, type=Path)
    parser.add_argument("--destination", required=True, type=Path)
    parser.add_argument("--expected-bytes", required=True, type=int)
    parser.add_argument("--chunk-mib", type=int, default=8)
    args = parser.parse_args()

    source = args.source.resolve()
    destination = args.destination.resolve()
    partial = destination.with_name(destination.name + ".partial")

    if not source.is_file():
        raise SystemExit(f"Source does not exist: {source}")
    if source.stat().st_size != args.expected_bytes:
        raise SystemExit(
            f"Unexpected source size: {source.stat().st_size}; "
            f"expected {args.expected_bytes}"
        )

    destination.parent.mkdir(parents=True, exist_ok=True)
    offset = partial.stat().st_size if partial.exists() else 0
    if offset > args.expected_bytes:
        raise SystemExit(f"Partial file is too large: {offset} bytes")

    chunk_size = args.chunk_mib * 1024 * 1024
    started = time.monotonic()
    last_report = started
    last_offset = offset

    print(
        f"Streaming {source} -> {partial} from {offset:,}/{args.expected_bytes:,} bytes",
        flush=True,
    )

    with source.open("rb", buffering=0) as source_file, partial.open("ab", buffering=0) as output:
        source_file.seek(offset)
        while offset < args.expected_bytes:
            chunk = source_file.read(min(chunk_size, args.expected_bytes - offset))
            if not chunk:
                raise SystemExit(f"Source ended early at {offset:,} bytes")
            output.write(chunk)
            offset += len(chunk)

            now = time.monotonic()
            if now - last_report >= 10 or offset == args.expected_bytes:
                interval_mib_s = (offset - last_offset) / (now - last_report) / 1024 / 1024
                percent = offset / args.expected_bytes * 100
                print(
                    f"{percent:6.2f}%  {offset:,}/{args.expected_bytes:,} bytes  "
                    f"{interval_mib_s:5.1f} MiB/s",
                    flush=True,
                )
                last_report = now
                last_offset = offset

        output.flush()
        os.fsync(output.fileno())

    if partial.stat().st_size != args.expected_bytes:
        raise SystemExit(f"Incomplete partial file: {partial.stat().st_size} bytes")

    os.replace(partial, destination)
    elapsed = time.monotonic() - started
    print(f"Copy complete in {elapsed / 60:.1f} minutes: {destination}", flush=True)


if __name__ == "__main__":
    main()
