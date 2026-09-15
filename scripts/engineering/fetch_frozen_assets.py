"""Fetch an explicit, hash-pinned file manifest with resumable HTTP ranges."""

import concurrent.futures
import hashlib
import json
from pathlib import Path
import sys
import time

import requests


def digest(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def fetch(entry, root):
    path = root / entry["destination"]
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists() and digest(path) == entry["sha256"]:
        return
    size = entry["size"]
    block = 16 * 1024**2
    part_dir = path.parent / (path.name + ".parts")
    part_dir.mkdir(exist_ok=True)

    def part(index):
        start = index * block
        end = min(size - 1, start + block - 1)
        target = part_dir / str(index)
        if target.exists() and target.stat().st_size == end - start + 1:
            return target
        for attempt in range(8):
            try:
                session = requests.Session()
                session.trust_env = False
                url = f"https://hf-mirror.com/{entry['repo']}/resolve/{entry['revision']}/{entry['file']}?part={index}"
                response = session.get(
                    url, headers={"Range": f"bytes={start}-{end}"}, timeout=90
                )
                response.raise_for_status()
                data = response.content
                if len(data) != end - start + 1:
                    raise ValueError("Wrong range length")
                if (
                    size > block
                    and response.headers.get("Content-Range")
                    != f"bytes {start}-{end}/{size}"
                ):
                    raise ValueError("Wrong Content-Range")
                target.write_bytes(data)
                if index % 16 == 0:
                    print(
                        json.dumps(
                            {"file": entry["destination"], "finished_part": index}
                        ),
                        flush=True,
                    )
                return target
            except Exception:
                if attempt == 7:
                    raise
                time.sleep(min(30, 2**attempt))

    with concurrent.futures.ThreadPoolExecutor(max_workers=16) as pool:
        parts = list(pool.map(part, range((size + block - 1) // block)))
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("wb") as stream:
        for piece in parts:
            stream.write(piece.read_bytes())
    if digest(temporary) != entry["sha256"]:
        raise ValueError(f"Hash mismatch: {entry['destination']}")
    temporary.replace(path)
    print(
        json.dumps(
            {
                "downloaded": entry["destination"],
                "bytes": size,
                "sha256": entry["sha256"],
            }
        ),
        flush=True,
    )


if __name__ == "__main__":
    manifest = json.loads(Path(sys.argv[1]).read_text())
    root = Path(sys.argv[2])
    for entry in manifest["files"]:
        fetch(entry, root)
    print("ALL_FROZEN_ASSETS_VERIFIED", flush=True)
