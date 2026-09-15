"""Reproduce the accepted runtime and fetch or verify pinned external artifacts.

All downloads are opt-in. Verification never needs a GPU, model import, private
service, user cache, or simulator. Missing bytes remain failures in the receipt.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import platform
from pathlib import Path
import shutil
import sys
import tempfile


RUNTIME = {
    "torch": "2.8.0",
    "torchvision": "0.23.0",
    "Pillow": "11.3.0",
    "numpy": "2.2.6",
    "transformers": "5.5.4",
    "lerobot": "0.6.2",
    "hf-libero": "0.1.4",
    "mujoco": "3.8.1",
    "robosuite": "1.4.0",
}


def digest(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def contained(root, relative):
    root = Path(root).resolve()
    path = (root / relative).resolve()
    if not relative or Path(relative).is_absolute() or not path.is_relative_to(root):
        raise ValueError("Artifact path must remain inside the selected root")
    return path


def environment(*, require_simulator=True, require_cuda=False):
    packages, failures = {}, []
    if sys.version_info[:2] != (3, 12):
        failures.append("python: requires 3.12")
    required = dict(RUNTIME)
    if not require_simulator:
        for name in ("hf-libero", "mujoco", "robosuite"):
            required.pop(name)
    for name, expected in required.items():
        try:
            actual = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            actual = None
        packages[name] = actual
        if actual is None or actual.split("+")[0] != expected:
            failures.append(f"{name}: expected {expected}, found {actual}")
    gpu = None
    if require_cuda:
        import torch

        gpu = dict(available=torch.cuda.is_available(), build_cuda=torch.version.cuda)
        if not gpu["available"] or gpu["build_cuda"] != "12.8":
            failures.append("CUDA 12.8 runtime and accessible GPU required")
        else:
            gpu["name"] = torch.cuda.get_device_name(0)
            gpu["total_memory_bytes"] = torch.cuda.get_device_properties(0).total_memory
    return dict(
        status="PASS" if not failures else "FAIL",
        python=sys.version,
        platform=platform.platform(),
        packages=packages,
        gpu=gpu,
        errors=failures,
    )


def entries(manifest):
    """Accept runtime manifests and the existing external-blob hash dictionary."""
    if "files" in manifest:
        rows = manifest["files"]
    else:
        rows = [
            dict(destination=name, size=row["bytes"], **row)
            for name, row in manifest.items()
        ]
    if not rows:
        raise ValueError("An empty manifest cannot establish delivery")
    names = [row["destination"] for row in rows]
    if len(names) != len(set(names)):
        raise ValueError("Duplicate artifact destinations")
    for row in rows:
        if len(row["sha256"]) != 64 or any(
            c not in "0123456789abcdef" for c in row["sha256"]
        ):
            raise ValueError("Expected a SHA-256 identity")
        if (
            type(row.get("size", row.get("bytes"))) is not int
            or row.get("size", row.get("bytes")) < 0
        ):
            raise ValueError("Expected a nonnegative exact byte size")
    return rows


def verify(manifest, root):
    rows = []
    for entry in entries(manifest):
        path = contained(root, entry["destination"])
        size = entry.get("size", entry.get("bytes"))
        actual_size = path.stat().st_size if path.is_file() else None
        actual_hash = digest(path) if actual_size == size else None
        rows.append(
            dict(
                path=entry["destination"],
                expected_bytes=size,
                actual_bytes=actual_size,
                expected_sha256=entry["sha256"],
                actual_sha256=actual_hash,
                passed=actual_hash == entry["sha256"],
            )
        )
    return dict(
        status="PASS" if all(r["passed"] for r in rows) else "FAIL",
        expected=len(rows),
        verified=sum(r["passed"] for r in rows),
        files=rows,
    )


def materialize(manifest, root, cache):
    from huggingface_hub import hf_hub_download

    root, cache = Path(root), Path(cache)
    # Check the entire manifest before downloading or modifying anything.
    rows = entries(manifest)
    for row in rows:
        contained(root, row["destination"])
        if not row.get("repo", manifest.get("repo")) or not row.get("file"):
            raise ValueError(
                "External experiment blobs require an explicit owner-provided copy"
            )
        revision = row.get("revision", manifest.get("revision", ""))
        if len(revision) != 40 or any(c not in "0123456789abcdef" for c in revision):
            raise ValueError("Downloads require an immutable full upstream revision")
    for row in rows:
        path = contained(root, row["destination"])
        if (
            path.is_file()
            and path.stat().st_size == row["size"]
            and digest(path) == row["sha256"]
        ):
            continue
        downloaded = Path(
            hf_hub_download(
                repo_id=row.get("repo", manifest.get("repo")),
                repo_type=row.get("repo_type", manifest.get("repo_type", "model")),
                revision=row.get("revision", manifest.get("revision")),
                filename=row["file"],
                cache_dir=cache,
            )
        )
        if (
            downloaded.stat().st_size != row["size"]
            or digest(downloaded) != row["sha256"]
        ):
            raise ValueError(
                "Downloaded artifact differs from pinned content: " + row["destination"]
            )
        path.parent.mkdir(parents=True, exist_ok=True)
        # Stage before atomic replacement, including cross-filesystem caches.
        temporary = None
        try:
            with tempfile.NamedTemporaryFile(
                dir=path.parent, prefix=path.name + ".", delete=False
            ) as output:
                temporary = Path(output.name)
                with downloaded.open("rb") as source:
                    shutil.copyfileobj(source, output)
            temporary.replace(path)
        finally:
            if temporary is not None:
                temporary.unlink(missing_ok=True)
    return verify(manifest, root)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    doctor = commands.add_parser("doctor")
    doctor.add_argument("--software-only", action="store_true")
    doctor.add_argument("--cuda", action="store_true")
    doctor.add_argument("--output", type=Path)
    for name in ("verify", "fetch"):
        sub = commands.add_parser(name)
        sub.add_argument("--manifest", type=Path, required=True)
        sub.add_argument("--root", type=Path, required=True)
        sub.add_argument("--output", type=Path)
        if name == "fetch":
            sub.add_argument("--cache", type=Path, required=True)
    args = parser.parse_args()
    try:
        if args.command == "doctor":
            receipt = environment(
                require_simulator=not args.software_only, require_cuda=args.cuda
            )
        else:
            manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
            receipt = (
                materialize(manifest, args.root, args.cache)
                if args.command == "fetch"
                else verify(manifest, args.root)
            )
            receipt["manifest_sha256"] = digest(args.manifest)
    except Exception as exc:
        receipt = dict(status="FAIL", error_type=type(exc).__name__, error=str(exc))
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(receipt, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(receipt, indent=2))
    return 0 if receipt["status"] == "PASS" else 2


if __name__ == "__main__":
    raise SystemExit(main())
