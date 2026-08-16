#!/usr/bin/env python3
"""Small, dependency-free helpers for the fail-closed Linux M8 runner.

The native runner intentionally keeps orchestration in Bash while delegating
JSON parsing/writing and path containment checks to Python.  This module never
imports Isaac Sim or ROS and never creates benchmark result artifacts.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
from pathlib import Path, PurePosixPath, PureWindowsPath
import re
import shutil
import subprocess
import sys
from typing import Any, Iterable, Mapping, Sequence


MILESTONE = "M8-G0"
SCHEMA_VERSION = 1
STRATEGIES = {"sync_hold", "naive_async", "aligned_async"}
LINUX_RUNNER_SOURCE = "scripts/m8_run_isaac.sh"
LINUX_RUNNER_SUPPORT_SOURCE = "scripts/m8_linux_runner_support.py"
ISAAC_WORKSPACE_REPOSITORY_URL = (
    "https://github.com/isaac-sim/IsaacSim-ros_workspaces.git"
)
ISAAC_WORKSPACE_COMMIT = "dd3eeede7912755996a18f4884285d9f50843f79"
ISAAC_WORKSPACE_RELATIVE_PATH = "jazzy_ws"
OFFICIAL_LINUX_PIXI_VERSION = "0.75.0"
OFFICIAL_LINUX_PIXI_VERSION_OUTPUT = f"pixi {OFFICIAL_LINUX_PIXI_VERSION}"
OFFICIAL_LINUX_PIXI_EXECUTABLE_SIZE_BYTES = 77_311_024
OFFICIAL_LINUX_PIXI_EXECUTABLE_SHA256 = (
    "4383aed18b2d5569cf34a19638daf954aa4415cc87ad3a9da9f34059cc4a004c"
)
OFFICIAL_WORKSPACE_FILE_SPECS = {
    "pixi.toml": {
        "canonical_lf_size_bytes": 5_467,
        "canonical_lf_sha256": (
            "b4e7a34c264e88f19ba6bfb3c7a72ee46b0843f3e6eb7e75619dc0ebb87b313d"
        ),
        "exact_crlf_size_bytes": 5_618,
        "exact_crlf_sha256": (
            "9649bf57644781a1fe0203ed6b80828ccb42ea07555475080e5d11a9b0c3e1ae"
        ),
        "evidence_name": "isaac_workspace.pixi.toml",
    },
    "pixi.lock": {
        "canonical_lf_size_bytes": 1_491_808,
        "canonical_lf_sha256": (
            "ba8e59eef962cbf49a1ff06ff947ed5eaa4547d018389b048771a1e7d8bb890d"
        ),
        "exact_crlf_size_bytes": 1_533_045,
        "exact_crlf_sha256": (
            "2c2f9097b129847735b5abb805045a22076a731e2138c8192caac95d09a2866e"
        ),
        "evidence_name": "isaac_workspace.pixi.lock",
    },
}
EXPECTED_RUNTIME_VERSIONS = {
    "python_version": "3.12.13",
    "isaacsim": "6.0.1.0",
    "isaacsim-app": "6.0.1.0",
    "isaacsim-core": "6.0.1.0",
    "isaacsim-robot": "6.0.1.0",
    "isaacsim-ros2": "6.0.1.0",
    "rclpy": "7.1.9",
    "rosgraph-msgs": "2.0.3",
    "ros_distribution": "jazzy",
    "rmw_implementation": "rmw_zenoh_cpp",
    "rmw_zenoh_cpp": "0.2.9",
}
EXCLUDED_SOURCE_DIRECTORIES = {
    "__pycache__",
    ".pytest_cache",
    ".ruff_cache",
    "build",
    "install",
    "log",
}


def _read_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _utc_now() -> str:
    from datetime import datetime, timezone

    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _write_json_exclusive(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8", newline="\n") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True, ensure_ascii=False)
        handle.write("\n")


def _contained(path: Path, base: Path, *, description: str) -> Path:
    resolved = path.resolve(strict=True)
    resolved_base = base.resolve(strict=True)
    try:
        resolved.relative_to(resolved_base)
    except ValueError as exc:
        raise ValueError(f"{description} escapes {resolved_base}: {resolved}") from exc
    return resolved


def _contained_output(path: Path, base: Path, *, description: str) -> Path:
    resolved = path.resolve(strict=False)
    resolved_base = base.resolve(strict=True)
    try:
        resolved.relative_to(resolved_base)
    except ValueError as exc:
        raise ValueError(f"{description} escapes {resolved_base}: {resolved}") from exc
    return resolved


def _portable_relative(
    value: Any, *, description: str, allow_parent: bool = False
) -> Path:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{description} must be a non-empty relative path")
    posix_candidate = PurePosixPath(value)
    windows_candidate = PureWindowsPath(value)
    if (
        "\\" in value
        or posix_candidate.is_absolute()
        or windows_candidate.is_absolute()
        or bool(windows_candidate.drive)
        or (not allow_parent and ".." in posix_candidate.parts)
        or (not allow_parent and ".." in windows_candidate.parts)
    ):
        raise ValueError(f"{description} is not an allowed portable relative path: {value}")
    return Path(*posix_candidate.parts)


def _relative(path: Path, base: Path, *, description: str) -> str:
    resolved = _contained(path, base, description=description)
    return resolved.relative_to(base.resolve(strict=True)).as_posix()


def _emit_nul(values: Iterable[str]) -> None:
    encoded = b"\0".join(value.encode("utf-8") for value in values) + b"\0"
    sys.stdout.buffer.write(encoded)


def _run_text(arguments: Sequence[str], *, description: str, cwd: Path | None = None) -> str:
    completed = subprocess.run(
        list(arguments),
        cwd=str(cwd) if cwd is not None else None,
        check=False,
        capture_output=True,
        text=True,
        timeout=120,
    )
    if completed.returncode != 0:
        detail = completed.stderr.strip() or completed.stdout.strip() or "no diagnostic"
        raise RuntimeError(
            f"{description} failed with exit code {completed.returncode}: {detail}"
        )
    return completed.stdout.strip()


def _official_workspace_file_record(path: Path, *, filename: str) -> dict[str, Any]:
    spec = OFFICIAL_WORKSPACE_FILE_SPECS[filename]
    data = path.read_bytes()
    digest = hashlib.sha256(data).hexdigest()
    size = len(data)
    if (
        size == spec["canonical_lf_size_bytes"]
        and digest == spec["canonical_lf_sha256"]
    ):
        byte_form = "lf"
    elif size == spec["exact_crlf_size_bytes"] and digest == spec["exact_crlf_sha256"]:
        if b"\r" in data.replace(b"\r\n", b""):
            raise ValueError(f"official {filename} contains a bare carriage return")
        normalized = data.replace(b"\r\n", b"\n")
        if (
            len(normalized) != spec["canonical_lf_size_bytes"]
            or hashlib.sha256(normalized).hexdigest()
            != spec["canonical_lf_sha256"]
        ):
            raise ValueError(
                f"official {filename} CRLF bytes do not normalize to the canonical Git blob"
            )
        byte_form = "crlf"
    else:
        raise ValueError(
            f"official {filename} bytes are not the exact LF Git blob or its exact "
            f"all-CRLF checkout form: size={size}, sha256={digest}"
        )
    return {
        "source_path": str(path.resolve(strict=True)),
        "size_bytes": size,
        "sha256": digest,
        "canonical_lf_size_bytes": spec["canonical_lf_size_bytes"],
        "canonical_lf_sha256": spec["canonical_lf_sha256"],
        "byte_form": byte_form,
    }


def _require_matching_workspace_byte_forms(
    records: Mapping[str, Mapping[str, Any]],
) -> None:
    byte_forms = {
        records[filename].get("byte_form") for filename in OFFICIAL_WORKSPACE_FILE_SPECS
    }
    if len(byte_forms) != 1:
        raise ValueError(
            "official pixi.toml and pixi.lock must use the same exact newline form"
        )


def _external_workspace_state(
    *, isaac_workspace: Path, pixi_executable: Path
) -> dict[str, Any]:
    workspace = isaac_workspace.resolve(strict=True)
    if not workspace.is_dir():
        raise ValueError(f"Isaac workspace is not a directory: {workspace}")
    pixi = pixi_executable.resolve(strict=True)
    if not pixi.is_file() or not os.access(pixi, os.X_OK):
        raise ValueError(f"official Pixi executable is missing or not executable: {pixi}")
    if pixi.stat().st_size != OFFICIAL_LINUX_PIXI_EXECUTABLE_SIZE_BYTES:
        raise ValueError("Pixi executable size does not match official Linux v0.75.0")
    pixi_hash = _sha256(pixi)
    if pixi_hash != OFFICIAL_LINUX_PIXI_EXECUTABLE_SHA256:
        raise ValueError("Pixi executable hash does not match official Linux v0.75.0")
    version_output = _run_text((str(pixi), "--version"), description="Pixi version probe")
    if version_output != OFFICIAL_LINUX_PIXI_VERSION_OUTPUT:
        raise ValueError(
            "Pixi version output does not match the pinned formal runtime: "
            f"{version_output!r}"
        )

    repository_root_text = _run_text(
        ("git", "-C", str(workspace), "rev-parse", "--show-toplevel"),
        description="Isaac workspace Git-root probe",
    )
    repository_root = Path(repository_root_text).resolve(strict=True)
    try:
        workspace_relative = workspace.relative_to(repository_root).as_posix()
    except ValueError as exc:
        raise ValueError("Isaac workspace escapes its Git repository") from exc
    if workspace_relative != ISAAC_WORKSPACE_RELATIVE_PATH:
        raise ValueError(
            "Isaac workspace is not the exact official jazzy_ws repository directory"
        )
    commit = _run_text(
        ("git", "-C", str(repository_root), "rev-parse", "HEAD"),
        description="Isaac workspace commit probe",
    )
    if commit != ISAAC_WORKSPACE_COMMIT:
        raise ValueError(f"Isaac workspace commit is not pinned: {commit}")
    repository_url = _run_text(
        ("git", "-C", str(repository_root), "remote", "get-url", "origin"),
        description="Isaac workspace origin probe",
    )
    if repository_url != ISAAC_WORKSPACE_REPOSITORY_URL:
        raise ValueError(f"Isaac workspace origin is not canonical: {repository_url}")

    tracked_relative_paths = [
        f"{ISAAC_WORKSPACE_RELATIVE_PATH}/{filename}"
        for filename in OFFICIAL_WORKSPACE_FILE_SPECS
    ]
    _run_text(
        (
            "git",
            "-C",
            str(repository_root),
            "ls-files",
            "--error-unmatch",
            "--",
            *tracked_relative_paths,
        ),
        description="Isaac workspace tracked manifest/lock probe",
    )
    status = _run_text(
        (
            "git",
            "-C",
            str(repository_root),
            "status",
            "--porcelain=v1",
            "--untracked-files=no",
            "--",
            *tracked_relative_paths,
        ),
        description="Isaac workspace tracked manifest/lock cleanliness probe",
    )
    if status:
        raise ValueError("Isaac workspace pixi.toml/pixi.lock have tracked changes")

    files = {
        filename: _official_workspace_file_record(workspace / filename, filename=filename)
        for filename in OFFICIAL_WORKSPACE_FILE_SPECS
    }
    _require_matching_workspace_byte_forms(files)
    return {
        "repository_root": str(repository_root),
        "repository_url": repository_url,
        "workspace_commit": commit,
        "workspace_relative_path": workspace_relative,
        "tracked_manifest_lock_clean": True,
        "pixi_executable": str(pixi),
        "pixi_executable_size_bytes": pixi.stat().st_size,
        "pixi_executable_sha256": pixi_hash,
        "pixi_version": OFFICIAL_LINUX_PIXI_VERSION,
        "pixi_version_output": version_output,
        "files": files,
    }


def _validate_runtime_payload(payload: Any) -> dict[str, Any]:
    if not isinstance(payload, Mapping):
        raise ValueError("Pixi runtime probe did not return an object")
    expected_keys = {
        "platform_system",
        "platform_machine",
        "python_version",
        "python_executable",
        "sys_prefix",
        "packages",
        "ros_distribution",
        "rmw_implementation",
        "rmw_zenoh_cpp",
    }
    if set(payload) != expected_keys:
        raise ValueError("Pixi runtime probe fields are incomplete or unexpected")
    if payload.get("platform_system") != "Linux" or payload.get("platform_machine") not in {
        "x86_64",
        "AMD64",
    }:
        raise ValueError("formal native runtime must be Linux x86-64")
    packages = payload.get("packages")
    if not isinstance(packages, Mapping):
        raise ValueError("Pixi runtime package versions are malformed")
    for name in (
        "isaacsim",
        "isaacsim-app",
        "isaacsim-core",
        "isaacsim-robot",
        "isaacsim-ros2",
        "rclpy",
        "rosgraph-msgs",
    ):
        if packages.get(name) != EXPECTED_RUNTIME_VERSIONS[name]:
            raise ValueError(
                f"Pixi runtime {name} version is not pinned: {packages.get(name)!r}"
            )
    for name in (
        "python_version",
        "ros_distribution",
        "rmw_implementation",
        "rmw_zenoh_cpp",
    ):
        if payload.get(name) != EXPECTED_RUNTIME_VERSIONS[name]:
            raise ValueError(f"Pixi runtime {name} is not pinned: {payload.get(name)!r}")
    for name in ("python_executable", "sys_prefix"):
        if not isinstance(payload.get(name), str) or not Path(payload[name]).is_absolute():
            raise ValueError(f"Pixi runtime {name} must be an informational absolute path")
    return dict(payload)


def _runtime_probe(_args: argparse.Namespace) -> int:
    import importlib.metadata
    import platform

    conda_meta = Path(sys.prefix) / "conda-meta"
    rmw_records = sorted(conda_meta.glob("ros-jazzy-rmw-zenoh-cpp-*.json"))
    if len(rmw_records) != 1:
        raise ValueError("runtime must contain exactly one rmw_zenoh_cpp conda record")
    rmw_record = _read_json(rmw_records[0])
    if not isinstance(rmw_record, Mapping) or rmw_record.get("name") != (
        "ros-jazzy-rmw-zenoh-cpp"
    ):
        raise ValueError("rmw_zenoh_cpp conda record is malformed")
    package_names = (
        "isaacsim",
        "isaacsim-app",
        "isaacsim-core",
        "isaacsim-robot",
        "isaacsim-ros2",
        "rclpy",
        "rosgraph-msgs",
    )
    payload = {
        "platform_system": platform.system(),
        "platform_machine": platform.machine(),
        "python_version": platform.python_version(),
        "python_executable": str(Path(sys.executable).resolve()),
        "sys_prefix": str(Path(sys.prefix).resolve()),
        "packages": {
            name: importlib.metadata.version(name) for name in package_names
        },
        "ros_distribution": os.environ.get("ROS_DISTRO"),
        "rmw_implementation": os.environ.get("RMW_IMPLEMENTATION"),
        "rmw_zenoh_cpp": rmw_record.get("version"),
    }
    print(json.dumps(payload, sort_keys=True, separators=(",", ":")))
    return 0


def _validate_external_environment(args: argparse.Namespace) -> int:
    _external_workspace_state(
        isaac_workspace=Path(args.isaac_workspace),
        pixi_executable=Path(args.pixi_exe),
    )
    return 0


def _write_external_environment(args: argparse.Namespace) -> int:
    output = Path(args.output).resolve(strict=False)
    receipt_dir = output.parent.resolve(strict=True)
    if output.exists():
        raise FileExistsError(f"refusing to overwrite external environment evidence: {output}")
    before = _external_workspace_state(
        isaac_workspace=Path(args.isaac_workspace),
        pixi_executable=Path(args.pixi_exe),
    )
    pixi = before["pixi_executable"]
    manifest = Path(args.isaac_workspace).resolve(strict=True) / "pixi.toml"
    completed = subprocess.run(
        [
            pixi,
            "run",
            "--frozen",
            "--manifest-path",
            str(manifest),
            "--",
            "python",
            str(Path(__file__).resolve(strict=True)),
            "runtime-probe",
        ],
        check=False,
        capture_output=True,
        text=True,
        timeout=300,
    )
    if completed.returncode != 0:
        detail = completed.stderr.strip() or completed.stdout.strip() or "no diagnostic"
        raise RuntimeError(
            "frozen Pixi runtime probe failed with exit code "
            f"{completed.returncode}: {detail}"
        )
    try:
        runtime = _validate_runtime_payload(json.loads(completed.stdout))
    except json.JSONDecodeError as exc:
        raise ValueError("frozen Pixi runtime probe emitted invalid JSON") from exc
    after = _external_workspace_state(
        isaac_workspace=Path(args.isaac_workspace),
        pixi_executable=Path(args.pixi_exe),
    )
    if before != after:
        raise ValueError("external Isaac/Pixi environment changed during frozen runtime probe")

    evidence_records: dict[str, Any] = {}
    for filename, spec in OFFICIAL_WORKSPACE_FILE_SPECS.items():
        source = Path(args.isaac_workspace).resolve(strict=True) / filename
        destination = receipt_dir / str(spec["evidence_name"])
        if destination.exists():
            raise FileExistsError(f"refusing to overwrite workspace evidence: {destination}")
        with source.open("rb") as reader, destination.open("xb") as writer:
            shutil.copyfileobj(reader, writer, length=1024 * 1024)
            writer.flush()
            os.fsync(writer.fileno())
        record = dict(after["files"][filename])
        if _sha256(destination) != record["sha256"]:
            raise ValueError(f"copied {filename} evidence hash mismatch")
        record["evidence"] = destination.relative_to(receipt_dir).as_posix()
        evidence_records[filename] = record

    payload = {
        "schema_version": SCHEMA_VERSION,
        "milestone": MILESTONE,
        "evidence_kind": "native_external_environment",
        "created_utc": _utc_now(),
        "repository_root": after["repository_root"],
        "repository_url": after["repository_url"],
        "workspace_commit": after["workspace_commit"],
        "workspace_relative_path": after["workspace_relative_path"],
        "tracked_manifest_lock_clean": after["tracked_manifest_lock_clean"],
        "workspace_files": evidence_records,
        "pixi": {
            "executable": after["pixi_executable"],
            "executable_size_bytes": after["pixi_executable_size_bytes"],
            "executable_sha256": after["pixi_executable_sha256"],
            "version": after["pixi_version"],
            "version_output": after["pixi_version_output"],
        },
        "runtime": runtime,
    }
    _write_json_exclusive(output, payload)
    return 0


def _suite_records(args: argparse.Namespace) -> int:
    root = Path(args.repository_root).resolve(strict=True)
    suite_path = _contained(Path(args.suite), root, description="matrix suite")
    suite = _read_json(suite_path)
    if not isinstance(suite, Mapping):
        raise ValueError("matrix suite root must be an object")
    if suite.get("milestone") != MILESTONE:
        raise ValueError("matrix suite is not M8-G0")
    if suite.get("native_results_status_at_creation") != "not_run":
        raise ValueError("matrix suite must be a fresh not_run manifest")
    records = suite.get("batch_manifests")
    if not isinstance(records, list) or not records:
        raise ValueError("matrix suite has no persistent batch manifests")
    if suite.get("persistent_batch_count") != len(records):
        raise ValueError("matrix suite persistent batch count is not exact")

    split = suite.get("split")
    if split not in {"baseline_gate", "development", "frozen_holdout"}:
        raise ValueError(f"unsupported M8 split: {split!r}")
    freeze_path = ""
    if split == "frozen_holdout":
        relative_freeze = _portable_relative(
            suite.get("freeze_manifest"),
            description="freeze manifest",
            allow_parent=True,
        )
        freeze_file = _contained(
            suite_path.parent / relative_freeze,
            root,
            description="freeze manifest",
        )
        if not freeze_file.is_file():
            raise ValueError(f"freeze manifest is not a file: {freeze_file}")
        freeze_path = str(freeze_file)

    fields = [str(split), freeze_path]
    seen_paths: set[Path] = set()
    seen_stems: set[str] = set()
    for index, record in enumerate(records):
        if not isinstance(record, Mapping):
            raise ValueError(f"batch record {index} is not an object")
        relative_batch = _portable_relative(
            record.get("path"), description=f"batch record {index} path"
        )
        batch_path = _contained(
            suite_path.parent / relative_batch,
            root,
            description=f"batch record {index}",
        )
        if not batch_path.is_file() or batch_path in seen_paths:
            raise ValueError(f"batch record {index} is missing or duplicated")
        seen_paths.add(batch_path)
        batch = _read_json(batch_path)
        if not isinstance(batch, Mapping):
            raise ValueError(f"batch manifest {index} root must be an object")
        strategy = batch.get("batch_strategy")
        if strategy not in STRATEGIES:
            raise ValueError(f"batch manifest {index} has unknown strategy: {strategy!r}")
        episodes = batch.get("episodes")
        expected = batch.get("expected_episode_count")
        if (
            not isinstance(episodes, list)
            or isinstance(expected, bool)
            or not isinstance(expected, int)
            or expected < 1
            or len(episodes) != expected
        ):
            raise ValueError(f"batch manifest {index} episode count is not exact")
        stem = re.sub(r"[^A-Za-z0-9_.-]", "_", batch_path.stem)
        if stem in seen_stems:
            raise ValueError(f"batch log stem is duplicated: {stem}")
        seen_stems.add(stem)
        profile = batch.get("batch_profile_id")
        if not isinstance(profile, str) or not profile:
            raise ValueError(f"batch manifest {index} profile id is missing")
        fields.extend((str(batch_path), str(strategy), stem, profile, str(expected)))
    _emit_nul(fields)
    return 0


def _validate_batch_artifacts(args: argparse.Namespace) -> int:
    root = Path(args.repository_root).resolve(strict=True)
    batch_path = _contained(Path(args.batch), root, description="batch manifest")
    batch = _read_json(batch_path)
    episodes = batch.get("episodes") if isinstance(batch, Mapping) else None
    expected = batch.get("expected_episode_count") if isinstance(batch, Mapping) else None
    if not isinstance(episodes, list) or expected != len(episodes):
        raise ValueError("batch episode inventory is malformed")
    for index, episode in enumerate(episodes):
        if not isinstance(episode, Mapping):
            raise ValueError(f"batch episode {index} is not an object")
        for field in ("event_log_path", "summary_path"):
            relative = _portable_relative(
                episode.get(field), description=f"episode {index} {field}"
            )
            artifact = _contained(
                batch_path.parent / relative,
                root,
                description=f"episode {index} {field}",
            )
            if not artifact.is_file() or artifact.stat().st_size <= 0:
                raise ValueError(f"native artifact is missing or empty: {artifact}")
    print(json.dumps({"passed": True, "episode_count": len(episodes)}, sort_keys=True))
    return 0


def _stable_source_file(path: Path, source_root: Path) -> bool:
    relative_parts = path.relative_to(source_root).parts
    directory_parts = {part.casefold() for part in relative_parts[:-1]}
    if directory_parts & EXCLUDED_SOURCE_DIRECTORIES:
        return False
    if any(part.casefold().endswith(".egg-info") for part in relative_parts[:-1]):
        return False
    return path.suffix.casefold() not in {".pyc", ".pyo"}


def _source_manifest(args: argparse.Namespace) -> int:
    root = Path(args.repository_root).resolve(strict=True)
    source_root = _contained(Path(args.source_root), root, description="ROS source root")
    if not source_root.is_dir():
        raise ValueError(f"ROS source root is not a directory: {source_root}")
    files = [
        path
        for path in source_root.rglob("*")
        if path.is_file() and _stable_source_file(path, source_root)
    ]
    files.sort(key=lambda path: str(path).casefold())
    records = [
        f"{path.relative_to(root).as_posix()}|{_sha256(path)}" for path in files
    ]
    digest = hashlib.sha256("\n".join(records).encode("utf-8")).hexdigest()
    output = Path(args.output)
    with output.open("x", encoding="utf-8", newline="\n") as handle:
        if records:
            handle.write("\n".join(records))
            handle.write("\n")
    _emit_nul((str(len(records)), digest, source_root.relative_to(root).as_posix()))
    return 0


def _parse_int(value: str, *, description: str, minimum: int = 0) -> int:
    stripped = value.strip()
    if not re.fullmatch(r"\d+", stripped):
        raise ValueError(f"{description} is not an integer: {value!r}")
    parsed = int(stripped)
    if parsed < minimum:
        raise ValueError(f"{description} is below {minimum}: {parsed}")
    return parsed


def _csv_rows(text: str) -> list[list[str]]:
    return [[field.strip() for field in row] for row in csv.reader(text.splitlines()) if row]


def _run_nvidia_smi(executable: Path, arguments: Sequence[str]) -> str:
    completed = subprocess.run(
        [str(executable), *arguments],
        check=False,
        capture_output=True,
        text=True,
        timeout=30,
    )
    if completed.returncode != 0:
        raise RuntimeError(
            "nvidia-smi failed with exit code "
            f"{completed.returncode}: {completed.stderr.strip()}"
        )
    return completed.stdout


def _gpu_snapshot(args: argparse.Namespace) -> int:
    executable = Path(args.nvidia_smi).resolve(strict=True)
    if not executable.is_file():
        raise ValueError(f"nvidia-smi is not a file: {executable}")
    inventory_text = _run_nvidia_smi(
        executable,
        (
            "--query-gpu=index,name,uuid,driver_version,memory.total,memory.used,utilization.gpu",
            "--format=csv,noheader,nounits",
        ),
    )
    inventory: list[dict[str, Any]] = []
    for row in _csv_rows(inventory_text):
        if len(row) != 7:
            raise ValueError(f"unexpected nvidia-smi GPU row: {row!r}")
        parsed = {
            "index": _parse_int(row[0], description="GPU index"),
            "name": row[1],
            "uuid": row[2],
            "driver_version": row[3],
            "memory_total_mib": _parse_int(
                row[4], description="GPU total memory", minimum=1
            ),
            "memory_used_mib": _parse_int(row[5], description="GPU used memory"),
            "utilization_gpu_percent": _parse_int(
                row[6], description="GPU utilization"
            ),
        }
        if not parsed["name"] or not parsed["uuid"] or not parsed["driver_version"]:
            raise ValueError("nvidia-smi GPU identity fields must be non-empty")
        if parsed["memory_used_mib"] > parsed["memory_total_mib"]:
            raise ValueError("nvidia-smi reports used memory greater than total memory")
        if parsed["utilization_gpu_percent"] > 100:
            raise ValueError("nvidia-smi reports utilization greater than 100 percent")
        if parsed["index"] == args.gpu_index:
            inventory.append(parsed)
    if len(inventory) != 1:
        raise ValueError(f"GPU index {args.gpu_index} did not resolve to exactly one GPU")
    selected_uuid = inventory[0]["uuid"]

    process_text = _run_nvidia_smi(
        executable,
        (
            "--query-compute-apps=gpu_uuid,pid,process_name,used_memory",
            "--format=csv,noheader,nounits",
        ),
    )
    reported: list[dict[str, Any]] = []
    for row in _csv_rows(process_text):
        if len(row) != 4:
            raise ValueError(f"unexpected nvidia-smi compute-process row: {row!r}")
        if not row[0] or not re.fullmatch(r"\d+", row[1]) or not row[2]:
            raise ValueError(f"malformed nvidia-smi compute-process row: {row!r}")
        if re.fullmatch(r"\d+", row[3]):
            used_memory: int | None = int(row[3])
        elif row[3].casefold() in {"n/a", "not supported", "[not supported]"}:
            used_memory = None
        else:
            raise ValueError(f"unexpected nvidia-smi compute memory value: {row!r}")
        if row[0] != selected_uuid:
            continue
        reported.append(
            {
                "gpu_uuid": row[0],
                "process_id": int(row[1]),
                "process_name": row[2],
                "used_memory_mib": used_memory,
                "actionable_compute_allocation": (
                    isinstance(used_memory, int) and used_memory > 0
                ),
            }
        )
    actionable = [record for record in reported if record["actionable_compute_allocation"]]
    occupied = [
        record
        for record in inventory
        if record["memory_used_mib"] > args.memory_threshold_mib
    ]
    payload = {
        "phase": args.phase,
        "captured_utc": _utc_now(),
        "selected_gpu_index": args.gpu_index,
        "gpu_inventory": inventory,
        "reported_compute_processes": reported,
        "actionable_compute_processes": actionable,
        "blocking_compute_processes": reported,
        "unknown_memory_compute_processes": [
            record for record in reported if record["used_memory_mib"] is None
        ],
        "occupied_gpus": occupied,
        # On Linux every compute-app row on the selected UUID blocks launch,
        # even when the driver cannot quantify its memory. Aggregate used
        # memory remains an independent refusal gate.
        "passed": not reported and not occupied,
    }
    _write_json_exclusive(Path(args.output), payload)
    print(selected_uuid)
    return 0 if payload["passed"] else 3


def _copy_evidence(args: argparse.Namespace) -> int:
    source = Path(args.source).resolve(strict=True)
    destination = Path(args.destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    with source.open("rb") as reader, destination.open("xb") as writer:
        shutil.copyfileobj(reader, writer, length=1024 * 1024)
        writer.flush()
        os.fsync(writer.fileno())
    if _sha256(source) != _sha256(destination):
        raise RuntimeError("copied evidence hash mismatch")
    destination.chmod(0o444)
    print(_sha256(destination))
    return 0


def _gpu_refusal(args: argparse.Namespace) -> int:
    snapshot = _read_json(Path(args.snapshot))
    payload = {
        "schema_version": SCHEMA_VERSION,
        "milestone": MILESTONE,
        "receipt_kind": "gpu_preflight_refusal",
        "created_utc": _utc_now(),
        "operator_authorized_native_gpu_run": True,
        "preflight_phase": snapshot.get("phase"),
        "threshold_mib": args.memory_threshold_mib,
        "gpu_inventory": snapshot.get("gpu_inventory", []),
        "reported_compute_processes": snapshot.get("reported_compute_processes", []),
        "actionable_compute_processes": snapshot.get(
            "actionable_compute_processes", []
        ),
        "blocking_compute_processes": snapshot.get("blocking_compute_processes", []),
        "unknown_memory_compute_processes": snapshot.get(
            "unknown_memory_compute_processes", []
        ),
        "occupied_gpus": snapshot.get("occupied_gpus", []),
        "action": "refused_without_killing_any_process",
    }
    _write_json_exclusive(Path(args.output), payload)
    return 0


def _validated_gpu_identity(
    initial: Any,
    post: Any,
    *,
    gpu_index: int,
    memory_threshold_mib: int,
) -> dict[str, Any]:
    identities: list[dict[str, Any]] = []
    for snapshot, expected_phase in (
        (initial, "initial_pre_build"),
        (post, "post_build_pre_launch"),
    ):
        if not isinstance(snapshot, Mapping):
            raise ValueError("successful GPU preflight snapshot must be an object")
        inventory = snapshot.get("gpu_inventory")
        if (
            snapshot.get("phase") != expected_phase
            or snapshot.get("passed") is not True
            or snapshot.get("selected_gpu_index") != gpu_index
            or not isinstance(inventory, list)
            or len(inventory) != 1
            or not isinstance(inventory[0], Mapping)
        ):
            raise ValueError("successful GPU preflight snapshot identity is malformed")
        row = dict(inventory[0])
        if (
            row.get("index") != gpu_index
            or not isinstance(row.get("name"), str)
            or not row["name"].strip()
            or not isinstance(row.get("uuid"), str)
            or not re.fullmatch(r"GPU-[A-Za-z0-9-]+", row["uuid"])
            or not isinstance(row.get("driver_version"), str)
            or not row["driver_version"].strip()
            or type(row.get("memory_total_mib")) is not int
            or row["memory_total_mib"] <= 0
            or type(row.get("memory_used_mib")) is not int
            or not 0 <= row["memory_used_mib"] <= memory_threshold_mib
            or type(row.get("utilization_gpu_percent")) is not int
            or not 0 <= row["utilization_gpu_percent"] <= 100
        ):
            raise ValueError("successful GPU preflight inventory row is malformed")
        for name in (
            "reported_compute_processes",
            "actionable_compute_processes",
            "blocking_compute_processes",
            "unknown_memory_compute_processes",
            "occupied_gpus",
        ):
            if snapshot.get(name) != []:
                raise ValueError(f"successful GPU preflight {name} must be empty")
        identities.append(
            {
                key: row[key]
                for key in (
                    "index",
                    "name",
                    "uuid",
                    "driver_version",
                    "memory_total_mib",
                )
            }
        )
    if identities[0] != identities[1]:
        raise ValueError("selected GPU identity changed between preflight snapshots")
    return identities[1]


def _preflight(args: argparse.Namespace) -> int:
    root = Path(args.repository_root).resolve(strict=True)
    output = _contained_output(Path(args.output), root, description="preflight receipt")
    receipt_dir = output.parent.resolve(strict=True)
    suite = _contained(Path(args.suite), root, description="matrix suite")
    source_root = _contained(Path(args.source_root), root, description="ROS source root")
    initial = _read_json(Path(args.initial_snapshot))
    post = _read_json(Path(args.post_snapshot))
    selected_gpu = _validated_gpu_identity(
        initial,
        post,
        gpu_index=args.gpu_index,
        memory_threshold_mib=args.memory_threshold_mib,
    )
    external_environment = _contained(
        Path(args.external_environment),
        receipt_dir,
        description="external environment evidence",
    )
    external_payload = _read_json(external_environment)
    if (
        not isinstance(external_payload, Mapping)
        or external_payload.get("schema_version") != SCHEMA_VERSION
        or external_payload.get("milestone") != MILESTONE
        or external_payload.get("evidence_kind") != "native_external_environment"
    ):
        raise ValueError("external environment evidence is not a formal M8 record")
    adapter = Path(args.installed_adapter).resolve(strict=True)
    adapter_evidence = _contained(
        Path(args.installed_adapter_evidence),
        receipt_dir,
        description="installed adapter evidence",
    )
    runner = _contained(Path(args.runner), root, description="runner source")
    executor = Path(args.executor).resolve(strict=True)
    if not executor.is_file():
        raise ValueError("installed executor is not a regular file")
    executor_evidence = _contained(
        Path(args.executor_evidence),
        receipt_dir,
        description="installed executor evidence",
    )
    executor_hash = _sha256(executor)
    if executor_hash != _sha256(executor_evidence):
        raise ValueError("installed executor evidence does not match installed executor")
    router = Path(args.router).resolve(strict=True)
    if not router.is_file():
        raise ValueError("Zenoh router is not a regular file")
    router_hash = _sha256(router)
    runner_evidence = _contained(
        Path(args.runner_evidence), receipt_dir, description="runner evidence"
    )
    runner_support = _contained(
        Path(args.runner_support), root, description="runner support source"
    )
    runner_support_evidence = _contained(
        Path(args.runner_support_evidence),
        receipt_dir,
        description="runner support evidence",
    )
    if runner.relative_to(root).as_posix() != LINUX_RUNNER_SOURCE:
        raise ValueError("Linux preflight runner source path is not canonical")
    if runner_support.relative_to(root).as_posix() != LINUX_RUNNER_SUPPORT_SOURCE:
        raise ValueError("Linux preflight runner support source path is not canonical")
    adapter_hash = _sha256(adapter)
    runner_hash = _sha256(runner)
    runner_support_hash = _sha256(runner_support)
    if adapter_hash != _sha256(adapter_evidence):
        raise ValueError("installed adapter evidence does not match installed adapter")
    if runner_hash != _sha256(runner_evidence):
        raise ValueError("runner evidence does not match runner source")
    if runner_support_hash != _sha256(runner_support_evidence):
        raise ValueError("runner support evidence does not match runner support source")
    payload = {
        "schema_version": SCHEMA_VERSION,
        "milestone": MILESTONE,
        "receipt_kind": "native_preflight",
        "created_utc": _utc_now(),
        "operator_authorized_native_gpu_run": True,
        "gpu_memory_refusal_threshold_mib": args.memory_threshold_mib,
        "gpu_inventory": post["gpu_inventory"],
        "reported_compute_processes": post["reported_compute_processes"],
        "preexisting_compute_processes": post["actionable_compute_processes"],
        "initial_gpu_preflight": initial,
        "post_build_gpu_preflight": post,
        "frozen_live_inputs_validated": args.frozen_live_inputs_validated,
        "current_source_batch_validation_passed": True,
        "current_source_batch_validation_count": len(args.validation_log),
        "current_source_batch_validation_logs": args.validation_log,
        "suite_manifest": suite.relative_to(root).as_posix(),
        "suite_manifest_sha256": _sha256(suite),
        "source_root": source_root.relative_to(root).as_posix(),
        "source_file_count": args.source_file_count,
        "source_manifest_sha256": args.source_manifest_sha256,
        "external_environment_evidence": external_environment.relative_to(
            receipt_dir
        ).as_posix(),
        "external_environment_evidence_sha256": _sha256(external_environment),
        "installed_dynamic_adapter": str(adapter),
        "installed_dynamic_adapter_evidence": adapter_evidence.relative_to(
            receipt_dir
        ).as_posix(),
        "installed_dynamic_adapter_evidence_sha256": _sha256(adapter_evidence),
        "installed_dynamic_adapter_sha256": adapter_hash,
        "executor_path": str(executor),
        "executor_evidence": executor_evidence.relative_to(receipt_dir).as_posix(),
        "executor_evidence_sha256": _sha256(executor_evidence),
        "executor_sha256": executor_hash,
        "router_path": str(router),
        "router_sha256": router_hash,
        "runner": str(runner),
        "runner_source": runner.relative_to(root).as_posix(),
        "runner_evidence": runner_evidence.relative_to(receipt_dir).as_posix(),
        "runner_evidence_sha256": _sha256(runner_evidence),
        "runner_sha256": runner_hash,
        "runner_support_source": runner_support.relative_to(root).as_posix(),
        "runner_support_evidence": runner_support_evidence.relative_to(
            receipt_dir
        ).as_posix(),
        "runner_support_evidence_sha256": _sha256(runner_support_evidence),
        "runner_support_sha256": runner_support_hash,
        "selected_gpu_index": selected_gpu["index"],
        "selected_gpu_uuid": selected_gpu["uuid"],
        "selected_gpu_identity": selected_gpu,
        "batch_timeout_seconds": args.batch_timeout_seconds,
        "planned_process_logs": args.planned_log,
    }
    _write_json_exclusive(output, payload)
    return 0


def _available_logs(log_directory: Path) -> list[dict[str, Any]]:
    if not log_directory.is_dir():
        return []
    return [
        {
            "path": path.name,
            "size_bytes": path.stat().st_size,
            "sha256": _sha256(path),
        }
        for path in sorted(log_directory.glob("*.log"), key=lambda item: item.name)
        if path.is_file()
    ]


def _optional_snapshot(value: str | None) -> Any:
    if not value:
        return None
    path = Path(value)
    return _read_json(path) if path.is_file() else None


def _failure(args: argparse.Namespace) -> int:
    root = Path(args.repository_root).resolve(strict=True)
    log_directory = _contained(
        Path(args.log_directory), root, description="native log directory"
    )
    output = _contained_output(Path(args.output), root, description="failure receipt")
    suite = Path(args.suite).resolve(strict=False)
    suite_reference: str = str(suite)
    suite_hash: str | None = None
    if suite.is_file():
        suite_hash = _sha256(suite)
        try:
            suite_reference = suite.relative_to(root).as_posix()
        except ValueError:
            pass
    preflight_reference: str | None = None
    preflight_hash: str | None = None
    if args.preflight_receipt:
        preflight = Path(args.preflight_receipt)
        if preflight.is_file():
            preflight = _contained(
                preflight, log_directory, description="preflight receipt"
            )
            preflight_reference = preflight.relative_to(log_directory).as_posix()
            preflight_hash = _sha256(preflight)
    current_batch: str | None = None
    if args.current_batch:
        batch = Path(args.current_batch).resolve(strict=False)
        try:
            current_batch = os.path.relpath(batch, log_directory).replace(os.sep, "/")
        except ValueError:
            current_batch = str(batch)
    payload = {
        "schema_version": SCHEMA_VERSION,
        "milestone": MILESTONE,
        "receipt_kind": "native_timeout" if args.timed_out else "native_failure",
        "status": "timeout" if args.timed_out else "failed",
        "created_utc": _utc_now(),
        "stage": args.stage,
        "current_batch": current_batch,
        "batch_timeout_seconds": args.batch_timeout_seconds,
        "suite_manifest": suite_reference,
        "suite_manifest_sha256": suite_hash,
        "source_manifest_sha256": args.source_manifest_sha256 or None,
        "preflight_receipt": preflight_reference,
        "preflight_receipt_sha256": preflight_hash,
        "initial_gpu_preflight": _optional_snapshot(args.initial_snapshot),
        "post_build_gpu_preflight": _optional_snapshot(args.post_snapshot),
        "error": {
            "type": "native_runner_exit",
            "message": args.message,
            "command": args.command,
            "line": args.line,
            "exit_code": args.exit_code,
        },
        "available_logs": _available_logs(log_directory),
        "action": "stopped_only_owned_process_groups_and_preserved_available_logs",
    }
    _write_json_exclusive(output, payload)
    return 0


def _completion(args: argparse.Namespace) -> int:
    root = Path(args.repository_root).resolve(strict=True)
    output = _contained_output(Path(args.output), root, description="completion receipt")
    receipt_dir = output.parent.resolve(strict=True)

    def receipt_relative(value: str, *, description: str) -> tuple[str, str]:
        artifact = _contained(Path(value), root, description=description)
        return os.path.relpath(artifact, receipt_dir).replace(os.sep, "/"), _sha256(
            artifact
        )

    preflight_path, preflight_hash = receipt_relative(
        args.preflight_receipt, description="preflight receipt"
    )
    replay_path, replay_hash = receipt_relative(
        args.replay, description="replay validation"
    )
    archive_path, archive_hash = receipt_relative(
        args.process_log_archive, description="process-log archive"
    )
    manifest_path, manifest_hash = receipt_relative(
        args.process_log_manifest, description="process-log manifest"
    )
    analysis_path: str | None = None
    analysis_hash: str | None = None
    if args.analysis:
        analysis_path, analysis_hash = receipt_relative(
            args.analysis, description="analysis output"
        )
    payload = {
        "schema_version": SCHEMA_VERSION,
        "milestone": MILESTONE,
        "receipt_kind": "native_completion",
        "completed_utc": _utc_now(),
        "status": "complete",
        "preflight_receipt": preflight_path,
        "preflight_receipt_sha256": preflight_hash,
        "replay_validation": replay_path,
        "replay_validation_sha256": replay_hash,
        "analysis": analysis_path,
        "analysis_sha256": analysis_hash,
        "process_log_archive": archive_path,
        "process_log_archive_sha256": archive_hash,
        "process_log_manifest": manifest_path,
        "process_log_manifest_sha256": manifest_hash,
        "process_log_member_count": args.process_log_member_count,
    }
    _write_json_exclusive(output, payload)
    return 0


def _sha256_command(args: argparse.Namespace) -> int:
    print(_sha256(Path(args.path).resolve(strict=True)))
    return 0


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)

    runtime = commands.add_parser("runtime-probe")
    runtime.set_defaults(handler=_runtime_probe)

    environment = commands.add_parser("validate-external-environment")
    environment.add_argument("--isaac-workspace", required=True)
    environment.add_argument("--pixi-exe", required=True)
    environment.set_defaults(handler=_validate_external_environment)

    environment_write = commands.add_parser("write-external-environment")
    environment_write.add_argument("--isaac-workspace", required=True)
    environment_write.add_argument("--pixi-exe", required=True)
    environment_write.add_argument("--output", required=True)
    environment_write.set_defaults(handler=_write_external_environment)

    suite = commands.add_parser("suite-records")
    suite.add_argument("--repository-root", required=True)
    suite.add_argument("--suite", required=True)
    suite.set_defaults(handler=_suite_records)

    artifacts = commands.add_parser("validate-batch-artifacts")
    artifacts.add_argument("--repository-root", required=True)
    artifacts.add_argument("--batch", required=True)
    artifacts.set_defaults(handler=_validate_batch_artifacts)

    source = commands.add_parser("source-manifest")
    source.add_argument("--repository-root", required=True)
    source.add_argument("--source-root", required=True)
    source.add_argument("--output", required=True)
    source.set_defaults(handler=_source_manifest)

    gpu = commands.add_parser("gpu-snapshot")
    gpu.add_argument("--nvidia-smi", required=True)
    gpu.add_argument("--gpu-index", type=int, required=True)
    gpu.add_argument("--memory-threshold-mib", type=int, required=True)
    gpu.add_argument("--phase", required=True)
    gpu.add_argument("--output", required=True)
    gpu.set_defaults(handler=_gpu_snapshot)

    copy = commands.add_parser("copy-evidence")
    copy.add_argument("--source", required=True)
    copy.add_argument("--destination", required=True)
    copy.set_defaults(handler=_copy_evidence)

    refusal = commands.add_parser("write-gpu-refusal")
    refusal.add_argument("--snapshot", required=True)
    refusal.add_argument("--memory-threshold-mib", type=int, required=True)
    refusal.add_argument("--output", required=True)
    refusal.set_defaults(handler=_gpu_refusal)

    preflight = commands.add_parser("write-preflight")
    preflight.add_argument("--repository-root", required=True)
    preflight.add_argument("--output", required=True)
    preflight.add_argument("--suite", required=True)
    preflight.add_argument("--source-root", required=True)
    preflight.add_argument("--source-file-count", type=int, required=True)
    preflight.add_argument("--source-manifest-sha256", required=True)
    preflight.add_argument("--external-environment", required=True)
    preflight.add_argument("--memory-threshold-mib", type=int, required=True)
    preflight.add_argument("--gpu-index", type=int, required=True)
    preflight.add_argument("--initial-snapshot", required=True)
    preflight.add_argument("--post-snapshot", required=True)
    preflight.add_argument(
        "--frozen-live-inputs-validated", action=argparse.BooleanOptionalAction
    )
    preflight.add_argument("--validation-log", action="append", default=[])
    preflight.add_argument("--installed-adapter", required=True)
    preflight.add_argument("--installed-adapter-evidence", required=True)
    preflight.add_argument("--executor", required=True)
    preflight.add_argument("--executor-evidence", required=True)
    preflight.add_argument("--router", required=True)
    preflight.add_argument("--runner", required=True)
    preflight.add_argument("--runner-evidence", required=True)
    preflight.add_argument("--runner-support", required=True)
    preflight.add_argument("--runner-support-evidence", required=True)
    preflight.add_argument("--batch-timeout-seconds", type=int, required=True)
    preflight.add_argument("--planned-log", action="append", default=[])
    preflight.set_defaults(handler=_preflight)

    failure = commands.add_parser("write-failure")
    failure.add_argument("--repository-root", required=True)
    failure.add_argument("--output", required=True)
    failure.add_argument("--log-directory", required=True)
    failure.add_argument("--stage", required=True)
    failure.add_argument("--suite", required=True)
    failure.add_argument("--batch-timeout-seconds", type=int, required=True)
    failure.add_argument("--current-batch")
    failure.add_argument("--preflight-receipt")
    failure.add_argument("--source-manifest-sha256")
    failure.add_argument("--initial-snapshot")
    failure.add_argument("--post-snapshot")
    failure.add_argument("--message", required=True)
    failure.add_argument("--command", required=True)
    failure.add_argument("--line", type=int, required=True)
    failure.add_argument("--exit-code", type=int, required=True)
    failure.add_argument("--timed-out", action="store_true")
    failure.set_defaults(handler=_failure)

    completion = commands.add_parser("write-completion")
    completion.add_argument("--repository-root", required=True)
    completion.add_argument("--output", required=True)
    completion.add_argument("--preflight-receipt", required=True)
    completion.add_argument("--replay", required=True)
    completion.add_argument("--analysis")
    completion.add_argument("--process-log-archive", required=True)
    completion.add_argument("--process-log-manifest", required=True)
    completion.add_argument("--process-log-member-count", type=int, required=True)
    completion.set_defaults(handler=_completion)

    digest = commands.add_parser("sha256")
    digest.add_argument("--path", required=True)
    digest.set_defaults(handler=_sha256_command)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    try:
        return int(args.handler(args))
    except Exception as exc:
        print(f"m8_linux_runner_support: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
