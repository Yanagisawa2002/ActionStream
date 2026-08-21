"""Run the frozen Transport H1-Final learned-policy GPU holdout."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time
from typing import Any

from actionstream.current_baselines import load_protocol


ROOT = Path(__file__).resolve().parents[2]
ALLOWED_RUNTIMES = {
    "actionstream_backend_aligned",
    "actionstream_backend_pipelined_aligned",
    "lerobot_latest_only",
}
FROZEN_FILES = {
    "current_baselines_sha256": ROOT / "src/actionstream/current_baselines.py",
    "lerobot_inference_sha256": ROOT / "src/actionstream/lerobot_inference.py",
    "gpu_measurement_sha256": ROOT / "src/actionstream/gpu_measurement.py",
    "orchestrator_sha256": Path(__file__).resolve(),
}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _write_json(path: Path, value: Any) -> None:
    path.write_text(
        json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
        newline="\n",
    )


def _git_head() -> str:
    completed = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
        timeout=20,
    )
    return completed.stdout.strip()


def _nvidia_snapshot() -> dict[str, Any]:
    completed = subprocess.run(
        [
            "nvidia-smi",
            "--query-gpu=index,name,uuid,utilization.gpu,memory.used,memory.total",
            "--format=csv,noheader,nounits",
        ],
        check=False,
        capture_output=True,
        encoding="utf-8",
        errors="replace",
        timeout=20,
    )
    return {
        "returncode": completed.returncode,
        "stdout": completed.stdout.strip(),
        "stderr": completed.stderr.strip(),
    }


def _validate_protocol(path: Path) -> Any:
    protocol = load_protocol(path)
    raw = protocol.raw
    if raw.get("protocol_status") != "frozen_holdout":
        raise ValueError(f"{path} is not a frozen_holdout protocol")
    if raw.get("benchmark_version") != 3:
        raise ValueError(f"{path} is not GPU benchmark v3")
    if raw.get("hypothesis_id") != "transport_h1_final_serialized_delivery":
        raise ValueError(f"Unexpected hypothesis in {path}")
    runtimes = set(raw.get("runtimes", []))
    if runtimes != ALLOWED_RUNTIMES:
        raise ValueError(f"Unexpected H1-Final runtimes: {sorted(runtimes)}")
    execution_order = raw.get("runtime_execution_order", [])
    if set(execution_order) != ALLOWED_RUNTIMES or len(execution_order) != 3:
        raise ValueError(f"Invalid runtime execution order in {path}")
    if raw.get("delay_profiles") != [
        {
            "key": "fixed_0950_transport_h1_final",
            "kind": "fixed",
            "milliseconds": 950,
        }
    ]:
        raise ValueError(f"H1-Final must contain only fixed 950 ms: {path}")
    measurement = raw.get("measurement_v2", {})
    if measurement.get("fresh_process_per_runtime") is not True:
        raise ValueError("H1-Final requires a fresh process per runtime and family")
    if int(measurement.get("worker_warmup", {}).get("inference_calls", 0)) != 2:
        raise ValueError("H1-Final freezes exactly two actual-worker warmup calls")
    if raw.get("evidence_boundary", {}).get("episodes_per_runtime_family") != 5:
        raise ValueError("H1-Final freezes five scored resets per runtime and family")
    candidate = raw.get("candidate", {})
    mismatches = {
        key: {"expected": candidate.get(key), "actual": _sha256(file_path)}
        for key, file_path in FROZEN_FILES.items()
        if candidate.get(key) != _sha256(file_path)
    }
    if mismatches:
        raise RuntimeError(f"Frozen H1-Final candidate hash mismatch: {mismatches}")
    return protocol


def _validate_cell(cell_dir: Path, runtime: str) -> dict[str, Any]:
    receipt_path = cell_dir / "run_receipt.json"
    if not receipt_path.is_file():
        raise RuntimeError(f"Missing child receipt: {receipt_path}")
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    selection = receipt.get("selection", {})
    if selection.get("runtimes") != [runtime]:
        raise RuntimeError(f"Child process mixed runtimes: {selection}")
    if selection.get("delay_profiles") != ["fixed_0950_transport_h1_final"]:
        raise RuntimeError(f"Child process used the wrong delay profile: {selection}")
    gpu = receipt.get("system_gpu") or {}
    steady = (gpu.get("phases") or {}).get("steady_state") or {}
    if int(gpu.get("sample_count", 0)) <= 0:
        raise RuntimeError(f"No nvidia-smi samples for {cell_dir}")
    if int(steady.get("resident_sample_count", 0)) <= 0:
        raise RuntimeError(f"No steady process GPU residency for {cell_dir}")
    if steady.get("process_gpu_memory_mib_max") is None:
        raise RuntimeError(f"Missing process VRAM residency for {cell_dir}")
    records = [
        json.loads(line)
        for line in (cell_dir / "episodes.jsonl")
        .read_text(encoding="utf-8")
        .splitlines()
        if line.strip()
    ]
    if len(records) != 5:
        raise RuntimeError(f"Expected five H1-Final rows in {cell_dir}")
    if any(row.get("worker_warmup") is None for row in records):
        raise RuntimeError(f"Missing actual-worker warmup provenance: {cell_dir}")
    if any(row.get("inference_requests_per_second") is None for row in records):
        raise RuntimeError(f"Missing true request/s: {cell_dir}")
    if any(row.get("system_gpu") is None for row in records):
        raise RuntimeError(f"Missing GPU measurement: {cell_dir}")
    return {"receipt": receipt, "records": records}


def run(args: argparse.Namespace) -> dict[str, Any]:
    output_root = args.output_root.resolve()
    output_root.mkdir(parents=True, exist_ok=False)
    protocols = [
        (path.resolve(), _validate_protocol(path.resolve())) for path in args.protocol
    ]
    families = [str(protocol.raw["task_family"]) for _, protocol in protocols]
    if len(families) != 3 or len(set(families)) != 3:
        raise ValueError("H1-Final requires exactly three task families")
    if args.capture is not True:
        raise ValueError("H1-Final requires paired video capture")

    base_commits = {
        str(protocol.raw["candidate"]["base_commit"])
        for _, protocol in protocols
    }
    if len(base_commits) != 1:
        raise ValueError("H1-Final protocols do not share one candidate commit")
    source_commit = next(iter(base_commits))
    environment = os.environ.copy()
    environment["ACTIONSTREAM_SOURCE_COMMIT"] = source_commit
    environment.setdefault("MUJOCO_GL", "egl")
    environment.setdefault("PYOPENGL_PLATFORM", "egl")
    environment.setdefault("HF_HUB_OFFLINE", "1")

    execution_head = _git_head()
    child_receipts: list[dict[str, Any]] = []
    all_records: list[dict[str, Any]] = []
    run_started = time.monotonic()
    for protocol_path, protocol in protocols:
        family = str(protocol.raw["task_family"])
        for runtime in protocol.raw["runtime_execution_order"]:
            cell_dir = output_root / "cells" / family / runtime
            cell_dir.mkdir(parents=True, exist_ok=False)
            log_path = cell_dir / "subprocess.log"
            command = [
                sys.executable,
                "-m",
                "actionstream.current_baselines",
                "--protocol",
                str(protocol_path),
                "--lerobot-root",
                str(args.lerobot_root.resolve()),
                "--output-dir",
                str(cell_dir),
                "--models",
                "xvla",
                "--runtimes",
                runtime,
                "--capture",
            ]
            started_utc_ns = time.time_ns()
            started = time.monotonic()
            pre_gpu = _nvidia_snapshot()
            with log_path.open("w", encoding="utf-8", newline="\n") as log:
                process = subprocess.Popen(
                    command,
                    cwd=ROOT,
                    env=environment,
                    stdout=log,
                    stderr=subprocess.STDOUT,
                    text=True,
                )
                returncode = process.wait()
            post_gpu = _nvidia_snapshot()
            child = {
                "schema_version": 1,
                "family": family,
                "runtime": runtime,
                "fresh_process": True,
                "pid": process.pid,
                "command": command,
                "started_utc_unix_ns": started_utc_ns,
                "wall_clock_seconds": time.monotonic() - started,
                "returncode": returncode,
                "pre_gpu": pre_gpu,
                "post_gpu": post_gpu,
                "log_path": str(log_path),
                "log_sha256": _sha256(log_path),
            }
            _write_json(cell_dir / "subprocess_receipt.json", child)
            child_receipts.append(child)
            print(
                f"family={family} runtime={runtime} "
                f"exit={returncode} seconds={child['wall_clock_seconds']:.1f}",
                flush=True,
            )
            if returncode != 0:
                _write_json(
                    output_root / "orchestrator_failed.json",
                    {
                        "status": "failed",
                        "execution_git_head": execution_head,
                        "children": child_receipts,
                    },
                )
                raise RuntimeError(f"Child failed; inspect {log_path}")
            validated = _validate_cell(cell_dir, runtime)
            all_records.extend(validated["records"])

    if len(all_records) != 45:
        raise RuntimeError(f"Expected 45 H1-Final rows, got {len(all_records)}")
    episodes_path = output_root / "episodes.jsonl"
    episodes_path.write_text(
        "".join(
            json.dumps(row, sort_keys=True, allow_nan=False) + "\n"
            for row in all_records
        ),
        encoding="utf-8",
        newline="\n",
    )
    receipt = {
        "schema_version": 1,
        "status": "completed",
        "experiment_id": "actionstream_transport_h1_final_20260821",
        "hypothesis_id": "transport_h1_final_serialized_delivery",
        "candidate_base_commit": source_commit,
        "execution_git_head": execution_head,
        "protocols": [
            {"path": str(path), "sha256": _sha256(path)} for path, _ in protocols
        ],
        "families": families,
        "fresh_process_count": len(child_receipts),
        "fresh_process_pids": [item["pid"] for item in child_receipts],
        "record_count": len(all_records),
        "wall_clock_seconds": time.monotonic() - run_started,
        "episodes_path": str(episodes_path),
        "episodes_sha256": _sha256(episodes_path),
        "children": child_receipts,
    }
    _write_json(output_root / "orchestrator_receipt.json", receipt)
    artifacts = {
        str(path.relative_to(output_root).as_posix()): _sha256(path)
        for path in output_root.rglob("*")
        if path.is_file() and path.name != "artifact_manifest.json"
    }
    _write_json(
        output_root / "artifact_manifest.json",
        {"schema_version": 1, "artifacts": artifacts},
    )
    return receipt


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", type=Path, action="append", required=True)
    parser.add_argument("--lerobot-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--capture", action="store_true")
    return parser


def main() -> None:
    print(json.dumps(run(_parser().parse_args()), sort_keys=True))


if __name__ == "__main__":
    main()
