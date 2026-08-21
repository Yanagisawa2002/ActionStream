"""Resume H1-Final after the frozen orchestrator's receipt-key failure."""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
import time
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
ORIGINAL_ORCHESTRATOR = ROOT / "scripts/experiments/run_transport_h1_final.py"
RESUME_CONFIG = ROOT / "configs/actionstream_transport_h1_final_resume.json"
EXPECTED_REUSED_CELL = ("object", "actionstream_backend_aligned")


def _load_original() -> Any:
    spec = importlib.util.spec_from_file_location(
        "run_transport_h1_final_frozen", ORIGINAL_ORCHESTRATOR
    )
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot load frozen orchestrator: {ORIGINAL_ORCHESTRATOR}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


ORIGINAL = _load_original()


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


def _validate_resume_config(path: Path) -> dict[str, Any]:
    raw = json.loads(path.read_text(encoding="utf-8"))
    if raw.get("status") != "frozen_structural_resume_before_paired_result":
        raise ValueError("Resume authorization is not frozen before paired results")
    if raw.get("hypothesis_id") != "transport_h1_final_serialized_delivery":
        raise ValueError("Resume authorization changed the hypothesis")
    frozen = raw.get("frozen_artifacts", {})
    checks = {
        "original_orchestrator_sha256": _sha256(ORIGINAL_ORCHESTRATOR),
        "resume_orchestrator_sha256": _sha256(Path(__file__).resolve()),
    }
    mismatches = {
        key: {"expected": frozen.get(key), "actual": actual}
        for key, actual in checks.items()
        if frozen.get(key) != actual
    }
    if mismatches:
        raise RuntimeError(f"Structural resume hash mismatch: {mismatches}")
    if raw.get("resume_contract", {}).get("completed_cell_to_reuse") != {
        "family": EXPECTED_REUSED_CELL[0],
        "runtime": EXPECTED_REUSED_CELL[1],
        "record_count": 5,
    }:
        raise ValueError("Resume authorization does not identify one exact cell")
    return raw


def _validate_cell(cell_dir: Path, runtime: str) -> dict[str, Any]:
    """Validate the current-baselines receipt using its actual `profiles` key."""
    receipt_path = cell_dir / "run_receipt.json"
    if not receipt_path.is_file():
        raise RuntimeError(f"Missing child receipt: {receipt_path}")
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    selection = receipt.get("selection", {})
    if selection.get("runtimes") != [runtime]:
        raise RuntimeError(f"Child process mixed runtimes: {selection}")
    if selection.get("profiles") != ["fixed_0950_transport_h1_final"]:
        raise RuntimeError(f"Child process used the wrong profile: {selection}")
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
    trace_count = len(list((cell_dir / "traces").glob("*.json")))
    video_count = len(list((cell_dir / "videos").glob("*.mp4")))
    if trace_count != 5 or video_count != 1:
        raise RuntimeError(
            f"Expected five traces and one representative video in {cell_dir}; "
            f"got traces={trace_count}, videos={video_count}"
        )
    return {"receipt": receipt, "records": records}


def _run_new_cell(
    *,
    protocol_path: Path,
    lerobot_root: Path,
    cell_dir: Path,
    family: str,
    runtime: str,
    environment: dict[str, str],
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    cell_dir.mkdir(parents=True, exist_ok=False)
    log_path = cell_dir / "subprocess.log"
    command = [
        sys.executable,
        "-m",
        "actionstream.current_baselines",
        "--protocol",
        str(protocol_path),
        "--lerobot-root",
        str(lerobot_root),
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
    pre_gpu = ORIGINAL._nvidia_snapshot()
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
    post_gpu = ORIGINAL._nvidia_snapshot()
    child = {
        "schema_version": 1,
        "family": family,
        "runtime": runtime,
        "fresh_process": True,
        "reused_from_structural_failure": False,
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
    print(
        f"resume family={family} runtime={runtime} "
        f"exit={returncode} seconds={child['wall_clock_seconds']:.1f}",
        flush=True,
    )
    if returncode != 0:
        raise RuntimeError(f"Child failed; inspect {log_path}")
    return child, _validate_cell(cell_dir, runtime)["records"]


def run(args: argparse.Namespace) -> dict[str, Any]:
    output_root = args.output_root.resolve()
    if not output_root.is_dir():
        raise FileNotFoundError(f"Structural resume root is absent: {output_root}")
    resume = _validate_resume_config(args.resume_config.resolve())
    protocols = [
        (path.resolve(), ORIGINAL._validate_protocol(path.resolve()))
        for path in args.protocol
    ]
    families = [str(protocol.raw["task_family"]) for _, protocol in protocols]
    if len(families) != 3 or len(set(families)) != 3:
        raise ValueError("H1-Final resume requires exactly three task families")

    failure_log = ROOT / resume["frozen_artifacts"]["first_process_log_path"]
    reused_cell = output_root / "cells" / EXPECTED_REUSED_CELL[0] / EXPECTED_REUSED_CELL[1]
    frozen = resume["frozen_artifacts"]
    if _sha256(failure_log) != frozen["first_process_log_sha256"]:
        raise RuntimeError("First process-session log changed before resume")
    if _sha256(reused_cell / "run_receipt.json") != frozen["reused_run_receipt_sha256"]:
        raise RuntimeError("Completed cell receipt changed before resume")
    if _sha256(reused_cell / "episodes.jsonl") != frozen["reused_episodes_sha256"]:
        raise RuntimeError("Completed cell episodes changed before resume")

    environment = os.environ.copy()
    environment["ACTIONSTREAM_SOURCE_COMMIT"] = str(
        protocols[0][1].raw["candidate"]["base_commit"]
    )
    environment.setdefault("MUJOCO_GL", "egl")
    environment.setdefault("PYOPENGL_PLATFORM", "egl")
    environment.setdefault("HF_HUB_OFFLINE", "1")

    child_receipts: list[dict[str, Any]] = []
    all_records: list[dict[str, Any]] = []
    reused_cells = 0
    executed_cells = 0
    run_started = time.monotonic()
    for protocol_path, protocol in protocols:
        family = str(protocol.raw["task_family"])
        for runtime in protocol.raw["runtime_execution_order"]:
            cell_dir = output_root / "cells" / family / runtime
            identity = (family, runtime)
            if cell_dir.exists():
                if identity != EXPECTED_REUSED_CELL or reused_cells != 0:
                    raise RuntimeError(f"Unexpected pre-existing cell: {cell_dir}")
                validated = _validate_cell(cell_dir, runtime)
                child = json.loads(
                    (cell_dir / "subprocess_receipt.json").read_text(encoding="utf-8")
                )
                child["reused_from_structural_failure"] = True
                child_receipts.append(child)
                all_records.extend(validated["records"])
                reused_cells += 1
                print(f"resume reused family={family} runtime={runtime}", flush=True)
                continue
            child, records = _run_new_cell(
                protocol_path=protocol_path,
                lerobot_root=args.lerobot_root.resolve(),
                cell_dir=cell_dir,
                family=family,
                runtime=runtime,
                environment=environment,
            )
            child_receipts.append(child)
            all_records.extend(records)
            executed_cells += 1

    if reused_cells != 1 or executed_cells != 8 or len(all_records) != 45:
        raise RuntimeError(
            "H1-Final structural resume did not preserve the 1+8 cell contract"
        )
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
        "status": "completed_after_structural_resume",
        "experiment_id": "actionstream_transport_h1_final_20260821",
        "hypothesis_id": "transport_h1_final_serialized_delivery",
        "candidate_base_commit": protocols[0][1].raw["candidate"]["base_commit"],
        "execution_git_head": ORIGINAL._git_head(),
        "resume_config_path": str(args.resume_config.resolve()),
        "resume_config_sha256": _sha256(args.resume_config.resolve()),
        "first_process_log_sha256": _sha256(failure_log),
        "process_sessions": 2,
        "gpu_instance_power_cycles_for_holdout": 1,
        "reused_completed_cells": reused_cells,
        "newly_executed_cells": executed_cells,
        "duplicated_scored_episodes": 0,
        "fresh_process_count": 9,
        "record_count": len(all_records),
        "representative_video_count_expected": 9,
        "wall_clock_resume_seconds": time.monotonic() - run_started,
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
    parser.add_argument("--resume-config", type=Path, default=RESUME_CONFIG)
    parser.add_argument("--protocol", type=Path, action="append", required=True)
    parser.add_argument("--lerobot-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    return parser


def main() -> None:
    print(json.dumps(run(_parser().parse_args()), sort_keys=True))


if __name__ == "__main__":
    main()
