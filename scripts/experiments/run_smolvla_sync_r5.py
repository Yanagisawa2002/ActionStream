"""Run the bounded SmolVLA R5 sync-repair hypothesis in fresh processes."""

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
FROZEN_FILES = {
    "current_baselines_sha256": ROOT / "src/actionstream/current_baselines.py",
    "orchestrator_sha256": Path(__file__).resolve(),
    "report_sha256": ROOT / "src/actionstream/smolvla_sync_r5_report.py",
}
PHASE_STATUS = {
    "repair": "development_repair_only",
    "canary": "frozen_sync_canary",
}
PHASE_FAMILIES = {
    "repair": {"spatial", "goal"},
    "canary": {"object", "spatial", "goal"},
}
PHASE_RECORDS = {"repair": 2, "canary": 6}


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


def _validate_protocol(path: Path, phase: str) -> Any:
    protocol = load_protocol(path)
    raw = protocol.raw
    if raw.get("smolvla_sync_protocol_version") != 5:
        raise ValueError(f"{path} is not SmolVLA sync protocol R5")
    if raw.get("protocol_status") != PHASE_STATUS[phase]:
        raise ValueError(f"{path} has the wrong status for phase {phase}")
    if raw.get("runtimes") != ["sync_hold"]:
        raise ValueError(f"{path} must contain only sync_hold")
    if set(protocol.models) != {"smolvla"}:
        raise ValueError(f"{path} must contain only SmolVLA")
    if set(protocol.delays) != {"fixed_0000"}:
        raise ValueError(f"{path} must contain only the zero-delay profile")
    sync_execution = raw.get("sync_execution", {})
    if sync_execution.get("mode") != "receding_horizon":
        raise ValueError("R5 must use receding-horizon sync")
    if int(sync_execution.get("horizon_steps", 0)) != 10:
        raise ValueError("R5 freezes a 10-step sync execution horizon")
    if int(protocol.models["smolvla"].chunk_size) != 50:
        raise ValueError("R5 must retain the checkpoint's 50-step predicted chunk")
    measurement = raw.get("measurement_v2", {})
    if measurement.get("fresh_process_per_runtime") is not True:
        raise ValueError("R5 requires a fresh process for every task-family cell")
    if int(measurement.get("worker_warmup", {}).get("inference_calls", 0)) < 2:
        raise ValueError("R5 requires two actual-worker warmup calls")
    candidate = raw.get("candidate", {})
    mismatches = {
        key: {"expected": candidate.get(key), "actual": _sha256(file_path)}
        for key, file_path in FROZEN_FILES.items()
        if candidate.get(key) != _sha256(file_path)
    }
    if mismatches:
        raise RuntimeError(f"Frozen candidate hash mismatch: {mismatches}")
    return protocol


def _read_records(cell_dir: Path) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    receipt_path = cell_dir / "run_receipt.json"
    metrics_path = cell_dir / "episodes.jsonl"
    if not receipt_path.is_file() or not metrics_path.is_file():
        raise RuntimeError(f"Incomplete child evidence: {cell_dir}")
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    records = [
        json.loads(line)
        for line in metrics_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    return receipt, records


def _validate_cell(cell_dir: Path) -> list[dict[str, Any]]:
    receipt, records = _read_records(cell_dir)
    selection = receipt["selection"]
    if selection["models"] != ["smolvla"] or selection["runtimes"] != ["sync_hold"]:
        raise RuntimeError(f"Child process mixed selections: {selection}")
    if not records or any(row.get("model_key") != "smolvla" for row in records):
        raise RuntimeError(f"Missing SmolVLA records: {cell_dir}")
    if any(row.get("sync_execution_mode") != "receding_horizon" for row in records):
        raise RuntimeError(f"R5 did not execute receding-horizon sync: {cell_dir}")
    if any(int(row.get("sync_execution_horizon_steps", 0)) != 10 for row in records):
        raise RuntimeError(f"R5 execution horizon drifted: {cell_dir}")
    if any(int(row.get("chunk_size", 0)) != 50 for row in records):
        raise RuntimeError(f"R5 predicted chunk size drifted: {cell_dir}")
    if any(row.get("worker_warmup") is None for row in records):
        raise RuntimeError(f"Missing actual-worker warmup: {cell_dir}")
    if any(row.get("inference_requests_per_second") is None for row in records):
        raise RuntimeError(f"Missing request/s: {cell_dir}")
    gpu = receipt.get("system_gpu") or {}
    steady = (gpu.get("phases") or {}).get("steady_state") or {}
    if int(gpu.get("sample_count", 0)) <= 0 or int(gpu.get("error_count", 0)) != 0:
        raise RuntimeError(f"Invalid nvidia-smi evidence: {cell_dir}")
    if int(steady.get("resident_sample_count", 0)) <= 0:
        raise RuntimeError(f"Missing steady-state process residency: {cell_dir}")
    return records


def _repair_passed(repair_root: Path) -> bool:
    receipt_path = repair_root / "orchestrator_receipt.json"
    if not receipt_path.is_file():
        raise RuntimeError(f"Missing repair receipt: {receipt_path}")
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    return bool(
        receipt.get("phase") == "repair"
        and receipt.get("record_count") == PHASE_RECORDS["repair"]
        and receipt.get("gate_passed") is True
    )


def run(args: argparse.Namespace) -> dict[str, Any]:
    output_root = args.output_root.resolve()
    output_root.mkdir(parents=True, exist_ok=False)
    if args.phase == "canary":
        if args.repair_root is None:
            raise ValueError("Canary requires --repair-root")
        if not _repair_passed(args.repair_root.resolve()):
            raise RuntimeError("R5 repair gate did not pass; canary remains sealed")

    protocols = [
        (path.resolve(), _validate_protocol(path.resolve(), args.phase))
        for path in args.protocol
    ]
    families = [str(protocol.raw["task_family"]) for _, protocol in protocols]
    if set(families) != PHASE_FAMILIES[args.phase] or len(families) != len(
        PHASE_FAMILIES[args.phase]
    ):
        raise ValueError(
            f"{args.phase} requires exactly {sorted(PHASE_FAMILIES[args.phase])}; "
            f"got {families}"
        )
    source_commit = str(protocols[0][1].raw["candidate"]["base_commit"])
    environment = os.environ.copy()
    environment["ACTIONSTREAM_SOURCE_COMMIT"] = source_commit
    environment.setdefault("MUJOCO_GL", "egl")
    environment.setdefault("PYOPENGL_PLATFORM", "egl")

    child_receipts: list[dict[str, Any]] = []
    all_records: list[dict[str, Any]] = []
    run_started = time.monotonic()
    for protocol_path, protocol in protocols:
        family = str(protocol.raw["task_family"])
        cell_dir = output_root / "cells" / family / "sync_hold"
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
            "smolvla",
            "--runtimes",
            "sync_hold",
        ]
        if args.capture:
            command.append("--capture")
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
        child = {
            "schema_version": 1,
            "phase": args.phase,
            "family": family,
            "runtime": "sync_hold",
            "fresh_process": True,
            "pid": process.pid,
            "command": command,
            "started_utc_unix_ns": started_utc_ns,
            "wall_clock_seconds": time.monotonic() - started,
            "returncode": returncode,
            "pre_gpu": pre_gpu,
            "post_gpu": _nvidia_snapshot(),
            "log_path": str(log_path),
            "log_sha256": _sha256(log_path),
        }
        _write_json(cell_dir / "subprocess_receipt.json", child)
        child_receipts.append(child)
        print(
            f"{args.phase} family={family} exit={returncode} "
            f"seconds={child['wall_clock_seconds']:.1f}",
            flush=True,
        )
        if returncode != 0:
            _write_json(
                output_root / "orchestrator_failed.json",
                {"status": "failed", "children": child_receipts},
            )
            raise RuntimeError(f"Child failed; inspect {log_path}")
        all_records.extend(_validate_cell(cell_dir))

    episodes_path = output_root / "episodes.jsonl"
    episodes_path.write_text(
        "".join(
            json.dumps(row, sort_keys=True, allow_nan=False) + "\n"
            for row in all_records
        ),
        encoding="utf-8",
        newline="\n",
    )
    expected = PHASE_RECORDS[args.phase]
    if len(all_records) != expected:
        raise RuntimeError(f"Expected {expected} records, got {len(all_records)}")
    gate_passed = all(bool(row.get("success")) for row in all_records)
    receipt = {
        "schema_version": 1,
        "status": "PASS" if gate_passed else "NO_GO",
        "phase": args.phase,
        "hypothesis_id": "SMOLVLA_SYNC_R5_HORIZON_10",
        "source_commit": source_commit,
        "protocols": [
            {"path": str(path), "sha256": _sha256(path)} for path, _ in protocols
        ],
        "families": families,
        "fresh_process_count": len(child_receipts),
        "fresh_process_pids": [item["pid"] for item in child_receipts],
        "record_count": len(all_records),
        "success_count": sum(bool(row.get("success")) for row in all_records),
        "gate_passed": gate_passed,
        "formal_rtc_status": (
            "ELIGIBLE_FOR_SEPARATE_FREEZE"
            if args.phase == "canary" and gate_passed
            else "NOT_RUN_BY_SYNC_GATE"
        ),
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
    parser.add_argument("--phase", choices=tuple(PHASE_STATUS), required=True)
    parser.add_argument("--protocol", type=Path, action="append", required=True)
    parser.add_argument("--lerobot-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--repair-root", type=Path)
    parser.add_argument("--capture", action="store_true")
    return parser


def main() -> None:
    print(json.dumps(run(_parser().parse_args()), sort_keys=True))


if __name__ == "__main__":
    main()
