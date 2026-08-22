"""Run the frozen H1-R2 single-hypothesis learned-policy GPU holdout."""

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


ROOT = Path(__file__).resolve().parents[2]
EXPECTED_LEROBOT_COMMIT = "73e1584473028a2d53ecfc856f5290db84507f90"
EXPECTED_LEROBOT_ORIGIN = "https://github.com/Yanagisawa2002/lerobot.git"
EXPECTED_OUTPUT_BASENAME = "actionstream_transport_h1_r2_holdout_20260822"
EXPECTED_PROFILE = "fixed_0950_transport_h1_r2"
EXPECTED_STATES = [10, 11, 12, 13, 14]
ALLOWED_RUNTIMES = {
    "actionstream_backend_aligned",
    "actionstream_backend_pipelined_aligned",
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
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
        newline="\n",
    )


def _git(root: Path, *args: str) -> str:
    completed = subprocess.run(
        ["git", *args],
        cwd=root,
        check=True,
        capture_output=True,
        text=True,
        timeout=20,
    )
    return completed.stdout.strip()


def _pythonpath(lerobot_root: Path, inherited: str | None) -> str:
    entries = [str((lerobot_root / "src").resolve()), str((ROOT / "src").resolve())]
    if inherited:
        entries.extend(item for item in inherited.split(os.pathsep) if item)
    return os.pathsep.join(dict.fromkeys(entries))


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


def _exact_import_preflight(
    *, python: Path, lerobot_root: Path, environment: dict[str, str]
) -> dict[str, Any]:
    commit = _git(lerobot_root, "rev-parse", "HEAD")
    origin = _git(lerobot_root, "remote", "get-url", "origin")
    dirty = _git(lerobot_root, "status", "--porcelain")
    if commit != EXPECTED_LEROBOT_COMMIT:
        raise RuntimeError(f"Wrong LeRobot checkout: {commit}")
    if origin.rstrip("/") != EXPECTED_LEROBOT_ORIGIN.rstrip("/"):
        raise RuntimeError(f"Wrong LeRobot origin: {origin}")
    if dirty:
        raise RuntimeError("Frozen LeRobot checkout is dirty")
    code = """
import inspect
import json
import actionstream.current_baselines as current_baselines
import actionstream.lerobot_inference as actionstream_inference
from lerobot.rollout.inference.base import InferenceEngine
print(json.dumps({
    "inference_engine_file": inspect.getfile(InferenceEngine),
    "inference_engine_constructor": str(inspect.signature(InferenceEngine)),
    "current_baselines_file": inspect.getfile(current_baselines),
    "actionstream_inference_file": inspect.getfile(actionstream_inference),
}))
"""
    completed = subprocess.run(
        [str(python), "-c", code],
        check=True,
        capture_output=True,
        text=True,
        env=environment,
        timeout=120,
    )
    result = json.loads(completed.stdout.strip().splitlines()[-1])
    inference_file = Path(result["inference_engine_file"]).resolve()
    current_file = Path(result["current_baselines_file"]).resolve()
    actionstream_file = Path(result["actionstream_inference_file"]).resolve()
    if not inference_file.is_relative_to((lerobot_root / "src").resolve()):
        raise RuntimeError(f"LeRobot import escaped frozen checkout: {inference_file}")
    if not current_file.is_relative_to((ROOT / "src").resolve()):
        raise RuntimeError(f"ActionStream import escaped execution checkout: {current_file}")
    if not actionstream_file.is_relative_to((ROOT / "src").resolve()):
        raise RuntimeError(f"Inference backend escaped execution checkout: {actionstream_file}")
    if "task" not in result["inference_engine_constructor"]:
        raise RuntimeError(f"LeRobot InferenceEngine lacks task contract: {result}")
    return {
        **result,
        "checkout_commit": commit,
        "checkout_origin": origin,
        "checkout_clean": True,
    }


def _validate_protocol(path: Path) -> dict[str, Any]:
    raw = json.loads(path.read_text(encoding="utf-8"))
    if raw.get("protocol_status") != "frozen_holdout":
        raise ValueError(f"{path} is not a frozen holdout")
    if raw.get("benchmark_version") != 3:
        raise ValueError(f"{path} is not GPU benchmark v3")
    if raw.get("hypothesis_id") != "transport_h1_r2_delivery_pipeline":
        raise ValueError(f"Unexpected hypothesis in {path}")
    runtimes = list(raw.get("runtimes", []))
    order = list(raw.get("runtime_execution_order", []))
    if set(runtimes) != ALLOWED_RUNTIMES or len(runtimes) != 2:
        raise ValueError(f"Unexpected H1-R2 runtimes: {runtimes}")
    if set(order) != ALLOWED_RUNTIMES or len(order) != 2:
        raise ValueError(f"Invalid H1-R2 execution order: {order}")
    if raw.get("delay_profiles") != [
        {"key": EXPECTED_PROFILE, "kind": "fixed", "milliseconds": 950}
    ]:
        raise ValueError("H1-R2 must contain only the namespaced fixed 950 ms profile")
    environment = raw.get("environment", {})
    if environment.get("initial_state_indices") != EXPECTED_STATES:
        raise ValueError("H1-R2 scored reset identities changed")
    if raw.get("compact_matrix") != {
        "episodes_per_task": 5,
        "paired": True,
        "reuse_delay_trace_across_runtimes": True,
    }:
        raise ValueError("H1-R2 compact matrix changed")
    measurement = raw.get("measurement_v2", {})
    if measurement.get("fresh_process_per_runtime") is not True:
        raise ValueError("H1-R2 requires a fresh process per runtime and family")
    if int(measurement.get("worker_warmup", {}).get("inference_calls", 0)) != 2:
        raise ValueError("H1-R2 freezes exactly two actual-worker warmup calls")
    if int(measurement.get("worker_warmup", {}).get("initial_state_index", -1)) != 45:
        raise ValueError("H1-R2 non-scored warmup reset changed")
    serialized = dict(
        raw.get("actionstream_backend", {}).get("actionstream_backend_aligned", {})
    )
    pipelined = dict(
        raw.get("actionstream_backend", {}).get(
            "actionstream_backend_pipelined_aligned", {}
        )
    )
    if serialized.pop("delivery_scheduler_enabled", None) is not False:
        raise ValueError("Serialized H1-R2 delivery scheduler must be disabled")
    if pipelined.pop("delivery_scheduler_enabled", None) is not True:
        raise ValueError("Pipelined H1-R2 delivery scheduler must be enabled")
    if serialized != pipelined or serialized != {"latest_only_fallback": False}:
        raise ValueError("Delivery scheduling must be the only runtime difference")
    candidate = raw.get("candidate", {})
    mismatches = {
        key: {"expected": candidate.get(key), "actual": _sha256(file_path)}
        for key, file_path in FROZEN_FILES.items()
        if candidate.get(key) != _sha256(file_path)
    }
    if mismatches:
        raise RuntimeError(f"Frozen H1-R2 candidate hash mismatch: {mismatches}")
    return raw


def _validate_cell(
    cell_dir: Path, runtime: str, protocol: dict[str, Any]
) -> dict[str, Any]:
    receipt_path = cell_dir / "run_receipt.json"
    if not receipt_path.is_file():
        raise RuntimeError(f"Missing child receipt: {receipt_path}")
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    selection = receipt.get("selection", {})
    if selection.get("runtimes") != [runtime]:
        raise RuntimeError(f"Child process mixed runtimes: {selection}")
    if selection.get("profiles") != [EXPECTED_PROFILE]:
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
        raise RuntimeError(f"Expected five H1-R2 rows in {cell_dir}")
    expected_task = int(protocol["environment"]["task_ids"][0])
    expected_suite = str(protocol["environment"]["suite"])
    expected_seed = int(protocol["environment"]["base_seed"])
    for index, row in enumerate(records):
        identity = (
            str(row.get("suite")),
            int(row.get("task_id", -1)),
            int(row.get("initial_state_index", -1)),
            int(row.get("seed", -1)),
        )
        expected_identity = (
            expected_suite,
            expected_task,
            EXPECTED_STATES[index],
            expected_seed + index,
        )
        if identity != expected_identity:
            raise RuntimeError(
                f"Unexpected scored identity in {cell_dir}: {identity} != {expected_identity}"
            )
        if row.get("runtime") != runtime:
            raise RuntimeError(f"Runtime mismatch in {cell_dir}: {row.get('runtime')}")
        if row.get("worker_warmup") is None:
            raise RuntimeError(f"Missing actual-worker warmup provenance: {cell_dir}")
        if row.get("inference_requests_per_second") is None:
            raise RuntimeError(f"Missing true request/s: {cell_dir}")
        for field in (
            "depletion_safe_hold_steps",
            "responses_rejected_out_of_order",
            "fallback_activations",
        ):
            if row.get(field) is None:
                raise RuntimeError(f"Missing {field} in {cell_dir}")
        engine_config = row.get("engine_config") or {}
        if engine_config.get("latest_only_fallback") is not False:
            raise RuntimeError(f"Fallback was not disabled in {cell_dir}")
    trace_count = len(list((cell_dir / "traces").glob("*.json")))
    telemetry_count = len(list((cell_dir / "traces").glob("*.telemetry.jsonl")))
    video_count = len(list((cell_dir / "videos").glob("*.mp4")))
    if (trace_count, telemetry_count, video_count) != (5, 5, 1):
        raise RuntimeError(
            f"Incomplete H1-R2 artifacts in {cell_dir}: "
            f"traces={trace_count}, telemetry={telemetry_count}, videos={video_count}"
        )
    return {"receipt": receipt, "records": records}


def _wait_child(process: subprocess.Popen[str], timeout_s: float) -> tuple[int, bool]:
    try:
        return process.wait(timeout=timeout_s), False
    except subprocess.TimeoutExpired:
        process.terminate()
        try:
            return process.wait(timeout=10), True
        except subprocess.TimeoutExpired:
            process.kill()
            return process.wait(timeout=10), True


def run(args: argparse.Namespace) -> dict[str, Any]:
    protocol_rows = [(path.resolve(), _validate_protocol(path.resolve())) for path in args.protocol]
    families = [str(raw["task_family"]) for _, raw in protocol_rows]
    if len(families) != 3 or len(set(families)) != 3:
        raise ValueError("H1-R2 requires exactly three task families")
    if args.capture is not True:
        raise ValueError("H1-R2 requires paired representative video capture")
    output_root = args.output_root.resolve()
    if output_root.name != EXPECTED_OUTPUT_BASENAME:
        raise ValueError(
            f"H1-R2 output basename must be {EXPECTED_OUTPUT_BASENAME}, got {output_root.name}"
        )
    if output_root.exists():
        raise FileExistsError(f"One-shot H1-R2 output already exists: {output_root}")
    lerobot_root = args.lerobot_root.resolve()
    environment = os.environ.copy()
    environment["PYTHONPATH"] = _pythonpath(lerobot_root, environment.get("PYTHONPATH"))
    environment["ACTIONSTREAM_SOURCE_COMMIT"] = str(
        protocol_rows[0][1]["candidate"]["base_commit"]
    )
    environment.setdefault("MUJOCO_GL", "egl")
    environment.setdefault("PYOPENGL_PLATFORM", "egl")
    environment.setdefault("HF_HUB_OFFLINE", "1")
    preflight = _exact_import_preflight(
        python=Path(sys.executable).resolve(),
        lerobot_root=lerobot_root,
        environment=environment,
    )

    output_root.mkdir(parents=True, exist_ok=False)
    _write_json(output_root / "exact_import_preflight.json", preflight)
    execution_head = _git(ROOT, "rev-parse", "HEAD")
    child_receipts: list[dict[str, Any]] = []
    all_records: list[dict[str, Any]] = []
    run_started = time.monotonic()
    for protocol_path, protocol in protocol_rows:
        family = str(protocol["task_family"])
        timeout_s = float(protocol["measurement_v2"]["cell_process_timeout_s"])
        for runtime in protocol["runtime_execution_order"]:
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
                returncode, timed_out = _wait_child(process, timeout_s)
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
                "process_timeout_seconds": timeout_s,
                "timed_out": timed_out,
                "returncode": returncode,
                "pre_gpu": pre_gpu,
                "post_gpu": post_gpu,
                "log_path": str(log_path),
                "log_sha256": _sha256(log_path),
            }
            _write_json(cell_dir / "subprocess_receipt.json", child)
            child_receipts.append(child)
            print(
                f"family={family} runtime={runtime} exit={returncode} "
                f"timeout={timed_out} seconds={child['wall_clock_seconds']:.1f}",
                flush=True,
            )
            if returncode != 0 or timed_out:
                _write_json(
                    output_root / "orchestrator_failed.json",
                    {
                        "status": "failed",
                        "execution_git_head": execution_head,
                        "children": child_receipts,
                    },
                )
                raise RuntimeError(f"Child failed; inspect {log_path}")
            validated = _validate_cell(cell_dir, runtime, protocol)
            all_records.extend(validated["records"])

    if len(all_records) != 30:
        raise RuntimeError(f"Expected 30 H1-R2 rows, got {len(all_records)}")
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
        "experiment_id": "actionstream_transport_h1_r2_20260822",
        "hypothesis_id": "transport_h1_r2_delivery_pipeline",
        "candidate_base_commit": protocol_rows[0][1]["candidate"]["base_commit"],
        "execution_git_head": execution_head,
        "exact_import_preflight": preflight,
        "protocols": [
            {"path": str(path), "sha256": _sha256(path)} for path, _ in protocol_rows
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
