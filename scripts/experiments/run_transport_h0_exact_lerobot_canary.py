"""Run one fail-closed GPU canary against an exact LeRobot source checkout."""

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
DEFAULT_PROTOCOL = ROOT / "configs/actionstream_transport_h0_exact_lerobot_canary.json"
EXPECTED_COMMIT = "73e1584473028a2d53ecfc856f5290db84507f90"
EXPECTED_RUNTIME = "actionstream_backend_pipelined_aligned"
EXPECTED_PROFILE = "fixed_0950_transport_h0_canary"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
        newline="\n",
    )


def _pythonpath(lerobot_root: Path, inherited: str | None) -> str:
    entries = [str((lerobot_root / "src").resolve()), str((ROOT / "src").resolve())]
    if inherited:
        entries.extend(item for item in inherited.split(os.pathsep) if item)
    return os.pathsep.join(dict.fromkeys(entries))


def _validate_protocol(path: Path) -> dict[str, Any]:
    raw = json.loads(path.read_text(encoding="utf-8"))
    if raw.get("protocol_status") != "frozen_prerequisite_canary":
        raise ValueError("H0 canary must be frozen before execution")
    if raw.get("hypothesis_id") != "transport_h0_exact_lerobot_worker_binding":
        raise ValueError("H0 canary hypothesis changed")
    if raw.get("runtimes") != [EXPECTED_RUNTIME]:
        raise ValueError("H0 canary must exercise only the pipelined backend")
    if raw.get("environment", {}).get("initial_state_indices") != [47]:
        raise ValueError("H0 canary must consume only development reset 47")
    profiles = raw.get("delay_profiles", [])
    if profiles != [{"key": EXPECTED_PROFILE, "kind": "fixed", "milliseconds": 950}]:
        raise ValueError("H0 canary delay changed")
    if raw.get("compact_matrix") != {
        "episodes_per_task": 1,
        "paired": False,
        "reuse_delay_trace_across_runtimes": False,
    }:
        raise ValueError("H0 canary must remain one unpaired episode")
    if raw.get("source", {}).get("commit") != EXPECTED_COMMIT:
        raise ValueError("H0 canary LeRobot commit changed")
    return raw


def _exact_import_preflight(
    *, python: Path, lerobot_root: Path, environment: dict[str, str]
) -> dict[str, Any]:
    commit = subprocess.run(
        ["git", "-C", str(lerobot_root), "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    if commit != EXPECTED_COMMIT:
        raise RuntimeError(f"Wrong LeRobot checkout: {commit}")
    code = """
import inspect
import json
from lerobot.rollout.inference.base import InferenceEngine
print(json.dumps({
    "class_file": inspect.getfile(InferenceEngine),
    "constructor": str(inspect.signature(InferenceEngine)),
    "init_constructor": str(inspect.signature(InferenceEngine.__init__)),
}))
"""
    completed = subprocess.run(
        [str(python), "-c", code],
        check=True,
        capture_output=True,
        text=True,
        env=environment,
    )
    result = json.loads(completed.stdout.strip().splitlines()[-1])
    class_file = Path(result["class_file"]).resolve()
    expected_source = (lerobot_root / "src").resolve()
    if not class_file.is_relative_to(expected_source):
        raise RuntimeError(f"LeRobot import escaped frozen checkout: {class_file}")
    if "task" not in result["constructor"]:
        raise RuntimeError(f"LeRobot InferenceEngine lacks task contract: {result}")
    return {**result, "checkout_commit": commit}


def _validate_result(output_root: Path) -> dict[str, Any]:
    receipt = json.loads((output_root / "run_receipt.json").read_text(encoding="utf-8"))
    selection = receipt.get("selection", {})
    if selection.get("runtimes") != [EXPECTED_RUNTIME]:
        raise RuntimeError(f"Canary used wrong runtime: {selection}")
    if selection.get("profiles") != [EXPECTED_PROFILE]:
        raise RuntimeError(f"Canary used wrong profile: {selection}")
    rows = [
        json.loads(line)
        for line in (output_root / "episodes.jsonl")
        .read_text(encoding="utf-8")
        .splitlines()
        if line.strip()
    ]
    if len(rows) != 1:
        raise RuntimeError(f"H0 canary produced {len(rows)} rows instead of one")
    row = rows[0]
    gates = {
        "episode_completed": row.get("status") == "completed",
        "exact_runtime": row.get("runtime") == EXPECTED_RUNTIME,
        "inference_executed": int(row.get("inference_completed", 0)) > 0,
        "environment_advanced": int(row.get("environment_steps", 0)) > 0,
        "nonconstant_actions": float(row.get("action_discontinuity_max_l2", 0.0)) > 0.0,
        "delivery_scheduled": int(row.get("responses_scheduled", 0)) > 0,
        "delivery_delivered": int(row.get("responses_delivered", 0)) > 0,
        "no_inference_error": int(row.get("inference_errors", 0)) == 0,
    }
    gpu = receipt.get("system_gpu") or {}
    steady = (gpu.get("phases") or {}).get("steady_state") or {}
    gates.update(
        {
            "gpu_sampled": int(gpu.get("sample_count", 0)) > 0,
            "gpu_process_resident": int(steady.get("resident_sample_count", 0)) > 0,
            "gpu_vram_recorded": steady.get("process_gpu_memory_mib_max") is not None,
        }
    )
    trace_path = Path(str(row.get("trace_path", "")))
    telemetry_path = Path(str(row.get("telemetry_jsonl_path", "")))
    video_path = Path(str(row.get("video_path", "")))
    gates.update(
        {
            "trace_present": trace_path.is_file(),
            "telemetry_present": telemetry_path.is_file(),
            "video_present": video_path.is_file(),
        }
    )
    return {
        "status": "PASS" if all(gates.values()) else "FAIL",
        "gates": gates,
        "task_success_observed_not_gated": bool(row.get("success")),
        "record": row,
        "system_gpu": gpu,
    }


def run(args: argparse.Namespace) -> dict[str, Any]:
    protocol_path = args.protocol.resolve()
    protocol = _validate_protocol(protocol_path)
    output_root = args.output_root.resolve()
    if output_root.exists():
        raise FileExistsError(
            f"One-shot H0 canary output already exists: {output_root}"
        )
    lerobot_root = args.lerobot_root.resolve()
    environment = os.environ.copy()
    environment["PYTHONPATH"] = _pythonpath(lerobot_root, environment.get("PYTHONPATH"))
    environment["ACTIONSTREAM_SOURCE_COMMIT"] = str(
        protocol["candidate"]["base_commit"]
    )
    environment.setdefault("MUJOCO_GL", "egl")
    environment.setdefault("PYOPENGL_PLATFORM", "egl")
    environment.setdefault("HF_HUB_OFFLINE", "1")
    preflight = _exact_import_preflight(
        python=Path(sys.executable).resolve(),
        lerobot_root=lerobot_root,
        environment=environment,
    )
    command = [
        sys.executable,
        "-m",
        "actionstream.current_baselines",
        "--protocol",
        str(protocol_path),
        "--lerobot-root",
        str(lerobot_root),
        "--output-dir",
        str(output_root),
        "--models",
        "xvla",
        "--runtimes",
        EXPECTED_RUNTIME,
        "--capture",
    ]
    started = time.monotonic()
    completed = subprocess.run(command, env=environment, check=False)
    receipt: dict[str, Any] = {
        "schema_version": 1,
        "hypothesis_id": protocol["hypothesis_id"],
        "protocol_path": str(protocol_path),
        "protocol_sha256": _sha256(protocol_path),
        "orchestrator_sha256": _sha256(Path(__file__).resolve()),
        "exact_import_preflight": preflight,
        "command": command,
        "returncode": completed.returncode,
        "wall_clock_seconds": time.monotonic() - started,
        "formal_h1_reopened": False,
    }
    if completed.returncode == 0:
        receipt["result"] = _validate_result(output_root)
        receipt["status"] = receipt["result"]["status"]
    else:
        receipt["status"] = "ERROR"
    _write_json(output_root / "h0_canary_receipt.json", receipt)
    if receipt["status"] != "PASS":
        raise RuntimeError(
            f"H0 exact-checkout canary did not pass: {receipt['status']}"
        )
    return receipt


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", type=Path, default=DEFAULT_PROTOCOL)
    parser.add_argument("--lerobot-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    return parser


def main() -> None:
    print(json.dumps(run(_parser().parse_args()), sort_keys=True))


if __name__ == "__main__":
    main()
