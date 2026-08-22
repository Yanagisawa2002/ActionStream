"""Exercise hard deadline, reset, and stop preemption in real child processes."""

from __future__ import annotations

import argparse
import hashlib
import json
import multiprocessing
from pathlib import Path
import time

from actionstream.lerobot_inference import (
    ActionStreamInferenceConfig,
    ActionStreamInferenceEngine,
)


FACTORY = "actionstream.transport_stress_fixture:create_transport"


def _wait_for(predicate, timeout_s: float) -> None:
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        if predicate():
            return
        time.sleep(0.005)
    raise TimeoutError("stress condition was not reached before its deadline")


def _engine(
    telemetry_path: Path, *, request_timeout_s: float
) -> ActionStreamInferenceEngine:
    return ActionStreamInferenceEngine(
        policy=object(),
        preprocessor=None,
        postprocessor=None,
        hw_features={},
        task="transport stress",
        device="cpu",
        robot_type="mock",
        config=ActionStreamInferenceConfig(
            inference_timeout_s=request_timeout_s,
            retry_backoff_s=0.0,
            max_consecutive_failures=100,
            join_timeout_s=2.0,
            transport_mode="process",
            process_transport_factory=FACTORY,
            process_transport_start_method="spawn",
            process_transport_startup_timeout_s=15.0,
            process_transport_terminate_timeout_s=1.0,
            telemetry_jsonl_path=str(telemetry_path),
        ),
        reset_provider=lambda: None,
    )


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def run(output_dir: Path, cycles: int, deadline_s: float) -> dict:
    if cycles <= 0 or deadline_s <= 0:
        raise ValueError("cycles and deadline must be positive")
    output_dir.mkdir(parents=True, exist_ok=False)
    telemetry_path = output_dir / "transport_events.jsonl"

    deadline_engine = _engine(telemetry_path, request_timeout_s=deadline_s)
    deadline_engine.reset()
    deadline_engine.start()
    deadline_engine.resume()
    deadline_started = time.monotonic()
    deadline_engine.notify_observation({"mode": "hang", "value": 1})
    _wait_for(lambda: deadline_engine.telemetry.inference_timeouts >= 1, 20.0)
    enforced_after_s = time.monotonic() - deadline_started
    timeout_telemetry = deadline_engine.telemetry.to_dict()
    deadline_engine.reset()
    deadline_engine.resume()
    deadline_engine.notify_observation({"mode": "ok", "value": 7})
    _wait_for(lambda: deadline_engine.telemetry.chunks_accepted == 1, 20.0)
    recovery_telemetry = deadline_engine.telemetry.to_dict()
    deadline_engine.stop()

    reset_engine = _engine(telemetry_path, request_timeout_s=30.0)
    reset_engine.reset()
    reset_engine.start()
    reset_engine.resume()
    reset_durations: list[float] = []
    for cycle in range(cycles):
        reset_engine.notify_observation({"mode": "hang", "value": cycle})
        _wait_for(
            lambda: reset_engine.telemetry.transport_process_restarts >= cycle + 1,
            20.0,
        )
        started = time.monotonic()
        reset_engine.reset()
        reset_durations.append(time.monotonic() - started)
        reset_engine.resume()
    reset_telemetry = reset_engine.telemetry.to_dict()
    reset_engine.stop()

    stop_engine = _engine(telemetry_path, request_timeout_s=30.0)
    stop_engine.reset()
    stop_engine.start()
    stop_engine.resume()
    stop_engine.notify_observation({"mode": "hang", "value": 99})
    _wait_for(lambda: stop_engine.telemetry.transport_process_restarts >= 1, 20.0)
    stop_started = time.monotonic()
    stop_engine.stop()
    stop_duration = time.monotonic() - stop_started
    stop_telemetry = stop_engine.telemetry.to_dict()

    orphan_names = [
        child.name
        for child in multiprocessing.active_children()
        if child.name == "ActionStreamTransport"
    ]
    summary = {
        "schema_version": 1,
        "status": "pass"
        if (
            timeout_telemetry["inference_timeouts"] >= 1
            and timeout_telemetry["deadline_enforced"] is True
            and timeout_telemetry["transport_request_latency_ms"]
            <= (deadline_s + 0.5) * 1000.0
            and recovery_telemetry["chunks_accepted"] == 1
            and max(reset_durations) < 2.0
            and stop_duration < 2.0
            and not orphan_names
        )
        else "fail",
        "cycles": cycles,
        "deadline_seconds": deadline_s,
        "deadline_observed_after_seconds": enforced_after_s,
        "deadline_startup_seconds": (
            timeout_telemetry["transport_startup_latency_ms"] / 1000.0
        ),
        "deadline_request_seconds": (
            timeout_telemetry["transport_request_latency_ms"] / 1000.0
        ),
        "reset_duration_seconds": reset_durations,
        "reset_duration_max_seconds": max(reset_durations),
        "stop_duration_seconds": stop_duration,
        "timeout_telemetry": timeout_telemetry,
        "recovery_telemetry": recovery_telemetry,
        "reset_telemetry": reset_telemetry,
        "stop_telemetry": stop_telemetry,
        "orphan_transport_children": orphan_names,
        "telemetry_jsonl": str(telemetry_path.resolve()),
        "telemetry_sha256": _sha256(telemetry_path),
    }
    summary_path = output_dir / "summary.json"
    summary_path.write_text(
        json.dumps(summary, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    if summary["status"] != "pass":
        raise RuntimeError(f"transport stress failed: {summary}")
    return summary


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--cycles", type=int, default=10)
    parser.add_argument("--deadline-seconds", type=float, default=0.2)
    return parser


def main() -> None:
    args = _parser().parse_args()
    print(
        json.dumps(
            run(args.output_dir.resolve(), args.cycles, args.deadline_seconds),
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
